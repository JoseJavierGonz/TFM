import sys
import os
import carla
import gym
from gym import spaces
import time
import numpy as np
from env.carlaControler import CarlaControler


class envCARLA(gym.Env):
    """Class to create a gym env, where implement the steps, rewards and so on"""
    def __init__(self):
        self.current_step = 0
        self.max_steps = 1000 
        self.action_space =  [
            spaces.Box(low=np.array([0.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32),
            spaces.Box(low=np.array([0.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32) 
        ]
        vehicle_obs_space = spaces.Box(
            low = np.array([-np.inf]*10),
            high = np.array([np.inf]*10),
            dtype = np.float32
        )
        camera_obs = spaces.Box(
            low = 0,
            high = 255, 
            shape = (84, 84, 3), 
            dtype = np.uint8)

        lidar_obs = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1000, 4),
            dtype=np.float32
        )

        self.observation_space = [
            spaces.Dict({
                "vehicle_state": vehicle_obs_space
                # "camera": camera_obs,
                # "lidar": lidar_obs
            }),
            spaces.Dict({
                "vehicle_state": vehicle_obs_space
                # "camera": camera_obs,
                # "lidar": lidar_obs
            }),
        ]

        self.__agent=[]
        self.agent_id=[]
        self.CARLA = CarlaControler()
        self.position_change = {}
        self.goal_positions = {}
        self.better_distance = {}

        for i, vehicle in enumerate(self.CARLA.vehicles_marl_list):
            self.__agent.append(vehicle)
            agent_id = f"agent_{i}"
            self.agent_id.append(agent_id)
            self.better_distance[agent_id] = np.inf




    def step(self, action):
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0
        
        for i, vehicle in enumerate(self.__agent):
            throttle = float(action[i][0])
            steer = float(action[i][1])
            brake = float(action[i][2])
            
            #imprimir acciones cada 100 pasos
            if self._step_counter % 100 == 0:
                agent_id = self.agent_id[i]
                print(f"Action {agent_id}: throttle={throttle:.3f}, steer={steer:.3f}, brake={brake:.3f}")

            move = carla.VehicleControl(throttle, steer, brake)
            vehicle.apply_control(move)
        
        self._step_counter += 1
        self.CARLA.tick()
        
        observations = self.__get_obs()
        rewards, dones = self.__calculate_rewards(observations, action)
        self.current_step += 1

        return observations, rewards, dones, {}




    def __get_obs(self):
        observation = {}
        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            velocity = agent.get_velocity()
            acceleration = agent.get_acceleration()

            transform = agent.get_transform()
            vehicle_location = transform.location
            vehicle_angle = transform.rotation.yaw
            
            waypoint = self.CARLA.get_map().get_waypoint(transform.location)
            lane_center = waypoint.transform.location
            angle_center = waypoint.transform.rotation.yaw

            distance_x = vehicle_location.x - lane_center.x
            distance_y = vehicle_location.y - lane_center.y
            lane_direction = np.array([np.cos(np.radians(angle_center)), np.sin(np.radians(angle_center))])
            lateral_distance = - distance_x * lane_direction[1] + distance_y * lane_direction[0]

            angular_diff = (vehicle_angle - angle_center + 180) % 360 - 180

            goal_location = self.goal_positions[agent_id].location
            dist_to_goal = vehicle_location.distance(goal_location)


            vehicle_state = np.array([velocity.x, velocity.y, velocity.z, #Velocidad
                                    acceleration.x, acceleration.y, acceleration.z, #Aceleración
                                    transform.rotation.yaw, #Orientacion
                                    lateral_distance, #desviacion lateral
                                    angular_diff, #desviacion angular
                                    dist_to_goal],
                                    dtype=np.float32) 
            

            # sensor_obs = self.CARLA.get_sensor_data(agent)
            # lidar_data = sensor_obs['lidar_data']
            # if lidar_data is None:
            #     lidar_data = np.zeros((1000, 4), dtype=np.float32)

            # camera_data = sensor_obs['camera_data']
            # if camera_data is None:
            #     camera_data = np.zeros((84, 84, 3), dtype=np.uint8)

            obs = {
                "vehicle_state": vehicle_state
                # "camera": camera_data,
                # "lidar": lidar_data
            }
            
            observation[agent_id] = obs



        return observation

    
        
    def __calculate_rewards(self, observations, actions):
        rewards = {}
        dones = {}
        factor = 0.1
        max_speed = 8.5
        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            done = False
            reward = 0

            vehicle_state = observations[agent_id]['vehicle_state']
            dist_to_goal = vehicle_state[9]
            velocity = np.linalg.norm(vehicle_state[:3])
            lateral_distance = abs(vehicle_state[7])
            angular_diff = abs(vehicle_state[8])
            wrong_direction = 5 if angular_diff > 90 else 0
            #recompensa por velociad alta(mirar documentacion de si puedo obtener la velocidad maxima del carril para actualizar max_speed)
            speed_error = abs(velocity - max_speed)
            reward += (max_speed - speed_error) * factor
            
            #penalizacion extra si velocidad es 0 (no se mueve)
            if velocity < 0.1:  # umbral pequeño para considerar parado
                reward -= 50.0
            elif 0.1 < velocity <= 1:
                reward += 5.0
            elif 1 < velocity <= 2:
                reward += 10
            else:
                reward+=20
            reward_for_goal = -3 if self.better_distance[agent_id] < dist_to_goal else 8
            self.better_distance[agent_id] = min(self.better_distance[agent_id], dist_to_goal)
            #recompensa negativa cuanto mas distancia lateral al centro del carril tenga y mas desviacion del mismo
            reward_lat = 3.0 if lateral_distance < 1.0 else 0
            reward -= lateral_distance * 1.5 + angular_diff * 0.05 + wrong_direction + reward_for_goal
            reward += reward_lat
            if dist_to_goal < 5.0:
                reward += 200
                done = True
            if self.CARLA.collision_occurs[agent]:
                reward -= 50
                done = True
            
            if lateral_distance > 4:
                reward -= 20
                done = True
                
            if self.current_step >= self.max_steps:
                done = True
            
            rewards[agent_id] = reward
            dones[agent_id] = done
        
        dones["__all__"] = all(dones.values())
        return rewards, dones
    

    def reset(self, agent_ids=None, same_position=False):
        """Reset environment.reset only specified agents."""
        if agent_ids is None:
            self.current_step = 0
            agent_ids = self.agent_id
        
        for agent_id in agent_ids:
            agent_idx = self.agent_id.index(agent_id)
            agent = self.__agent[agent_idx]
            
            self.CARLA.reset_collision(agent)
            if agent in self.CARLA.sensors_data:
                self.CARLA.sensors_data[agent] = {'camera_data': None, 'lidar_data': None}
            
            #spawn point aleatorio en diferentes episodios, el mismo en cada paso
            if not same_position:
                spawn_points = self.CARLA.world.get_map().get_spawn_points()
                if spawn_points:
                    self.position_change[agent_idx] = np.random.choice(spawn_points)
                    available_goals = [sp for sp in spawn_points if sp != self.position_change[agent_idx]]
                    self.goal_positions[agent_id] = np.random.choice(available_goals)
                    agent.set_transform(self.position_change[agent_idx])
                    self.better_distance[agent_id] = np.inf
                    
            else:
                agent.set_transform(self.position_change[agent_idx])
            
            agent.set_target_velocity(carla.Vector3D(0, 0, 0))

        
        self.CARLA.vehicles_npcs_list = [v for v in self.CARLA.vehicles_npcs_list if v.is_alive]
        if len(self.CARLA.vehicles_npcs_list) < 5:
            self.CARLA.spawn_vehicle(True)
        
        return self.__get_obs()
    
    def close(self):
        self.CARLA.destroy_actors()

