import carla
import gym
from gym import spaces
import time
import numpy as np
from carlaControler import CarlaControler


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
            low = np.array([-np.inf]*9),
            high = np.array([np.inf]*9),
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
                "vehicle_state": vehicle_obs_space,
                "camera": camera_obs,
                "lidar": lidar_obs
            }),
            spaces.Dict({
                "vehicle_state": vehicle_obs_space,
                "camera": camera_obs,
                "lidar": lidar_obs
            }),
        ]

        self.__agent=[]
        self.agent_id=[]
        self.CARLA = CarlaControler()

        for i, vehicle in enumerate(self.CARLA.vehicles_marl_list):
            self.__agent.append(vehicle)
            self.agent_id.append(f"agent_{i}")




    def step(self, action):
        for i, vehicle in enumerate(self.__agent):
            throttle = float(action[i][0])
            if throttle > 1.0:
                throttle = 1.0

            steer = float(action[i][1])
            if abs(steer) > 1.0:
                steer = 1.0 if steer > 0 else -1.0

            brake = float(action[i][2])
            if brake > 1.0:
                brake = 1.0

            move = carla.VehicleControl(throttle, steer, brake)
            vehicle.apply_control(move)
        self.CARLA.tick()
        time.sleep(0.05)
        
        observations = self.__get_obs()
        rewards, dones = self.__calculate_rewards(observations)
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


            vehicle_state = np.array([velocity.x, velocity.y, velocity.z, #Velocidad
                                    acceleration.x, acceleration.y, acceleration.z, #Aceleración
                                    transform.rotation.yaw, #Orientacion
                                    lateral_distance, #desviacion lateral
                                    angular_diff], #desviacion angular
                                    dtype=np.float32) 
            

            sensor_obs = self.CARLA.get_sensor_data(agent)
            lidar_data = sensor_obs['lidar_data']
            if lidar_data is None:
                lidar_data = np.zeros((1000, 4), dtype=np.float32)

            camera_data = sensor_obs['camera_data']
            if camera_data is None:
                camera_data = np.zeros((84, 84, 3), dtype=np.uint8)

            obs = {
                "vehicle_state": vehicle_state,
                "camera": camera_data,
                "lidar": lidar_data
            }
            
            observation[agent_id] = obs



        return observation

    
        
    def __calculate_rewards(self, observations):
        rewards = {}
        dones = {}
        factor = 0.1
        max_speed = 8.5
        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            done = False
            reward = 0

            vehicle_state = observations[agent_id]['vehicle_state']
            velocity = np.linalg.norm(vehicle_state[:3])
            lateral_distance = abs(vehicle_state[7])
            angular_diff = abs(vehicle_state[8])
            #recompensa por velociad alta(mirar documentacion de si puedo obtener la velocidad maxima del carril para actualizar max_speed)
            speed_error = abs(velocity - max_speed)
            reward += (max_speed - speed_error) * factor
            #recompensa negativa cuanto mas distancia lateral al centro del carril tenga y mas desviacion del mismo
            reward -= lateral_distance * 1.5 + angular_diff * 0.05


            if self.CARLA.collision_occurs[agent]:
                reward -= 100
                done = True
            
            if lateral_distance > 2:
                reward -= 50
                done = True
                
            if self.current_step >= self.max_steps:
                done = True
            
            rewards[agent_id] = reward
            dones[agent_id] = done
        
        dones["__all__"] = all(dones.values())
        return rewards, dones
    

    def reset(self):
        self.current_step = 0

        for agent in self.__agent:
            self.CARLA.reset_collision(agent)
            if agent in self.CARLA.sensors_data:
                self.CARLA.sensors_data[agent] = {'camera_data': None, 'lidar_data': None}

        self.CARLA.world.tick()
        time.sleep(0.1)

        return self.__get_obs()
    
    def close(self):
        self.CARLA.destroy_actors()