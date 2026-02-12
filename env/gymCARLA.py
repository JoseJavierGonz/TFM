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

            wp = waypoint
            goal_location = self.goal_positions[agent_id].location
            route = 0
            for _ in range(100):
                next_wp = wp.next(2.0)
                if not next_wp:
                    break
                wp = next_wp[0]
                route +=2
                if wp.transform.location.distance(goal_location) < 5.0:
                    route += wp.transform.location.distance(goal_location)
                    break
            dist_to_goal = route



            vehicle_state = np.array([velocity.x/10, velocity.y/10, velocity.z/10, #Velocidad
                                    acceleration.x/2, acceleration.y/2, acceleration.z/2, #Aceleración
                                    transform.rotation.yaw/180, #Orientacion
                                    lateral_distance/2, #desviacion lateral
                                    angular_diff/180, #desviacion angular
                                    dist_to_goal/400], #distancia al destino fijado
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
        factor = 0.3
        max_speed = 10.0
        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            done = False
            reward = 0

            throttle = float(actions[i][0])
            steer = float(actions[i][1])
            brake = float(actions[i][2])

            vehicle_state = observations[agent_id]['vehicle_state']
            dist_to_goal = vehicle_state[9]*400
            velocity = np.linalg.norm(vehicle_state[:3])*10
            lateral_distance = abs(vehicle_state[7]*2)
            angular_diff = abs(vehicle_state[8]*180)
            wrong_direction = 5 if angular_diff > 90 else 0
            #recompensa por velociad alta(mirar documentacion de si puedo obtener la velocidad maxima del carril para actualizar max_speed)
            speed_error = abs(velocity - max_speed)


            #EMPEZAMOS CON LAS RECOMPENSAS
            reward += (max_speed - speed_error) * factor
          
            #penalizacion extra si velocidad es 0 (no se mueve)
            if velocity < 1:
                reward -= 5.0/(velocity+0.1)
            if 1 <= velocity <= 7:
                reward += 5*velocity
            else:
                reward += velocity

            #penalizacion por alejarse del centro, recompensa por ir centrado
            #tambien tenemos en cuenta el angulo
            reward += 1.5/(lateral_distance+0.1) if lateral_distance < 1.0 else -lateral_distance*2
            reward -= angular_diff * 0.05 + wrong_direction

            #recompensa por acercarnos al objetivo
            reward += -3 if self.better_distance[agent_id] <= dist_to_goal else 8       
            self.better_distance[agent_id] = min(self.better_distance[agent_id], dist_to_goal)
            
            #RECOMPENSAS Y PENALIZACIONES EXTRA
            #muy cerca de la meta
            if dist_to_goal < 5.0:
                reward += 200
                done = True
            #colision
            if self.CARLA.collision_occurs[agent]:
                reward -= 50
                done = True
            #muy alejados del carril
            if lateral_distance > 4:
                reward -= 20
                done = True

            #evitar que pise freno y acelerador a la vez
            # if brake > 0.1 and throttle > 0.1:
            #     reward -=10
                
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

        dead_npcs = [v for v in self.CARLA.vehicles_npcs_list if not v.is_alive]
        for npc in dead_npcs:
            try:
                npc.destroy()
            except:
                pass
        
        self.CARLA.vehicles_npcs_list = [v for v in self.CARLA.vehicles_npcs_list if v.is_alive]
        if len(self.CARLA.vehicles_npcs_list) < 5:
            self.CARLA.spawn_vehicle(True)
        
        return self.__get_obs()
    
    def close(self):
        self.CARLA.destroy_actors()


# SÍ, indirectamente. Compara el ángulo del vehículo con el ángulo del waypoint del carril:

# Ya tienes angular_diff (diferencia de orientación con el carril)
# Si abs(angular_diff) > 90° → vas en sentido contrario
# 2. Semáforos
# SÍ, directamente. CARLA detecta infracciones:

# vehicle.is_at_traffic_light() - detecta si hay semáforo cerca
# vehicle.get_traffic_light_state() - devuelve el estado (Red, Yellow, Green)
# O usa el Traffic Manager que registra infracciones automáticamente
# 3. Exceso de velocidad
# SÍ, con waypoints. Los waypoints tienen límite de velocidad:

# waypoint.lane_id te da el carril
# Los carriles tienen límites de velocidad configurados en el mapa
# Comparas velocidad actual vs límite del waypoint
# 4. Distancia por waypoints
# Distancia por carretera (no línea recta).

# En vez de distancia euclidiana (línea recta A→B que atraviesa edificios), calculas:

# Waypoint actual del vehículo
# Waypoint del objetivo
# Generas una ruta por la carretera usando waypoint.next(distance) iterativamente hasta el objetivo
# Sumas las distancias entre waypoints consecutivos
# CARLA tiene GlobalRoutePlanner que hace esto automáticamente - te da la ruta siguiendo las calles y su distancia total.