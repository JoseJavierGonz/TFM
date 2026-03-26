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
        self.max_steps = 1050 
        self.action_space =  [
            spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32),
            spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32) 
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
        self.planner = {}
        self.lane_width = 1
        self.better_distance = {}

        for i, vehicle in enumerate(self.CARLA.vehicles_marl_list):
            self.__agent.append(vehicle)
            agent_id = f"agent_{i}"
            self.agent_id.append(agent_id)
            self.planner[agent_id] = None
            self.better_distance[agent_id] = np.inf




    def step(self, action): #vamos a bajar el espacio de acciones a 2 ya que freno y acelerador juntos resulta confuso para la red   
        for i, vehicle in enumerate(self.__agent):
            throttle = float(action[i][0])
            steer = float(action[i][1])

            if throttle > 0:
                brake = 0.0
            else:
                brake = abs(throttle)
                throttle = 0.0 

            if self.current_step % 100 == 0:
                print(f"agente {i} --> thorttle: {throttle}, steer: {steer}, brake: {brake}")      

            move = carla.VehicleControl(throttle, steer, brake)
            vehicle.apply_control(move)
        
        self.CARLA.tick()
        
        observations = self.__get_obs()
        self._last_obs = observations  # Guardar para logging en próximo step
        rewards, dones = self.__calculate_rewards(observations)
        self.current_step += 1

        return observations, rewards, dones, {}




    def __get_obs(self):
        observation = {}
        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            velocity = agent.get_velocity()
            #acceleration = agent.get_acceleration()

            transform = agent.get_transform()
            vehicle_location = transform.location
            vehicle_angle = transform.rotation.yaw


            bearing_forward = 0
            bearing_right = 0
            curvature = 0
            dist_to_next = 0

            if self.planner[agent_id] and len(self.planner[agent_id]) > 0:
                route = self.planner[agent_id]
                
                min_dist = float('inf')
                closest_idx = 0
                closest_wp = route[0][0]

                for idx, (wp, _) in enumerate(route):
                    dist = vehicle_location.distance(wp.transform.location)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = idx
                        closest_wp = wp
                
                waypoint = closest_wp

                if closest_idx < len(route) - 5:
                    next_wp = route[closest_idx + 1][0]  
                    dist_to_next = vehicle_location.distance(next_wp.transform.location)
                    vector_to_next = next_wp.transform.location - vehicle_location
                    vehicle_forward = np.array([np.cos(np.radians(vehicle_angle)), 
                                                np.sin(np.radians(vehicle_angle))])
                    vehicle_right = np.array([np.sin(np.radians(vehicle_angle)), 
                                            -np.cos(np.radians(vehicle_angle))])

                    bearing_forward = (vector_to_next.x * vehicle_forward[0] + 
                                    vector_to_next.y * vehicle_forward[1]) / (dist_to_next + 0.1)
                    bearing_right = (vector_to_next.x * vehicle_right[0] + 
                                    vector_to_next.y * vehicle_right[1]) / (dist_to_next + 0.1)
                    
                    curvature = (next_wp.transform.rotation.yaw - closest_wp.transform.rotation.yaw + 180) % 360 - 180.0

                if closest_idx < len(route) - 1:
                    dist_to_goal = sum(
                        route[j][0].transform.location.distance(route[j+1][0].transform.location)
                        for j in range(closest_idx, len(route)-1)
                    )
                    dist_to_goal += min_dist
                else:
                    dist_to_goal = vehicle_location.distance(route[-1][0].transform.location)
            else:
                dist_to_goal = 0
                waypoint = self.CARLA.get_map().get_waypoint(vehicle_location)
            
            lane_center = waypoint.transform.location
            angle_center = waypoint.transform.rotation.yaw
            self.lane_width = waypoint.lane_width
            distance_x = vehicle_location.x - lane_center.x
            distance_y = vehicle_location.y - lane_center.y
            lane_direction = np.array([np.cos(np.radians(angle_center)), np.sin(np.radians(angle_center))])
            lateral_distance = - distance_x * lane_direction[1] + distance_y * lane_direction[0]

            angular_diff = (vehicle_angle - angle_center + 180) % 360 - 180

                  

            vehicle_state = np.array([velocity.x/20, velocity.y/20, velocity.z/20, #Velocidad
                                    #acceleration.x/2, acceleration.y/2, acceleration.z/2, #Aceleración
                                    #transform.rotation.yaw/180, #Orientacion
                                    lateral_distance/self.lane_width, #desviacion lateral
                                    angular_diff/180, #desviacion angular
                                    bearing_forward,
                                    bearing_right,
                                    curvature/180,
                                    np.log(dist_to_next + 1) / 4, 
                                    np.log(dist_to_goal + 1) / 6],
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

    
        
    def __calculate_rewards(self, observations):
        rewards = {}
        dones = {}
        factor = 0.3
        max_speed = 10.0
        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            done = False
            reward = 0

            vehicle_state = observations[agent_id]['vehicle_state']
            dist_to_goal = np.exp(vehicle_state[9] * 6) -1
            velocity = np.linalg.norm(vehicle_state[:3])*20
            lateral_distance = abs(vehicle_state[3]*self.lane_width)
            angular_diff = abs(vehicle_state[4]*180)
            wrong_direction = -20 if angular_diff > 90 else 0

            #recompensa por velociad alta(mirar documentacion de si puedo obtener la velocidad maxima del carril para actualizar max_speed)
            speed_error = abs(velocity - max_speed)


            #EMPEZAMOS CON LAS RECOMPENSAS
            reward = -1 #cada paso resta 1 en la recompensa
            reward += (max_speed - speed_error) * factor
          
            #penalizacion extra si velocidad es 0 (no se mueve)
            if velocity < 0.5:
                reward -= 5.0/(velocity+0.1)
            elif 0.5 <= velocity <= 4:
                reward += velocity
 

            #penalizacion por alejarse del centro, recompensa por ir centrado
            #tambien tenemos en cuenta el angulo
            reward += 2.0/(lateral_distance+0.1) if lateral_distance < 0.8 else -lateral_distance*2
            reward += 5 - angular_diff if angular_diff < 2 else -angular_diff * 1.5 + wrong_direction 

            #recompensa por acercarnos al objetivo
            reward += -10 if self.better_distance[agent_id] <= dist_to_goal else 10       
            self.better_distance[agent_id] = min(self.better_distance[agent_id], dist_to_goal)
            
            # DEBUG: Log componentes de recompensa cada 100 steps
            if self.current_step % 100 == 0 and agent_id == "agent_0":
                print(f"[REWARD] vel:{velocity:.1f} lat:{lateral_distance:.2f} ang:{angular_diff:.1f} dist:{dist_to_goal:.1f} total:{reward:.1f}")
            
            #RECOMPENSAS Y PENALIZACIONES EXTRA
            #muy cerca de la meta
            if dist_to_goal < 5.0:
                reward += 200
                done = True
                print("hemos llegado a la meta")
            #colision
            if self.CARLA.collision_occurs[agent]:
                reward -= 50
                if agent_id == "agent_0":
                    print(f" COLISION better_distance: {self.better_distance[agent_id]}, lateral_distance: {lateral_distance}, angular_distance: {angular_diff}")
                done = True
            #muy alejados del carril
            if lateral_distance > 4:
                reward -= 20

            #evitar que pise freno y acelerador a la vez
            # if brake > 0.1 and throttle > 0.1:
            #     reward -=10
                
            if self.current_step >= self.max_steps:
                reward -= 50
                done = True
                print("no conseguimos llegar en 2500 steps")
            
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
                    available_goals = [sp for sp in spawn_points if sp != self.position_change[agent_idx]
                                                                and sp.location.distance(self.position_change[agent_idx].location) > 200]
                    if not available_goals:
                        available_goals = max(spawn_points, key=lambda sp: sp.location.distance(self.position_change[agent_idx].location))
                        self.goal_positions[agent_id] = available_goals
                    else:
                        self.goal_positions[agent_id] = np.random.choice(available_goals)

                    agent.set_transform(self.position_change[agent_idx])
                    self.better_distance[agent_id] = np.inf
                    self.planner[agent_id] = self.CARLA.route_planner.trace_route(
                                                            self.position_change[agent_idx].location,
                                                            self.goal_positions[agent_id].location)
                    
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