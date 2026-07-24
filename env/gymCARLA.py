import sys
import os
import carla
import gym
from gym import spaces
import time
import numpy as np
from env.carlaControler import CarlaControler
#from sensors.camera import CameraProcessor


class envCARLA(gym.Env):
    """Class to create a gym env, where implement the steps, rewards and so on"""
    def __init__(self):
        self.action_space =  [
            spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32),
            spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32) 
        ]
        self.low_v  = np.array([0.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0,
                                 0.0, -1.0, 0.0, -1.0], dtype=np.float32)
        self.high_v = np.array([1.0,  1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0,
                                 1.0,  1.0, 1.0,  1.0], dtype=np.float32)

            
        vehicle_obs_space = spaces.Box(
            low=self.low_v, high=self.high_v,
            dtype=np.float32  
        )

        camera_obs = spaces.Box(
            low = 0,
            high = 22, 
            shape = (128, 128), 
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
                # "lidar": lidar_obs
            }),
            spaces.Dict({
                "vehicle_state": vehicle_obs_space,
                # "lidar": lidar_obs
            }),
        ]

        self.current_step = 0
        self.max_steps = 2050
        self.velocity_target = 10
        self.max_speed = 15
        self.distance = {}
        self.velocity = {}
        self.throttle = {}
        self.steer = {}
        self.brake = {}
        self.dist_to_goal = {}
        self.closest_waypoint_idx = {}
        self.last_waypoint_idx = {}
        self.__agent=[]
        self.agent_id=[]

        self.CARLA = CarlaControler()
        self.last_valid_cam = {}
        self.position_change = {}
        self.goal_positions = {}
        self.planner = {}
        self.lane_width = 3
        self.lateral_distance = {}
        self.angular_diff_rad ={}
        self.better_distance = {}
        self.initial_dist = {}

        self.cam_features = {}                 
        self.proximity_coef_vehicle = 3.0      
        self.proximity_coef_pedestrian = 5.0   
        self.proximity_dist_threshold = 0.6    
        self.proximity_bearing_ahead = 0.4     

        for i, vehicle in enumerate(self.CARLA.vehicles_marl_list):
            self.__agent.append(vehicle)
            agent_id = f"agent_{i}"
            self.agent_id.append(agent_id)
            self.planner[agent_id] = None
            self.better_distance[agent_id] = np.inf

            self.last_waypoint_idx[agent_id] = 0



##steer, throttle por agente
    def step(self, action): #vamos a bajar el espacio de acciones a 2 ya que freno y acelerador juntos resulta confuso para la red   
        for i, vehicle in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            throttle_ = float(action[i][0])
            steer_i = float(action[i][1])
            if throttle_ > 0.0:
                throttle_i = throttle_
                brake_i = 0.0
            elif throttle_ < -0.1:
                brake_i = 0.5*(-throttle_)
                throttle_i = 0.0
            else:
                throttle_i = 0.0
                brake_i = 0.0

            self.throttle[agent_id] = throttle_i
            self.steer[agent_id] = steer_i
            self.brake[agent_id] = brake_i

            if self.current_step % 100 == 0:
                print(f"agente {i} --> thorttle: {throttle_i}, steer: {steer_i}, brake: {brake_i}")      

            move = carla.VehicleControl(throttle_i, steer_i, brake_i)
            vehicle.apply_control(move)
        
        self.CARLA.tick()
        
        observations = self.__get_obs()
        self._last_obs = observations  # Guardar para logging en próximo step
        rewards, dones = self.__calculate_rewards()
        self.current_step += 1

        return observations, rewards, dones, {}




    def __get_obs(self):
        observation = {}

        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            if i == 0:
                other_agent = self.__agent[1]
            else:
                other_agent = self.__agent[0]

            velocity_ = agent.get_velocity()
            # other_agent_vel = other_agent.get_velocity()
            

            # vel_x_rel = velocity_.x - other_agent_vel.x
            # vel_y_rel = velocity_.y - other_agent_vel.y



            self.velocity[agent_id] = np.linalg.norm(np.array([velocity_.x, velocity_.y, velocity_.z]))
            norm_velocity = np.clip(self.velocity[agent_id]/self.max_speed, 0, 1) #vamos a fijar una velocidad maxima de momento de 50km/h(unos 14m/s)
            #acceleration = agent.get_acceleration()

            transform = agent.get_transform()
            vehicle_location = transform.location
            vehicle_angle = transform.rotation.yaw

            other_agent_location = other_agent.get_transform().location
            pos_x_rel = vehicle_location.x - other_agent_location.x
            pos_y_rel = vehicle_location.y - other_agent_location.y
            self.distance[agent_id] = np.sqrt(pos_x_rel**2 + pos_y_rel**2)
            distance_norm = np.clip(1 - (self.distance[agent_id] / 5), 0, 1) #coordinacion entre agentes
            # vel_x_norm = np.clip(vel_x_rel / self.max_speed, -1, 1)*relevance #esto nos sirve para coordinarnos entre agentes
            # vel_y_norm = np.clip(vel_y_rel / self.max_speed, -1, 1)*relevance
            pos_x_norm = np.clip(pos_x_rel / 5, -1, 1) #esto nos sirve para coordinarnos entre agentes
            pos_y_norm = np.clip(pos_y_rel / 5, -1, 1)


            bearing_forward = 0
            bearing_right = 0
            #curvature = 0
            dist_to_next = 0
            e2_raw = 0.0

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
                self.closest_waypoint_idx[agent_id] = closest_idx
                next_idx = min(closest_idx+1, len(route)-1)
                dist_to_next = vehicle_location.distance(route[next_idx][0].transform.location)


                if closest_idx < len(route) - 3:
                    next_wp = route[closest_idx+1][0]
                    dist_to_next = vehicle_location.distance(next_wp.transform.location)
                    delta = next_wp.transform.location - vehicle_location
                    vector_to_next = np.array([delta.x, delta.y], dtype=np.float64)
                    direction_to_next = vector_to_next / (np.linalg.norm(vector_to_next) + 1e-6)

                    yaw_rad = np.radians(vehicle_angle)
                    vehicle_forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
                    vehicle_right = np.array([-np.sin(yaw_rad), np.cos(yaw_rad)])

                    bearing_forward = float(np.clip(np.dot(direction_to_next, vehicle_forward), -1.0, 1.0))
                    bearing_right = float(np.clip(np.dot(direction_to_next, vehicle_right), -1.0, 1.0))
                    #curvature = (next_wp.transform.rotation.yaw - closest_wp.transform.rotation.yaw + 180) % 360 - 180.0

                e2_errors = []
                for k in range(1, 4):  #próximos 3 waypoints
                    idx_k = min(closest_idx + k, len(route) - 1)  #para no salirse si estamos cerca de los wp
                    wp_k = route[idx_k][0]
                    yaw_k = wp_k.transform.rotation.yaw
                    err_k = (vehicle_angle - yaw_k + 180) % 360 - 180
                    e2_errors.append(np.deg2rad(err_k))

                e2_raw = np.mean(e2_errors) if e2_errors else 0.0
                
                if next_idx < len(route) - 1:
                    self.dist_to_goal[agent_id] = dist_to_next + sum(
                        route[j][0].transform.location.distance(route[j+1][0].transform.location)
                        for j in range(next_idx, len(route)-1)
                    )
                    #self.dist_to_goal[agent_id] += min_dist
                else:
                    self.dist_to_goal[agent_id] = dist_to_next
            else:
                self.dist_to_goal[agent_id] = 0
                self.closest_waypoint_idx[agent_id] = 0
                waypoint = self.CARLA.get_map().get_waypoint(vehicle_location)
            
            lane_center = waypoint.transform.location
            angle_center = waypoint.transform.rotation.yaw
            distance_x = vehicle_location.x - lane_center.x
            distance_y = vehicle_location.y - lane_center.y
            lane_direction = np.array([np.cos(np.radians(angle_center)), np.sin(np.radians(angle_center))])
            self.lateral_distance[agent_id] = - distance_x * lane_direction[1] + distance_y * lane_direction[0]

            lat_norm = np.clip(self.lateral_distance[agent_id] / self.lane_width, 0, 1)
            angular_diff = (vehicle_angle - angle_center + 180) % 360 - 180
            self.angular_diff_rad[agent_id] = np.deg2rad(angular_diff)


            e1_norm = np.clip(self.angular_diff_rad[agent_id] / np.pi, -1.0, 1.0)
            e2_norm = np.clip(e2_raw / np.pi, -1.0, 1.0)

            vehicle_state = np.array([self.throttle.get(agent_id, 0.0),
                                    self.steer.get(agent_id, 0.0),
                                    norm_velocity, #Velocidad
                                    #acceleration.x/2, acceleration.y/2, acceleration.z/2, #Aceleración
                                    #transform.rotation.yaw/180, #Orientacion
                                    lat_norm, #desviacion lateral
                                    e1_norm, #desviacion angular
                                    e2_norm,
                                    bearing_forward,
                                    bearing_right,
                                    pos_y_norm,
                                    pos_x_norm,
                                    distance_norm],
                                    # vel_x_norm,
                                    # vel_y_norm],
                                    #np.log(self.dist_to_goal + 1) / 6],
                                    dtype=np.float32)
            
            vehicle_state = np.nan_to_num(vehicle_state, nan=0.0, posinf=0.0, neginf=0.0)

            sensor_obs = self.CARLA.get_sensor_data(agent)
            camera_obs = sensor_obs['camera_data']
            if camera_obs is None:
                camera_obs = self.last_valid_cam.get(agent_id, np.zeros((128,128), dtype=np.uint8))
            else:
                self.last_valid_cam[agent_id] = camera_obs

            cam_features = self._estimate_distances_from_camera(camera_obs)
            self.cam_features[agent_id] = cam_features  
            vehicle_state = np.concatenate([vehicle_state, cam_features])
            vehicle_state = np.clip(vehicle_state, self.low_v, self.high_v)

            observation[agent_id] = {"vehicle_state": vehicle_state}



        return observation

    
        
    def __calculate_rewards(self):
        rewards = {}
        dones = {}
        factor = 2
        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            done = False
            reward = 0
            #reward -= 0.25 #cada paso resta 0.25 en la recompensa

            #vehicle_state = observations[agent_id]['vehicle_state']
            # self.dist_to_goal = np.exp(vehicle_state[9] * 6) -1
            # angular_diff = abs(vehicle_state[4]*180)
            # wrong_direction = -20 if angular_diff > 90 else 0

            angular_error = -2*(abs(self.angular_diff_rad[agent_id])/(np.pi/3))
            reward += angular_error

            #vairaciones bruscas del volante, buscando evitar que haga S
            current_steer = self.steer.get(agent_id, 0.0)
            last_steer = getattr(self, 'last_steer', {}).get(agent_id, 0.0)
            delta_steer = current_steer - last_steer
            reward -= (delta_steer ** 2) * 0.5

            if not hasattr(self, 'last_steer'): self.last_steer = {}
            self.last_steer[agent_id] = current_steer

            #recompensa por velociad objetivo
            speed_error = 1 -  min(1, (abs(self.velocity[agent_id] - self.velocity_target)/self.velocity_target))

            reward += speed_error*factor
          
            #penalizacion extra si velocidad es 0 (no se mueve) 
            # if self.velocity < 0.1:
            #     reward -= 5.0
 

            #penalizacion por alejarse del centro, recompensa por ir centrado
            #tambien tenemos en cuenta el angulo
            # distance_error = -(abs(self.lateral_distance[agent_id])/self.lane_width)
            # reward += 2.0 if abs(distance_error) < 0.1 else distance_error*2
            #reward += 5 - angular_diff if angular_diff < 2 else -angular_diff * 1.5 + wrong_direction 
            lat_err = abs(self.lateral_distance[agent_id]) / self.lane_width
            centering_reward = 2.0 - 4.0 * lat_err  # +2.0 centrado, 0.0 a 0.5 anchos de carril, -2.0 en el borde
            reward += centering_reward


            if self.dist_to_goal[agent_id] < self.better_distance[agent_id]:
                reward += 2
            self.better_distance[agent_id] = min(self.better_distance[agent_id], self.dist_to_goal[agent_id])

            #bonus por completar waypoint
            if self.closest_waypoint_idx.get(agent_id, 0) > self.last_waypoint_idx.get(agent_id, 0):
                reward += 2
                self.last_waypoint_idx[agent_id] = self.closest_waypoint_idx[agent_id]

            #Log componentes de recompensa cada 100 steps
            if self.current_step % 100 == 0 and agent_id == "agent_0":
                print(f"[REWARD] vel:{self.velocity[agent_id]:.1f} lat:{self.lateral_distance[agent_id]:.2f} ang:{self.angular_diff_rad[agent_id]:.1f} dist:{self.dist_to_goal[agent_id]:.1f} total:{reward:.1f}")
            
            
            #recompensas de coordinacion entre agentes
            cam = self.cam_features.get(agent_id)
            if cam is not None:
                for dist_n, bearing, coef in (
                    (cam[0], cam[1], self.proximity_coef_vehicle),    
                    (cam[2], cam[3], self.proximity_coef_pedestrian),  
                ):
                    if dist_n > self.proximity_dist_threshold and abs(bearing) < self.proximity_bearing_ahead:
                        closeness = (dist_n - self.proximity_dist_threshold) / (1.0 - self.proximity_dist_threshold)
                        ahead = 1.0 - abs(bearing) / self.proximity_bearing_ahead
                        reward -= coef * closeness * ahead


            #RECOMPENSAS Y PENALIZACIONES EXTRA
            #muy cerca de la meta
            if self.dist_to_goal[agent_id] < 2.0:
                reward += 20
                done = True
                print("hemos llegado a la meta")
            #colision
            if self.CARLA.collision_occurs[agent.id]:
                reward -= 20
                if agent_id == "agent_0":
                    print(f" COLISION better_distance: {self.better_distance[agent_id]}, lateral_distance: {self.lateral_distance}, angular_distance: {self.angular_diff_rad[agent_id]}")
                done = True
            #muy alejados del carril
            if abs(self.lateral_distance[agent_id]) > 4:
                reward -= 10

            #evitar que pise freno y acelerador a la vez
            # if brake > 0.1 and throttle > 0.1:
            #     reward -=10
                
            if self.current_step >= self.max_steps: #no estamos usando el numero maximo de pasos de momento
                reward -= 10
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
            if agent.id in self.CARLA.sensors_data:
                self.CARLA.sensors_data[agent.id] = {'camera_data': None, 'lidar_data': None}

            self.closest_waypoint_idx[agent_id] = 0
            self.last_waypoint_idx[agent_id] = 0
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
                    self.initial_dist[agent_id] = sum(
                        self.planner[agent_id][j][0].transform.location.distance(self.planner[agent_id][j+1][0].transform.location)
                        for j in range(0, len(self.planner[agent_id])-1)
                    )
                    
            else:
                agent.set_transform(self.position_change[agent_idx])
            
            time.sleep(2)
            agent.set_target_velocity(carla.Vector3D(0, 0, 0))

        dead_npcs = [v for v in self.CARLA.vehicles_npcs_list if not v.is_alive]
        for npc in dead_npcs:
            try:
                npc.destroy()
            except:
                pass
        
        self.CARLA.vehicles_npcs_list = [v for v in self.CARLA.vehicles_npcs_list if v.is_alive]
        if len(self.CARLA.vehicles_npcs_list) < 20:
            self.CARLA.spawn_vehicle(True)
        
        return self.__get_obs()
    
    def _estimate_distances_from_camera(self, camera_obs):
        """
        Estimate distance and lateral bearing to the nearest vehicle and pedestrian
        using the semantic-segmentation
        """
        FOCAL_PX   = 64.0   
        IMG_CX     = 64.0   
        MAX_DIST   = 30.0   
        MIN_PIXELS = 3      
        CLASSES = [
            (14, 2.0),   
            (12, 0.5),   
        ]
        clean_cam = camera_obs.copy()
        clean_cam[90:, :] = 0

        features: list = []
        for cls_id, real_width in CLASSES:
            mask = (clean_cam == cls_id)
            if mask.sum() < MIN_PIXELS:
                features.extend([0.0, 0.0])  
                continue

            
            row_counts   = mask.sum(axis=1)
            densest_row  = int(np.argmax(row_counts))
            col_indices  = np.where(mask[densest_row])[0]

            apparent_w_px = float(col_indices[-1] - col_indices[0] + 1)
            if apparent_w_px < 1.0:
                features.extend([0.0, 0.0])
                continue

            dist_m    = (real_width * FOCAL_PX) / apparent_w_px
            dist_norm = float(np.clip(1.0 - dist_m / MAX_DIST, 0.0, 1.0))  

            cx      = float(col_indices[0] + col_indices[-1]) / 2.0
            bearing = float(np.clip((cx - IMG_CX) / IMG_CX, -1.0, 1.0))

            features.extend([dist_norm, bearing])

        return np.array(features, dtype=np.float32)

    def close(self):
        self.CARLA.destroy_actors()
