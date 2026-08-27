import sys
import os
import carla
import gym
from gym import spaces
import time
import cv2
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
        #vehicle_state: throttle, steer, brake, velocity, lat, e1, e2, bearing_fwd, bearing_right, pos_y, pos_x, dist_other
        self.low_v  = np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        self.high_v = np.array([1.0,  1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0], dtype=np.float32)

            
        vehicle_obs_space = spaces.Box(
            low=self.low_v, high=self.high_v,
            dtype=np.float32  
        )

        # Radar features: [closest_proximity, closest_bearing, closest_closing_rate,
        #                   second_proximity, second_bearing, second_closing_rate]
        cam_features = spaces.Box(
            low = np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0]),
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
            dtype=np.float32)


        self.observation_space = [
            spaces.Dict({
                "vehicle_state": vehicle_obs_space,
                "cam_features": cam_features,
            }),
            spaces.Dict({
                "vehicle_state": vehicle_obs_space,
                "cam_features": cam_features,
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
        self.smoothed_radar = {}             
        self.proximity_coef_vehicle = 6.0      
        self.proximity_dist_threshold = 0.3    
        self.proximity_bearing_ahead = 0.5
        self.closing_rate_weight = 3.0
        self.radar_range = 50.0  
        #vamos a forzar la colision ya que no aprenden a evitarla
        self.curriculum_prob = 0.3
        self.curriculum_npc = {}

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
            else:
                throttle_i = 0.0
                brake_i = abs(throttle_)

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
                                    self.brake.get(agent_id, 0.0),
                                    norm_velocity, #Velocidad
                                    lat_norm, #desviacion lateral
                                    e1_norm, #desviacion angular
                                    e2_norm,
                                    bearing_forward,
                                    bearing_right,
                                    pos_y_norm,
                                    pos_x_norm,
                                    distance_norm],
                                    dtype=np.float32)
            
            vehicle_state = np.nan_to_num(vehicle_state, nan=0.0, posinf=0.0, neginf=0.0)

            # Radar-based obstacle features
            sensor_obs = self.CARLA.get_sensor_data(agent)
            radar_features = self._get_radar_features(sensor_obs.get('radar_data'), agent_id)
            self.cam_features[agent_id] = radar_features

            vehicle_state = np.clip(vehicle_state, self.low_v, self.high_v)

            observation[agent_id] = {"vehicle_state": vehicle_state,
                                     "cam_features": radar_features}



        return observation

    
        
    def __calculate_rewards(self):
        rewards = {}
        dones = {}
        factor = 2
        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            done = False
            reward = 0.0

            angular_error = -2*(abs(self.angular_diff_rad[agent_id])/(np.pi/3))
            reward += angular_error

            #vairaciones bruscas del volante, buscando evitar que haga S
            current_steer = self.steer.get(agent_id, 0.0)
            last_steer = getattr(self, 'last_steer', {}).get(agent_id, 0.0)
            delta_steer = current_steer - last_steer
            reward -= (delta_steer ** 2) * 4.0

            if not hasattr(self, 'last_steer'): self.last_steer = {}
            self.last_steer[agent_id] = current_steer

            lat_err = abs(self.lateral_distance[agent_id]) / self.lane_width
            centering_reward = 2.0 - 4.0 * lat_err  #+2.0 centrado, 0.0 a 0.5 anchos de carril, -2.0 en el borde
            reward += centering_reward

            prev_dist = getattr(self, 'prev_dist_to_goal', {}).get(agent_id, self.dist_to_goal[agent_id])
            progress_this_step = prev_dist - self.dist_to_goal[agent_id]
            progress_this_step = max(-1.0, min(1.0, progress_this_step))
            reward += progress_this_step

            if not hasattr(self, 'prev_dist_to_goal'):
                self.prev_dist_to_goal = {}
            self.prev_dist_to_goal[agent_id] = self.dist_to_goal[agent_id]

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
            
            
            #recompensas basadas en radar
            radar = self.cam_features.get(agent_id)
            danger_severity = 0.0
            if radar is not None:
                closest_proximity, closest_bearing, closest_closing = radar[0:3]
                second_proximity, second_bearing, second_closing = radar[3:6]

                for proximity, bearing, closing_rate, coef in ((closest_proximity,
                                                                closest_bearing,
                                                                closest_closing,
                                                                self.proximity_coef_vehicle),

                                                               (second_proximity,
                                                                second_bearing,
                                                                second_closing,
                                                                self.proximity_coef_vehicle),):

                    if proximity < 0.01:
                        continue

                    closeness = np.clip((proximity - self.proximity_dist_threshold) / (1.0 - self.proximity_dist_threshold), 0.0, 1.0)
                    ahead = np.clip(1.0 - abs(bearing) / self.proximity_bearing_ahead, 0.0, 1.0)
                    closing = max(0.0, closing_rate)
                    danger = closeness * ahead * (1.0 + self.closing_rate_weight * closing)

                    braking_effort = max(0.0, self.brake.get(agent_id, 0.0))
                    mitigation_factor = 1.0 - (braking_effort * 0.8) if (closing_rate > 0.1 or proximity > 0.6) else 1.0
                    reward -= coef * danger * mitigation_factor

                    danger_severity = max(danger_severity, closeness * ahead)
            
            speed_gate = 1.0 - danger_severity
            reward += progress_this_step * speed_gate

            speed_error = 1 - min(1, (abs(self.velocity[agent_id] - self.velocity_target)/max(0.01, self.velocity_target)))
            reward += speed_error * factor * speed_gate

            if danger_severity < 0.05 and self.velocity[agent_id] < 1.0:
                reward -= 2.0 * (1.0 - self.velocity[agent_id])

            #freno en situacion de peligro
            if danger_severity > 0.2:
                braking_now = max(0.0, self.brake.get(agent_id, 0.0))
                if braking_now > 0.2:
                    reward += 3.0 * braking_now * danger_severity
                elif self.velocity[agent_id] > 2.0:
                    reward -= 3.0 * danger_severity * (self.velocity[agent_id] / self.max_speed)

                if self.current_step % 50 == 0:
                    print(f"[DANGER] {agent_id} step={self.current_step},"
                    f"prox={closest_proximity:.2f} closing={closest_closing:.2f},"
                    f"brake={self.brake.get(agent_id, 0.0):.2f} vel={self.velocity[agent_id]:.1f}")

            #RECOMPENSAS Y PENALIZACIONES EXTRA
            #muy cerca de la meta
            if self.dist_to_goal[agent_id] < 2.0:
                reward += 20
                done = True
                print("hemos llegado a la meta")
            #colision
            if self.CARLA.collision_occurs[agent.id]:
                reward -= 200
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
        
        spawn_points = self.CARLA.world.get_map().get_spawn_points()
        
        for agent_id in agent_ids:
            agent_idx = self.agent_id.index(agent_id)
            agent = self.__agent[agent_idx]
            
            if agent.id in self.CARLA.sensors_data:
                self.CARLA.sensors_data[agent.id] = {'camera_data': None, 'lidar_data': None, 'radar_data': None}
                self.last_valid_cam.pop(agent_id, None)
                self.cam_features.pop(agent_id, None)
                self.smoothed_radar.pop(agent_id, None)

            self.closest_waypoint_idx[agent_id] = 0
            self.last_waypoint_idx[agent_id] = 0
            #spawn point aleatorio en diferentes episodios, el mismo en cada paso
            if not same_position:
                if spawn_points:
                    vehicles = self.CARLA.world.get_actors().filter('vehicle.*')
                    max_attempts = 15
                    safe_distance = 10.0

                    chosen_spawn = None
                    for attempt in range(max_attempts):
                        candidate_spawn = np.random.choice(spawn_points)
                        is_occupied = False

                        for vehicle in vehicles:
                            if vehicle.id != agent.id:
                                dist = vehicle.get_location().distance(candidate_spawn.location)
                                if dist < safe_distance:
                                    is_occupied = True
                                    break
                        if not is_occupied:
                            chosen_spawn = candidate_spawn
                            break
                    if chosen_spawn is None:
                        chosen_spawn = candidate_spawn    

                    self.position_change[agent_idx] = chosen_spawn

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
                    self._apply_collision_curriculum(agent_id)
                    
            else:
                agent.set_transform(self.position_change[agent_idx])
            
            agent.set_target_velocity(carla.Vector3D(0, 0, 0))
            self.CARLA.tick()

        dead_npcs = [v for v in self.CARLA.vehicles_npcs_list if not v.is_alive]
        for npc in dead_npcs:
            try:
                npc.destroy()
            except:
                pass
        self.prev_dist_to_goal = {}
        self.last_steer = {}
        
        self.CARLA.vehicles_npcs_list = [v for v in self.CARLA.vehicles_npcs_list if v.is_alive]
        if len(self.CARLA.vehicles_npcs_list) < 20:
            self.CARLA.spawn_vehicle(True)

        self.CARLA.tick()

        for agent in self.__agent:
            self.CARLA.reset_collision(agent)
        
        return self.__get_obs()
    
    def _get_radar_features(self, radar_data, agent_id):
        """Extract the 2 closest obstacle features from radar detections.
        Returns 6 features: [closest_proximity, closest_bearing, closest_closing_rate,
                             second_proximity, second_bearing, second_closing_rate]
        
        Radar gives real distance (m), azimuth (rad), velocity (m/s, negative=approaching).
        No object classification needed — the agent just needs to know what's ahead.
        """
        RADAR_RANGE = self.radar_range
        features = np.zeros(6, dtype=np.float32)
        
        if radar_data is None or len(radar_data) == 0:
            return self._smooth_radar(agent_id, features)
        
        scored_detections = []
        
        for det in radar_data:
            depth = det['depth']
            azimuth = det['azimuth']  
            altitude = det['altitude']  
            velocity = det['velocity']  
            
            if abs(altitude) > 0.3:
                continue
            if depth < 1.0:
                continue
                
            proximity = np.clip(1.0 - depth / RADAR_RANGE, 0.0, 1.0)
            bearing = np.clip(azimuth / (np.pi / 4.0), -1.0, 1.0)  
            closing_rate = np.clip(-velocity / self.max_speed, -1.0, 1.0)  
            
            score = proximity * (1.0 - 0.5 * abs(bearing))
            scored_detections.append((score, proximity, bearing, closing_rate))
        
        scored_detections.sort(key=lambda x: x[0], reverse=True)
        
        if len(scored_detections) >= 1:
            _, features[0], features[1], features[2] = scored_detections[0]
        if len(scored_detections) >= 2:
            _, features[3], features[4], features[5] = scored_detections[1]
        
        return self._smooth_radar(agent_id, features)
    
    def _smooth_radar(self, agent_id, features, alpha=0.5):
        """Smooth radar features to reduce jitter between ticks."""
        prev = self.smoothed_radar.get(agent_id)
        if prev is None:
            smoothed = features.copy()
        else:
            smoothed = alpha * features + (1.0 - alpha) * prev
        self.smoothed_radar[agent_id] = smoothed
        return smoothed

    def _apply_collision_curriculum(self, agent_id):
        route = self.planner[agent_id]

        if not route or len(route) < 2 or np.random.rand() >= self.curriculum_prob:
            self._retire_curriculum_npc(agent_id)
            return

        target_dist = np.random.uniform(20.0, 35.0) 
        cum_dist = 0.0
        chosen_wp = route[-1][0]
        chosen_idx = len(route) - 1

        for j in range(len(route) - 1):
            cum_dist += route[j][0].transform.location.distance(route[j + 1][0].transform.location)

            if cum_dist >= target_dist:
                chosen_wp = route[j + 1][0]
                chosen_idx = j + 1
                break

        npc = self.curriculum_npc.get(agent_id)

        if npc is None or not npc.is_alive:
            bp_list = [bp for bp in self.CARLA.world.get_blueprint_library().filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) >= 4]
            bp = bp_list[np.random.randint(len(bp_list))]

            
            spawned = None
            for offset in [0, 3, -3, 6, -6, 10]:
                try_idx = min(max(1, chosen_idx + offset), len(route) - 1)
                loc = route[try_idx][0].transform.location
                rot = route[try_idx][0].transform.rotation
                spawn_t = carla.Transform(
                    carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.3), rot
                )
                spawned = self.CARLA.world.try_spawn_actor(bp, spawn_t)
                if spawned is not None:
                    break

            if spawned is None:
                print(f"[curriculum] couldn't spawn stopped NPC for {agent_id}, skipping this episode")
                return

            npc = spawned
            self.curriculum_npc[agent_id] = npc

        else:
            loc = chosen_wp.transform.location
            rot = chosen_wp.transform.rotation
            npc.set_transform(carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.3), rot
            ))

        npc.set_autopilot(False)
        npc.set_target_velocity(carla.Vector3D(0, 0, 0))
        npc.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            
    def _retire_curriculum_npc(self, agent_id):
        npc = self.curriculum_npc.get(agent_id)
        if npc is not None and npc.is_alive:
            npc.set_target_velocity(carla.Vector3D(0, 0, 0))
            npc.set_transform(carla.Transform(carla.Location(x=0.0, y=0.0, z=-50.0))) 



    def close(self):
        for npc in self.curriculum_npc.values():
            try:
                if npc.is_alive:
                    npc.destroy()
            except Exception:
                pass
        self.CARLA.destroy_actors()
