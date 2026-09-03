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
    """Clase donde crearemos en entorno de gym, implementaremos el step, las observaciones, rewards y reset"""
    def __init__(self, num_vehicles=20, num_walkers=20, curriculum_prob=0.7):
        self.action_space =  [
            spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32),
            spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32) 
        ]
        #vehicle_state: throttle, steer, brake, velocity, lat, e1, e2, bearing_fwd, bearing_right, pos_y, pos_x, dist_other
        self.low_v  = np.array([0.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
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

        self.CARLA = CarlaControler(num_vehicles=num_vehicles, num_walkers=num_walkers)


        self.current_step = 0
        self.max_steps = 2050
        self.velocity_target = 8
        self.max_speed = 12
        self.distance = {}
        self.velocity = {}
        self.throttle = {}
        self.steer = {}
        self.brake = {}
        self.dist_to_goal = {}
        self.prev_dist_to_goal = {}
        self.last_steer = {}
        self.low_speed_streak = {}
        self.offroad_streak = {}
        self.termination_cause = {} #causa de terminacion por agente, para las metricas del informe
        self.closest_waypoint_idx = {}
        self.last_waypoint_idx = {}
        self.__agent=[]
        self.agent_id=[]
        self.last_valid_cam = {}
        self.position_change = {}
        self.goal_positions = {}
        self.planner = {}
        self.route_xy = {}
        self.route_window = {}
        self.route_min_dist = {}
        self.lane_width = 3
        self.lateral_distance = {}
        self.angular_diff_rad ={}
        self.better_distance = {}
        self.initial_dist = {}
        self.cam_features = {}
        self.smoothed_radar = {}             
        self.proximity_coef_vehicle = 2.5
        self.proximity_dist_threshold = 0.3    
        self.proximity_bearing_ahead = 0.5
        self.closing_rate_weight = 3.0
        self.radar_range = 30.0  
        self.curriculum_prob = curriculum_prob #vamos a forzar la colision ya que no aprenden a evitarla
        self.curriculum_npc = {}
        self.safe_stop_counter = {}
        self.curriculum_cooldown = {}
        self.curriculum_rearm_ticks = 100  
        self.curriculum_hold_ticks = {}
        self.curriculum_released = {}
        self.curriculum_hold_min_s = 4.0
        self.curriculum_hold_max_s = 10.0
        self.curriculum_release_dist = 12.0
        self.curriculum_release_idx = {}

        for i, vehicle in enumerate(self.CARLA.vehicles_marl_list):
            self.__agent.append(vehicle)
            agent_id = f"agent_{i}"
            self.agent_id.append(agent_id)
            self.planner[agent_id] = None
            self.better_distance[agent_id] = np.inf

            self.last_waypoint_idx[agent_id] = 0


        if len(self.__agent) < 2:
            raise RuntimeError(f"Solo {len(self.__agent)} vehiculos MARL spawneados; "
                               "abortando antes de perder el entrenamiento")



    ##steer, throttle por agente
    def step(self, action): #vamos a bajar el espacio de acciones a 2 ya que freno y acelerador juntos resulta confuso para la red  
        """Acción que harán nuestros agentes. Las proporciona nuestra politica y se la pasamos a CARLA""" 
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

        #Para ver si podemos quitar el coches que hemos fijado como obstaculo porque nuestro agente lo ha hecho bien
        for i, agent in enumerate(self.__agent):
            self._update_curriculum_release(self.agent_id[i], agent)

        self._last_obs = observations  #Guardar para logging en próximo step
        rewards, dones = self.__calculate_rewards()
        self.current_step += 1

        return observations, rewards, dones, self._build_info()


    def _build_info(self):
        """Metricas por agente para el informe. Solo lectura de estado ya existente."""
        completion = {}
        for aid in self.agent_id:
            total = self.initial_dist.get(aid, 0.0)
            best = self.better_distance.get(aid, np.inf)
            if total and total > 0 and np.isfinite(best):
                completion[aid] = float(np.clip(1.0 - best / total, 0.0, 1.0))
            else:
                completion[aid] = 0.0

        return {
            "termination": dict(self.termination_cause),
            "velocity": {a: float(v) for a, v in self.velocity.items()},
            "dist_to_goal": {a: float(v) for a, v in self.dist_to_goal.items()},
            "lateral": {a: float(v) for a, v in self.lateral_distance.items()},
            "route_completion": completion,
            "initial_dist": {a: float(v) for a, v in self.initial_dist.items()},
        }




    def __get_obs(self):
        """Estado de nuestros agentes y el entorno en cada step"""
        observation = {}

        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            if i == 0:
                other_agent = self.__agent[1]
            else:
                other_agent = self.__agent[0]

            velocity_ = agent.get_velocity()



            self.velocity[agent_id] = np.linalg.norm(np.array([velocity_.x, velocity_.y, velocity_.z]))
            norm_velocity = np.clip(self.velocity[agent_id]/self.max_speed, 0, 1) 

            transform = agent.get_transform()
            vehicle_location = transform.location
            vehicle_angle = transform.rotation.yaw

            other_agent_location = other_agent.get_transform().location
            pos_x_rel = vehicle_location.x - other_agent_location.x
            pos_y_rel = vehicle_location.y - other_agent_location.y
            self.distance[agent_id] = np.sqrt(pos_x_rel**2 + pos_y_rel**2)
            distance_norm = np.clip(1 - (self.distance[agent_id] / 5), 0, 1) #coordinacion entre agentes
            pos_x_norm = np.clip(pos_x_rel / 5, -1, 1) #esto nos sirve para coordinarnos entre agentes
            pos_y_norm = np.clip(pos_y_rel / 5, -1, 1)


            bearing_forward = 0
            bearing_right = 0
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
                self.route_min_dist[agent_id] = float(min_dist)
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

                else:
                    self.dist_to_goal[agent_id] = dist_to_next

            else:
                self.dist_to_goal[agent_id] = 0
                self.closest_waypoint_idx[agent_id] = 0
                self.route_min_dist[agent_id] = 0.0
                waypoint = self.CARLA.get_map().get_waypoint(vehicle_location)
            
            lane_center = waypoint.transform.location
            angle_center = waypoint.transform.rotation.yaw
            distance_x = vehicle_location.x - lane_center.x
            distance_y = vehicle_location.y - lane_center.y
            lane_direction = np.array([np.cos(np.radians(angle_center)), np.sin(np.radians(angle_center))])
            self.lateral_distance[agent_id] = - distance_x * lane_direction[1] + distance_y * lane_direction[0]

            lat_norm = np.clip(self.lateral_distance[agent_id] / self.lane_width, -1, 1)
            angular_diff = (vehicle_angle - angle_center + 180) % 360 - 180
            self.angular_diff_rad[agent_id] = np.deg2rad(angular_diff)


            e1_norm = np.clip(self.angular_diff_rad[agent_id] / np.pi, -1.0, 1.0)
            e2_norm = np.clip(e2_raw / np.pi, -1.0, 1.0)

            vehicle_state = np.array([self.throttle.get(agent_id, 0.0),
                                    self.steer.get(agent_id, 0.0),
                                    self.brake.get(agent_id, 0.0),
                                    norm_velocity, 
                                    lat_norm, 
                                    e1_norm, 
                                    e2_norm,
                                    bearing_forward,
                                    bearing_right,
                                    pos_y_norm,
                                    pos_x_norm,
                                    distance_norm],
                                    dtype=np.float32)
            
            vehicle_state = np.nan_to_num(vehicle_state, nan=0.0, posinf=0.0, neginf=0.0)

            radar_features = self._get_vehicle_features(agent, agent_id)
            self.cam_features[agent_id] = radar_features

            vehicle_state = np.clip(vehicle_state, self.low_v, self.high_v)

            observation[agent_id] = {"vehicle_state": vehicle_state,
                                     "cam_features": radar_features}



        return observation

    
        
    def __calculate_rewards(self):
        """Cálculo de recompensas"""
        rewards = {}
        dones = {}
        factor = 2
        for i, agent in enumerate(self.__agent):
            agent_id = self.agent_id[i]
            done = False
            reward = 0.0
            cause = None #causa de terminacion para las metricas. Las 4 ramas terminales son

            angular_error = -2*(abs(self.angular_diff_rad[agent_id])/(np.pi/3))
            reward += angular_error

            #vairaciones bruscas del volante, buscando evitar que haga zigzag
            current_steer = self.steer.get(agent_id, 0.0)
            last_steer = getattr(self, 'last_steer', {}).get(agent_id, 0.0)
            delta_steer = current_steer - last_steer
            reward -= (delta_steer ** 2) * 4.0
            if not hasattr(self, 'last_steer'): self.last_steer = {}
            self.last_steer[agent_id] = current_steer

            #distancia lateral pero solo recompensamos si va centrado
            lat_err = abs(self.lateral_distance[agent_id]) / self.lane_width
            centering_reward = max(-4.0, 2.0 - 4.0 * lat_err)
            if centering_reward > 0.0:
                centering_reward *= min(1.0, self.velocity[agent_id] / 2.0)
            reward += centering_reward

            prev_dist = getattr(self, 'prev_dist_to_goal', {}).get(agent_id, self.dist_to_goal[agent_id])
            progress_this_step = prev_dist - self.dist_to_goal[agent_id]
            progress_this_step = max(-1.0, min(1.0, progress_this_step)) #reward añadido mas adelante teniendo en cuenta el peligro por obstaculos
            if not hasattr(self, 'prev_dist_to_goal'):
                self.prev_dist_to_goal = {}
            self.prev_dist_to_goal[agent_id] = self.dist_to_goal[agent_id]

            if self.dist_to_goal[agent_id] < self.better_distance[agent_id]:
                reward += 2
            self.better_distance[agent_id] = min(self.dist_to_goal[agent_id], self.better_distance[agent_id])

            #bonus por completar waypoint
            if self.closest_waypoint_idx.get(agent_id, 0) > self.last_waypoint_idx.get(agent_id, 0):
                reward += 2
                self.last_waypoint_idx[agent_id] = self.closest_waypoint_idx[agent_id]

            #Log componentes de recompensa cada 100 steps
            if self.current_step % 100 == 0 and agent_id == "agent_0":
                print(f"[REWARD] vel:{self.velocity[agent_id]:.1f} lat:{self.lateral_distance[agent_id]:.2f} ang:{self.angular_diff_rad[agent_id]:.1f} dist:{self.dist_to_goal[agent_id]:.1f} total:{reward:.1f}")           
            
            radar = self.cam_features.get(agent_id, np.zeros(6, dtype=np.float32))
            danger_severity = 0.0
            stop_excuse = 0.0
            closest_proximity, closest_bearing, closest_closing = radar[0:3]
            second_proximity, second_bearing, second_closing = radar[3:6]
            if np.any(radar > 0.01):
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
                    ahead = self._in_path_weight(abs(bearing) * self.lane_width) #cuanto nos importa el vehiculo obstaculo
                    safe_gap = max(8.0, 2.5 * self.velocity[agent_id]) #distancia de seguridad con los obstaculos
                    gap_prox = np.clip(1.0 - safe_gap / self.radar_range, 0.0, 0.95)
                    closeness = np.clip((proximity - gap_prox) / max(1e-3, 1.0 - gap_prox), 0.0, 1.0)
                    danger_severity = max(danger_severity, closeness * ahead) #danger_severity simpre se acumula, aunque el agente este parado bien
                    closing = max(0.0, closing_rate) #como de rapido nos acercamos o nos alejamos
                    danger = closeness * ahead * (1.0 + self.closing_rate_weight * closing)
                    braking_effort = max(0.0, self.brake.get(agent_id, 0.0)) # si lo tenemos muy cerca no penalizamos tanto por un frenazo
                    mitigation_factor = 1.0 - (braking_effort * 0.8) if (closing_rate > 0.1 or proximity > 0.6) else 1.0
                    reward -= coef * danger * mitigation_factor
            
            speed_gate = max(0.3, 1.0 - 0.7 * danger_severity)
            reward += progress_this_step * speed_gate

            speed_error = 1 - min(1, (abs(self.velocity[agent_id] - self.velocity_target)/max(0.01, self.velocity_target)))
            reward += speed_error * factor * speed_gate
            if not hasattr(self, 'low_speed_streak') :
                self.low_speed_streak = {}
            if self.velocity[agent_id] < 1.0:
                self.low_speed_streak[agent_id] = self.low_speed_streak.get(agent_id, 0) + 1
            else:
                self.low_speed_streak[agent_id] = 0  
            stop_excuse = min(1.0, danger_severity / 0.3)
            if self.low_speed_streak.get(agent_id, 0) > 15:
                reward -= 2.0 * max(0.0, 1.0 - self.velocity[agent_id]) * (1.0 - stop_excuse)

            if self.current_step % 100 == 0:
                print(f"[DANGER] {agent_id} step={self.current_step},"
                f"prox={closest_proximity:.2f} closing={closest_closing:.2f},"
                f"brake={self.brake.get(agent_id, 0.0):.2f} vel={self.velocity[agent_id]:.1f}")

            if self.dist_to_goal[agent_id] < 2.0: #muy cerca de la meta
                reward += 20
                done = True
                cause = "goal"
                print("hemos llegado a la meta")
            
            if self.CARLA.collision_occurs[agent.id]: #colision
                reward -= 100
                cause = "collision"
                print(f" COLISION better_distance: {self.better_distance[agent_id]}, lateral_distance: {self.lateral_distance}, angular_distance: {self.angular_diff_rad[agent_id]}")
                done = True
            
            if abs(self.lateral_distance[agent_id]) > 4: #muy alejados del carril
                reward -= 10

            if abs(self.lateral_distance[agent_id]) > 8.0: #si esta muy alejado del carril durante mucho tiempo, paramos el episodio
                self.offroad_streak[agent_id] = self.offroad_streak.get(agent_id, 0) + 1
            else:
                self.offroad_streak[agent_id] = 0
            if self.offroad_streak.get(agent_id, 0) > 30:
                reward -= 30
                done = True
                cause = "offroad"
                print(f"{agent_id} perdido fuera de ruta ({self.lateral_distance[agent_id]:.1f}m), reinicio")
                self.offroad_streak[agent_id] = 0

            if self.current_step >= self.max_steps: #no estamos usando el numero maximo de pasos de momento
                reward -= 10
                done = True
                cause = "timeout"
                print("no conseguimos llegar en 2500 steps")
            
            rewards[agent_id] = reward
            dones[agent_id] = done
            self.termination_cause[agent_id] = cause
        dones["__all__"] = all(dones.values())
        return rewards, dones
    

    def reset(self, agent_ids=None, same_position=False):
        """Reseteo del entorno
        reseteo solo del agente especificado."""
        if agent_ids is None:
            self.current_step = 0
            agent_ids = self.agent_id
        
        spawn_points = self.CARLA.world.get_map().get_spawn_points()
        
        for agent_id in agent_ids:
            agent_idx = self.agent_id.index(agent_id)
            agent = self.__agent[agent_idx]
            self.safe_stop_counter[agent_id] = 0     
            #limpiar estado por agente
            self.last_valid_cam.pop(agent_id, None)
            self.cam_features.pop(agent_id, None)
            self.smoothed_radar.pop(agent_id, None)
            self.prev_dist_to_goal.pop(agent_id, None)
            self.steer.pop(agent_id, None)
            self.last_steer.pop(agent_id, None)
            self.low_speed_streak.pop(agent_id, None)
            self.offroad_streak.pop(agent_id, None)

            self.closest_waypoint_idx[agent_id] = 0
            self.last_waypoint_idx[agent_id] = 0
            #spawn point aleatorio en diferentes episodios, el mismo en cada paso
            if not same_position:
                if spawn_points:
                    vehicles = self.CARLA.world.get_actors().filter('vehicle.*')
                    max_attempts = 15
                    safe_distance = 8.0

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

                    self.better_distance[agent_id] = np.inf
                    self.planner[agent_id] = self.CARLA.route_planner.trace_route(
                                                            chosen_spawn.location,
                                                            self.goal_positions[agent_id].location)

                    if self.planner[agent_id]:
                        wp0 = self.planner[agent_id][0][0].transform
                        start = carla.Transform(
                            carla.Location(x=wp0.location.x, y=wp0.location.y,
                                           z=wp0.location.z + 0.3),
                            wp0.rotation)
                    else:
                        start = chosen_spawn

                    self.position_change[agent_idx] = start
                    agent.set_transform(start)
                    self.initial_dist[agent_id] = sum(
                        self.planner[agent_id][j][0].transform.location.distance(self.planner[agent_id][j+1][0].transform.location)
                        for j in range(0, len(self.planner[agent_id])-1)
                    )
                    self.route_xy[agent_id] = np.array(
                        [[wp.transform.location.x, wp.transform.location.y]
                         for wp, _ in self.planner[agent_id]], dtype=np.float64)
                    #con esto vamos a saber si el vehiculo que detectemos esta en un punto de nuestra ruta
                    steps = np.linalg.norm(np.diff(self.route_xy[agent_id], axis=0), axis=1)
                    spacing = float(np.median(steps)) if len(steps) else 2.0
                    self.route_window[agent_id] = int(np.ceil(self.radar_range / max(0.5, spacing))) + 4
                    self._apply_collision_curriculum(agent_id)
                    
            else:
                respawn_transform = carla.Transform(carla.Location(x =  self.position_change[agent_idx].location.x,
                                                                    y = self.position_change[agent_idx].location.y,
                                                                    z = self.position_change[agent_idx].location.z + 0.3,),
                                                                    self.position_change[agent_idx].rotation,)
                
                max_ticks = 15
                for _ in range(max_ticks):
                    print(f"block exist for agent {agent_id} ?")
                    blocked = [v for v in self.CARLA.world.get_actors().filter('vehicle.*')
                                if v.id != agent.id and v.get_location().distance(respawn_transform.location) < 8.0]
                    if not blocked:
                        print(f"Not block exist for agent {agent_id}")
                        break

                    self.CARLA.tick()
                

                protected = {a.id for a in self.__agent}
                protected.update(n.id for n in self.curriculum_npc.values()
                                 if n is not None and n.is_alive)
                blocked = [v for v in self.CARLA.world.get_actors().filter('vehicle.*')
                            if v.id != agent.id and v.id not in protected
                            and v.get_location().distance(respawn_transform.location) < 8.0]
                for v in blocked:
                    print(f"Destorying NPC so the agent {agent_id} could be respawned")
                    if v in self.CARLA.vehicles_npcs_list:
                        self.CARLA.vehicles_npcs_list.remove(v)
                    v.destroy()

                if blocked:
                    self.CARLA.tick()
                agent.set_transform(respawn_transform)
            
            agent.set_target_velocity(carla.Vector3D(0, 0, 0))
            self.CARLA.tick()

        dead_npcs = [v for v in self.CARLA.vehicles_npcs_list if not v.is_alive]
        for npc in dead_npcs:
            try:
                npc.destroy()
            except:
                pass
        
        self.CARLA.vehicles_npcs_list = [v for v in self.CARLA.vehicles_npcs_list if v.is_alive]
        missing = self.CARLA.vehicles_npcs - len(self.CARLA.vehicles_npcs_list)
        if missing > 0:
            self.CARLA.spawn_vehicle(True, count=missing)

        self.CARLA.tick()


        for aid in agent_ids:
            self.CARLA.reset_collision(self.__agent[self.agent_id.index(aid)])

        obs = self.__get_obs()
        for aid in agent_ids:
            print(f"[RESPAWN] {aid} lat={self.lateral_distance[aid]:+.2f}m "
                  f"min_dist={self.route_min_dist.get(aid, -1.0):.2f}m")
        return obs
    
    
    
    def _in_path_weight(self, offset_m):
        """Cuanto nos importa un obstaculo.
        Si va en sentido opuesto no le damos importancia"""
        return float(np.clip(1.0 - max(0.0, abs(offset_m) - 1.0) / 2.0, 0.0, 1.0))

    def _smooth_radar(self, agent_id, features, alpha=0.5):
        """suavizamos la variación entre ticks"""
        features = np.nan_to_num(np.asarray(features, dtype=np.float32),
                                 nan=0.0, posinf=0.0, neginf=0.0)
        prev = self.smoothed_radar.get(agent_id)
        if prev is None or not np.isfinite(prev).all():
            smoothed = features.copy()
        else:
            smoothed = alpha * features + (1.0 - alpha) * prev
        smoothed = np.nan_to_num(smoothed, nan=0.0, posinf=0.0, neginf=0.0)
        self.smoothed_radar[agent_id] = smoothed
        return smoothed.astype(np.float32)

    def _get_vehicle_features(self, agent, agent_id):
        """Detectamos peatones y vehiculos NPC delante nuestra para aprender a frenar.
        Devolvemos los dos mas peligrosos"""
        transform = agent.get_transform()
        location = transform.location
        ego_vel = agent.get_velocity()

  
        if not np.isfinite([location.x, location.y, location.z,
                            transform.rotation.yaw,
                            ego_vel.x, ego_vel.y]).all():
            self.smoothed_radar.pop(agent_id, None)
            return np.zeros(6, dtype=np.float32)

        yaw_rad = np.radians(transform.rotation.yaw)
        forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
        right = np.array([-np.sin(yaw_rad), np.cos(yaw_rad)])
        ego_speed_vec = np.array([ego_vel.x, ego_vel.y])

        scored = []
        seen_ids = {agent.id}

        route_xy = self.route_xy.get(agent_id)
        if route_xy is None or len(route_xy) < 2:
            route_xy = None
            wp_idx = 0
        else:
            wp_idx = int(np.clip(self.closest_waypoint_idx.get(agent_id, 0), 0, len(route_xy) - 1))
        route_win = self.route_window.get(agent_id, 20)

        vehicles_to_check = []
        for npc in self.CARLA.vehicles_npcs_list:
            if npc.is_alive and npc.id not in seen_ids:
                vehicles_to_check.append(npc)
                seen_ids.add(npc.id)
        #no queremos tener en cuenta nuestro propio agente
        for other in self.__agent:
            if other.id not in seen_ids:
                vehicles_to_check.append(other)
                seen_ids.add(other.id)
        for npc in self.curriculum_npc.values():
            if npc is not None and npc.is_alive and npc.id not in seen_ids:
                vehicles_to_check.append(npc)
                seen_ids.add(npc.id)
        for walker, _ctrl in self.CARLA.people_list:
            if walker.is_alive and walker.id not in seen_ids:
                vehicles_to_check.append(walker)
                seen_ids.add(walker.id)

        for v in vehicles_to_check:
            v_loc = v.get_location()
            dx = v_loc.x - location.x
            dy = v_loc.y - location.y
            dz = v_loc.z - location.z
            dist = np.sqrt(dx * dx + dy * dy)


            if not np.isfinite([dx, dy, dz, dist]).all():
                continue
            if abs(dz) > 5.0:
                continue

            if dist > self.radar_range or dist < 1.0:
                continue

            #no nos interesa si esta detrás nuestra, ventana de unos 80 grados
            direction = np.array([dx, dy]) / (dist + 1e-6)
            fwd_proj = np.dot(direction, forward)
            if fwd_proj < 0.2:
                continue

            if route_xy is not None:
                lo = max(0, wp_idx - 2)
                hi = min(len(route_xy), wp_idx + route_win)
                win = route_xy[lo:hi]
                if len(win) < 2:
                    continue
                #cuanto se desvia de nuestro carril el obstaculo
                d_wp = np.linalg.norm(win - np.array([v_loc.x, v_loc.y]), axis=1)
                k = int(np.argmin(d_wp))
                tan = win[min(k + 1, len(win) - 1)] - win[max(k - 1, 0)]
                tan = tan / (np.linalg.norm(tan) + 1e-6)
                to_v = np.array([v_loc.x, v_loc.y]) - win[k]
                signed_offset = -tan[1] * to_v[0] + tan[0] * to_v[1]
            else:
                signed_offset = np.dot(direction, right) * dist

            if not np.isfinite(signed_offset):
                continue

            in_path = self._in_path_weight(signed_offset)
            if in_path <= 0.0:
                continue  #a mas de un carril de mi ruta: no me afecta

            proximity = np.clip(1.0 - dist / self.radar_range, 0.0, 1.0)
            bearing = np.clip(signed_offset / self.lane_width, -1.0, 1.0)  #en anchos de carril

            #hacemos un score para ver cual nos importan mas segun la cercania y el carril
            v_vel = v.get_velocity()
            rel_vel = ego_speed_vec - np.array([v_vel.x, v_vel.y])
            closing_speed = np.dot(rel_vel, direction)
            closing_rate = np.clip(closing_speed / self.max_speed, -1.0, 1.0)
            score = proximity * in_path
            scored.append((score, proximity, bearing, closing_rate))

        features = np.zeros(6, dtype=np.float32)
        scored.sort(key=lambda x: x[0], reverse=True)

        if len(scored) >= 1:
            _, features[0], features[1], features[2] = scored[0]
        if len(scored) >= 2:
            _, features[3], features[4], features[5] = scored[1]

        return self._smooth_radar(agent_id, features)

    def _update_curriculum_release(self, agent_id, agent):
        """Cuenta cuando nuestro agente está detenido y suelta el NPC cuando se completa la espera."""

        if self.curriculum_released.get(agent_id, False):
            self.curriculum_cooldown[agent_id] = self.curriculum_cooldown.get(agent_id, 0) + 1
            if self.curriculum_cooldown[agent_id] >= self.curriculum_rearm_ticks:
                route = self.planner.get(agent_id)
                idx = self.closest_waypoint_idx.get(agent_id, 0)
                moved = idx - self.curriculum_release_idx.get(agent_id, 0)
                if route and idx + 25 < len(route) and moved >= 10:
                    self._apply_collision_curriculum(agent_id, from_idx=idx, force=True)
                    self.curriculum_cooldown[agent_id] = 0
            return

        npc = self.curriculum_npc.get(agent_id)
        if npc is None or not npc.is_alive:
            return

        npc_loc = npc.get_location()
        agent_loc = agent.get_location()
        if abs(npc_loc.z - agent_loc.z) > 5.0:  
            return

        dx = npc_loc.x - agent_loc.x
        dy = npc_loc.y - agent_loc.y
        dist = np.sqrt(dx * dx + dy * dy)
        #nuestro coche aguanta a baja velocidad o lejos, sin colisionar, bien hecho, lo retiramos
        if not np.isfinite(dist) or dist > self.curriculum_release_dist \
                or self.velocity.get(agent_id, 0.0) >= 0.5:
            self.safe_stop_counter[agent_id] = 0
            return

        self.safe_stop_counter[agent_id] = self.safe_stop_counter.get(agent_id, 0) + 1
        #nuestro vehiculo no ha aprendido, lo dejamos que siga un poco la ruta para que siga conduciendo
        #y espawneamos de nuevo el agente mas adelante
        if self.safe_stop_counter[agent_id] >= self.curriculum_hold_ticks.get(agent_id, 200):
            try:
                npc.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
                npc.set_autopilot(True, self.CARLA.traffic_manager.get_port())
                self.CARLA.traffic_manager.ignore_lights_percentage(npc, 0)
                self.CARLA.traffic_manager.vehicle_percentage_speed_difference(npc, 20)
                self.curriculum_released[agent_id] = True
                self.curriculum_release_idx[agent_id] = self.closest_waypoint_idx.get(agent_id, 0)
                held_s = self.safe_stop_counter[agent_id] * self.CARLA.fixed_delta_seconds
                print(f"{agent_id} mantuvo la parada {held_s:.1f}s, NPC released")
            except Exception as e:
                print(f"[curriculum] no se pudo liberar el NPC de {agent_id}: {e}")

    def _apply_collision_curriculum(self, agent_id, from_idx=0, force=False):
        """Colocamos un obstaculo por delante del agente para que aprenda a frenar."""
        route = self.planner[agent_id]

        if not route or len(route) < 2:
            self._retire_curriculum_npc(agent_id)
            return
        if not force and np.random.rand() >= self.curriculum_prob:
            self._retire_curriculum_npc(agent_id)
            return

        target_dist = np.random.uniform(20.0, 35.0) 
        cum_dist = 0.0
        chosen_wp = route[-1][0]
        chosen_idx = len(route) - 1
        for j in range(from_idx, len(route) - 1):
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
                print(f"couldn't spawn stopped NPC for {agent_id}, skipping this episode")
                return

            npc = spawned
            self.curriculum_npc[agent_id] = npc

        else:
            npc.set_autopilot(False)
            npc.set_simulate_physics(False)
            loc = chosen_wp.transform.location
            rot = chosen_wp.transform.rotation
            npc.set_transform(carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.3), rot
            ))
        npc.set_simulate_physics(True)
        npc.set_autopilot(False)
        npc.set_target_velocity(carla.Vector3D(0, 0, 0))
        npc.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))

        dt = self.CARLA.fixed_delta_seconds
        self.curriculum_hold_ticks[agent_id] = int(
            np.random.uniform(self.curriculum_hold_min_s, self.curriculum_hold_max_s) / dt)
        self.curriculum_released[agent_id] = False
        self.safe_stop_counter[agent_id] = 0

    def _retire_curriculum_npc(self, agent_id):
        """Quitamos los vehiculos que hemos puesto para aprender a frenar 
        si se cumple el requisito"""
        npc = self.curriculum_npc.pop(agent_id, None)
        if npc is not None and npc.is_alive:
            npc.set_autopilot(False)
            try:
                npc.destroy()
            except Exception as e:
                print(f"no se pudo destruir el NPC de {agent_id}: {e}")
        self.curriculum_released[agent_id] = False
        self.safe_stop_counter[agent_id] = 0
        self.curriculum_cooldown[agent_id] = 0



    def close(self):
        """Cierre de entrenamiento.
        Matamos los NPCs y actores"""
        for npc in self.curriculum_npc.values():
            try:
                if npc.is_alive:
                    npc.destroy()
            except Exception:
                pass
        self.CARLA.destroy_actors()
