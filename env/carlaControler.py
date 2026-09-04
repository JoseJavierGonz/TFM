import sys
import os
import carla
import time
import random
import numpy as np
import cv2
import pynput
import threading
from queue import Queue, Empty, Full
from agents.navigation.global_route_planner import GlobalRoutePlanner

class CarlaControler():
    """Clase para conectarse al servidor de carla, setear peatones, mapa, coches, condiciones meteorológicas, spawnear sensores y demás"""
    def __init__(self, num_vehicles=20, num_walkers=20, enable_radar=False):

        self.client = None
        self.world = None
        self.sensors = {}
        self.sensors_data = {}
        self.radar_queues = {}
        self.enable_radar = enable_radar
        self.radar_data = {}
        self.vehicles_npcs_list = []
        self.vehicles_marl_list = []
        self.people_list = []
        self.collision_occurs = {}
        self.camera_mode = 0


        try:
            #CONEXION CON EL SERVIDOR
            print("Connecting to CARLA server...")
            self.client = carla.Client('carla-engine', 2000)
            self.client.set_timeout(20.0)

            #SETEAMOS EL MAPA QUE QUEREMOS USAR
            print("Loading world...")
            self.world = self.client.load_world("Town10HD")

            #Rendering mode
            self.fixed_delta_seconds = 0.05
            self._original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            #settings.no_rendering_mode = False
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_delta_seconds
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substep = 10
            self.world.apply_settings(settings)            

            #SETEAMOS EL CLIMA
            print("Setting weather...")
            self.weather = self.world.get_weather()
            self.weather_values()
            self.world.set_weather(self.weather)

            #SETEAMOS VEHICULOS Y PERSONAS
            self.vehicles_npcs = num_vehicles
            print("Spawning vehicles...")
            
            #Configurar Traffic Manager para los npcs
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_random_device_seed(0)  #Comportamiento más predecible
            
            self.spawn_vehicle(False)

            self.people = num_walkers
            print("Spawning pedestrians...")
            self.spawn_people()

            #cargamos un planificador de rutas
            self.route_planner = GlobalRoutePlanner(self.world.get_map(), 2.0)
            print("Global Planner initialized")

            #Posiblidad de pasar un argumento según la vista que queramos tener del entorno(buscar si se podría seguir a nuestros vehiculos)
            #en el servidor de la universidad perdemos esto ya que carla esta fuera del contenedor
            print("Setting camera view...")
            spectator = self.world.get_spectator()
            self.map_view(spectator)



        except Exception as e:
            print(f"Error initializing CARLA controller: {e}")
            self.client = None
            self.world = None




    def weather_values(self):
        """Set de los parámetros de condiciones meteorológicas"""
        try:
            self.weather.precipitation = 80
            self.weather.precipitation_deposits = 70
            self.weather.cloudiness = 80
        except Exception as e:
            print(f"Error setting weather values: {e}")


    def which_camera(self, key):
        """Cambio de camara en caliente"""
        try:
            if key.char == '0':
                self.camera_mode = 0
            elif key.char == '1':
                self.camera_mode = 1
            elif key.char == '2':
                self.camera_mode = 2
        except AttributeError:
            pass

        except Exception as e:
            print(f"Error changing the camara view {e}")
        
    def map_view(self, spectator):
        """Set camara de espectador"""
        try:
            transform = carla.Transform(
                carla.Location(x=0, y=0, z=150),  
                carla.Rotation(pitch=-90)        
            )
            spectator.set_transform(transform)
        except Exception as e:
            print(f"Error setting spectator view: {e}")

    
    def follow_vehicle(self, vehicle):
        """Actualizador de la cámara personalizada de cada vehiculo"""
        try:
            spectator = self.world.get_spectator()
            transform = vehicle.get_transform()
            forward_vector = transform.get_forward_vector()
            camera_location = carla.Location(
                x=transform.location.x - forward_vector.x * 20,
                y=transform.location.y - forward_vector.y * 20,
                z=transform.location.z + 10
            )
            spectator.set_transform(carla.Transform(
                camera_location,
                carla.Rotation(pitch=-20, yaw=transform.rotation.yaw)
            ))
        except Exception as e:
            print(f"Error following vehicle: {e}")


    def spawn_people(self):
        """Set peatones en el mapa"""
        if not self.world:
            print("World not initialized")
            return
        
        try:

            walker_blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            if not walker_blueprints:
                print("No pederestian found")
                return
            
            control_blueprints = self.world.get_blueprint_library().find('controller.ai.walker')
            if not control_blueprints:
                print("No walker conroler found")
            for _ in range(self.people):
                try:
                    spawn_points = self.world.get_random_location_from_navigation()
                    if spawn_points:
                        blueprint = random.choice(walker_blueprints)
                        transform_location = carla.Transform(spawn_points, carla.Rotation(yaw=random.uniform(0, 360)))
                        walker_actor = self.world.try_spawn_actor(blueprint, transform_location)
                        if walker_actor:
                            controller = self.world.try_spawn_actor(control_blueprints, carla.Transform(), attach_to=walker_actor)
                            if controller:
                                controller.start()
                                controller.set_max_speed(2.0)
                                self.people_list.append((walker_actor, controller))
                            else:
                                walker_actor.destroy()

                except:
                    print("Failed spawned pedestrian")
            
        except Exception as e:
            print(f"Error in spawn people {e}")
    


    def spawn_vehicle(self, need_npcs=False, count=None):
        """Spawn NPC and MARL en el mapa"""
        #la idea es meter todo tipo de vehiculos pero controlar 2
        #proximos trabajos podría ser elevar el número de vehiculos a controlar
        if not self.world:
            print("World not initialized")
            return
        
        if need_npcs == False:
            vehicles_MARL=['vehicle.ford.mustang', 'vehicle.tesla.model3']
            for marl_idx, vehicle in enumerate(vehicles_MARL):
                try:
                    blueprint_marl=self.world.get_blueprint_library().find(vehicle)
                    if not blueprint_marl:
                        print("Vehicle MARL not found")
                        continue
                    if blueprint_marl.has_attribute('role_name'):
                        blueprint_marl.set_attribute('role_name', f'marl_agent_{marl_idx}')
                    
                    actor = None
                    attempts = 0
                    max_attempts = 5
                    while actor is None and attempts < max_attempts :
                        location = random.choice(self.world.get_map().get_spawn_points())
                        actor = self.world.try_spawn_actor(blueprint_marl, location)
                        attempts += 1

                    if actor:
                        self.vehicles_marl_list.append(actor)
                        self.initialize_sensors(actor)
                        print(f"MARL vehicle spawned: {vehicle}")
                    else:
                        print(f"Failed to spawn MARL vehicle: {vehicle} (location busy?)")

                except Exception as e:
                    print(f"Failed spawned MARL vehicles {e}")
            
            print(f"Total MARL vehicles: {len(self.vehicles_marl_list)}")
        try:
            vehicles_npcs_blueprint = self.world.get_blueprint_library().filter('vehicle')
            if not vehicles_npcs_blueprint:
                print("Vehicle not found")
            #count = los vehiculos que faltan hasta el numero que se setee, ya que van muriendo y desapareciendo del CARLA en 
            #algunos casos
            n_spawn = self.vehicles_npcs if count is None else max(0, int(count))
            for _ in range(n_spawn):
                blueprint = random.choice(vehicles_npcs_blueprint)
                random_points = random.choice(self.world.get_map().get_spawn_points())
                actor = self.world.try_spawn_actor(blueprint, random_points)
                if actor:
                    actor.set_autopilot(True, self.traffic_manager.get_port())
                    # Configurar comportamiento individual
                    self.traffic_manager.ignore_lights_percentage(actor, 0)  #Respetar semáforos
                    self.traffic_manager.distance_to_leading_vehicle(actor, 2.5)
                    self.traffic_manager.vehicle_percentage_speed_difference(actor, 20)  #20% más lento 
                    self.vehicles_npcs_list.append(actor)
        except Exception as e:
            print(f"Failed spawned vehicles {e}")


    def initialize_sensors(self, actor):
        """Inicialización de sensores"""
        blueprint_librariy = self.world.get_blueprint_library()
        collision = blueprint_librariy.find('sensor.other.collision')

        collision_sensor = self.world.spawn_actor(collision, carla.Transform(), attach_to=actor)

        actor_id = actor.id
        self.collision_occurs[actor_id] = False

        if actor_id in self.sensors:
            print(f"{actor_id} saved previously")
            return

        self.sensors[actor_id] = {'collision': collision_sensor}
        collision_sensor.listen(lambda data, v=actor_id: self.__on_collision(v, data))

        if not self.enable_radar:
            return
        #spawneamos el radar
        radar_bp = blueprint_librariy.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '60')
        radar_bp.set_attribute('vertical_fov', '20')
        radar_bp.set_attribute('range', '30')
        radar_bp.set_attribute('points_per_second', '2000')
        radar_bp.set_attribute('sensor_tick', str(self.fixed_delta_seconds))
        radar_tf = carla.Transform(carla.Location(x=2.2, z=0.8), carla.Rotation(pitch=0.0))
        radar_sensor = self.world.spawn_actor(radar_bp, radar_tf, attach_to=actor)

        self.sensors[actor_id]['radar'] = radar_sensor
        self.radar_queues[actor_id] = Queue(maxsize=4)
        radar_sensor.listen(lambda data, v=actor_id: self.__radar_callback(data, v))

    
    def get_map(self):
        """Getter del mapa de CARLA"""
        return self.world.get_map()
    
    def __radar_callback(self, radar_data, actor_id):
        if getattr(self, "closing", False):
            return
        q = self.radar_queues.get(actor_id)
        if q is None:
            return
        if q.full():
            try:
                q.get_nowait()
            except Empty:
                pass
        try:
            q.put_nowait(radar_data)
        except Full:
            pass

    def tick(self):
        """Tick de la simuladion."""
        if getattr(self, "closing", False):
            return
        self.world.tick()
        if not self.enable_radar:
            return
        #nos quedamos con la ultima medida de cada radar
        for actor_id, q in self.radar_queues.items():
            latest = None
            while not q.empty():
                try:
                    latest = q.get_nowait()
                except Empty:
                    break
            if latest is None:
                continue
            #se guarda el transform del sensor
            self.radar_data[actor_id] = {
                'transform': latest.transform,
                'detections': [{'depth': d.depth, 'azimuth': d.azimuth,
                                'altitude': d.altitude, 'velocity': d.velocity}
                               for d in latest],
            }
    

    def __on_collision(self, vehicle_id, measure):
        """Llamada si hay colision"""
        self.collision_occurs[vehicle_id] = True



    def reset_collision(self, vehicle):
        """Resetea la colision de los vehiculos en cada episodio"""
        if getattr(self, "closing", False):
            return
        if vehicle.id in self.collision_occurs:
            self.collision_occurs[vehicle.id] = False
            

    
    def destroy_actors(self):
        """Destructor"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        print(f"RSS before: {process.memory_info().rss / 1024**2:.1f} MB")
        print("WARNING: destroy_actors() called!")
        print("sensors:", sum(len(v) for v in self.sensors.values()))
        print("Sensor data:", len(self.sensors_data))
        

        self.closing = True
        for actor_id, sensors in self.sensors.items():
            for sensor in sensors.values():
                try:
                    sensor.stop()
                except:
                    pass
        time.sleep(0.5)
        for actor_id, sensors in self.sensors.items():
            for sensor in sensors.values():
                try:
                    sensor.destroy()
                except:
                    pass
        self.radar_queues.clear()
        self.sensors.clear()
        self.sensors_data.clear()
                
        for walker, controller in self.people_list:
            try:
                controller.stop()
                controller.destroy()
                walker.destroy()
            except Exception as e:
                pass
        self.people_list.clear()

        for actor in self.vehicles_npcs_list:
            try:
                actor.set_autopilot(False)
                actor.destroy()
                print("Actors destroyed")
            except Exception as e:
                print(f"Error destroying actors {e}")
        self.vehicles_npcs_list.clear()

        for actor in self.vehicles_marl_list:
            try:
                actor.destroy()
                print("Actors destroyed")
            except Exception as e:
                print(f"Error destroying actors {e}")
        self.vehicles_marl_list.clear()

        print("WARNING: destroy_actors() called!")
        print("sensors:", sum(len(v) for v in self.sensors.values()))
        print("Sensor data:", len(self.sensors_data))
        print(f"RSS after: {process.memory_info().rss / 1024**2:.1f} MB")

        try:
            if self.world is not None and getattr(self, "_original_settings", None) is not None:
                self.world.apply_settings(self._original_settings())
            if getattr(self, "traffic_manager", None) is not None:
                self.traffic_manager.set_synchronous_mode(False)
        except Exception as e:
            print(f"Error restoring world settings: {e}")

