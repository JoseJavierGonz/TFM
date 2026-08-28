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


# Paleta de colores para visualizar segmentación semántica de CARLA
SEG_PALETTE = {
    0:  (0, 0, 0),         # Unlabeled
    12:  (220, 20, 60),     # Pedestrian (rojo)
    24:  (157, 234, 50),    # RoadLine (verde claro)
    1:  (128, 64, 128),    # Road (morado)
    24:  (244, 35, 232),    # SideWalk (rosa)
    14: (0, 0, 142),       # Vehicle (azul)
    7: (250, 170, 30),    # TrafficLight (naranja)
}


class CarlaControler():
    """Class to connect with CARLA server, set the weather parameters, maps, cars and other simulator configurations"""
    def __init__(self):

        self.client = None
        self.world = None
        self.sensors = {}
        self.sensors_data = {}
        self.camera_queues = {}
        self.vehicles_npcs_list = []
        self.vehicles_marl_list = []
        self.people_list = []
        self.collision_occurs = {}
        self.__camera_miss_streak = {}
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
            # settings.no_rendering_mode = False
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_delta_seconds
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.maz_substep = 10
            self.world.apply_settings(settings)            

            #SETEAMOS EL CLIMA
            print("Setting weather...")
            self.weather = self.world.get_weather()
            self.weather_values()
            self.world.set_weather(self.weather)

            #SETEAMOS VEHICULOS Y PERSONAS
            self.vehicles_npcs = 20 #numero de vehiculos que tendremos en el mapa
            print("Spawning vehicles...")
            
            #Configurar Traffic Manager para los npcs
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_random_device_seed(0)  # Comportamiento más predecible
            
            self.spawn_vehicle(False)

            self.people = 20 #numero de personas que tendremos en el mapa
            print("Spawning pedestrians...")
            self.spawn_people()

            #cargamos un planificador de rutas
            self.route_planner = GlobalRoutePlanner(self.world.get_map(), 2.0)
            print("Global Planner initialized")

            #Posiblidad de pasar un argumento según la vista que queramos tener del entorno(buscar si se podría seguir a nuestros vehiculos)
            print("Setting camera view...")
            spectator = self.world.get_spectator()
            self.map_view(spectator)



        except Exception as e:
            print(f"Error initializing CARLA controller: {e}")
            self.client = None
            self.world = None




    def weather_values(self):
        """Set weather parameters for rainy conditions"""
        try:
            self.weather.precipitation = 80
            self.weather.precipitation_deposits = 70
            self.weather.cloudiness = 80
        except Exception as e:
            print(f"Error setting weather values: {e}")


    def which_camera(self, key):
        """Warm view change"""
        try:
            if key.char == '0':
                self.camera_mode = 0
                # spectator = self.world.get_spectator()
                # self.map_view(spectator)
            elif key.char == '1':
                self.camera_mode = 1
                # self.follow_vehicle(self.vehicles_marl_list[0])
            elif key.char == '2':
                self.camera_mode = 2
                #self.follow_vehicle(self.vehicles_marl_list[1])

        except AttributeError:
            pass

        except Exception as e:
            print(f"Error changing the camara view {e}")
        
    def map_view(self, spectator):
        """Set spectator camera"""
        try:
            transform = carla.Transform(
                carla.Location(x=0, y=0, z=150),  
                carla.Rotation(pitch=-90)        
            )
            spectator.set_transform(transform)
        except Exception as e:
            print(f"Error setting spectator view: {e}")

    
    def follow_vehicle(self, vehicle):
        """Update spectator to follow a vehicle"""
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
        """Set pederestian in the map"""
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
    


    def spawn_vehicle(self, need_npcs=False):
        """Set NPC and MARL vehicles in the map"""
        #la idea es meter todo tipo de vehiculos pero controlar 2
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
            for _ in range(self.vehicles_npcs):
                blueprint = random.choice(vehicles_npcs_blueprint)
                random_points = random.choice(self.world.get_map().get_spawn_points())
                actor = self.world.try_spawn_actor(blueprint, random_points)
                if actor:
                    actor.set_autopilot(True, self.traffic_manager.get_port())
                    # Configurar comportamiento individual
                    self.traffic_manager.ignore_lights_percentage(actor, 0)  # Respetar semáforos
                    self.traffic_manager.distance_to_leading_vehicle(actor, 2.5)
                    self.traffic_manager.vehicle_percentage_speed_difference(actor, -20)  # 20% más lento (más seguro)
                    self.vehicles_npcs_list.append(actor)
        except Exception as e:
            print(f"Failed spawned vehicles {e}")


    def initialize_sensors(self, actor):
        blueprint_librariy = self.world.get_blueprint_library()
        camera = blueprint_librariy.find('sensor.camera.semantic_segmentation')
        lidar = blueprint_librariy.find('sensor.lidar.ray_cast')
        collision = blueprint_librariy.find('sensor.other.collision')

        # #configurar cámara
        camera.set_attribute('image_size_x', '128')
        camera.set_attribute('image_size_y', '128')
        camera.set_attribute('fov', '90')

        # #configurar lidar
        # lidar.set_attribute('channels', '16')
        # lidar.set_attribute('range', '30.0')
        # lidar.set_attribute('points_per_second', '28000')
        # lidar.set_attribute('rotation_frequency', '10')
        
        camera_transform = carla.Transform(
            carla.Location(x=2.0, z=1.0), 
            carla.Rotation(pitch=0.0)
        )
        # lidar_transform = carla.Transform(
        #     carla.Location(x=0.0, z=2.5),
        #     carla.Rotation()
        # )
        camera = self.world.spawn_actor(camera, camera_transform, attach_to=actor)
        # lidar = self.world.spawn_actor(lidar, lidar_transform, attach_to=actor)
        collision_sensor = self.world.spawn_actor(collision, carla.Transform(), attach_to=actor)
        
        actor_id = actor.id
        #guardar una referencia para ver si el coche colisiona
        self.collision_occurs[actor_id] = False
        

        if actor_id in self.sensors:
            print(f"{actor_id} saved previously")
        else:
            #self.sensors[actor] = {'camera':camera, 'lidar':lidar, 'collision':collision_sensor}
            self.sensors[actor_id] = {'camera': camera, 'collision': collision_sensor}
            self.sensors_data[actor_id] = {'camera_data': None, 'lidar_data': None}

            # Cola por sensor para sincronízación estricta tick<->frame
            self.camera_queues[actor_id] = Queue(maxsize=4)
            camera.listen(lambda image, actor_id=actor_id : self.__camera_callback(image, actor_id))
            # lidar.listen(lambda data, v=actor: self.__lidar_buffer(v,data))
            collision_sensor.listen(lambda data, v=actor_id: self.__on_collision(v, data))


    def __camera_callback(self, image, actor_id):
        if getattr(self, "closing", False):
            return
        if actor_id not in self.camera_queues:
            return
        q = self.camera_queues[actor_id]
        if q.full():
            try:
                q.get_nowait()
            except Empty:
                pass
        try:
            q.put_nowait(image)
        except Full:
            pass


    def get_sensor_data(self, vehicle):
        """Get latest sensor data for a vehicle"""
        if vehicle.id in self.sensors_data:
            return self.sensors_data[vehicle.id]
        return {'camera_data': None, 'lidar_data': None}
    
    def get_map(self):
        """Get CARLA map"""
        return self.world.get_map()
    
    def tick(self):
        """Advance simulation by one tick and drain sensor queues with strict frame sync."""
        camera_timeout = 0.2
        max_attempts = 3
        warn_atfer_misses = 20
        if getattr(self, "closing", False):
            return
        frame = self.world.tick()
        for actor_id, q in self.camera_queues.items():
            got_frame = False
            while True:

                try:
                    data = q.get(timeout=2.0)
                except Empty:
                    print(f"[WARNING] Camera timeout for actor {actor_id} (frame {frame})")
                    break

                if data.frame < frame:
                    continue

                self.__save_camera_data(actor_id, data)
                got_frame = True
                break
            if got_frame:
                self.__camera_miss_streak[actor_id] = 0
            else:
                self.__camera_miss_streak[actor_id] = self.__camera_miss_streak.get(actor_id, 0) + 1
                if self.__camera_miss_streak[actor_id] == warn_atfer_misses:
                    print(f"camera for {actor_id} misses after consecutive ticks, frame {frame}")
    

    def __on_collision(self, vehicle_id, measure):
        """Call if collision occurs"""
        self.collision_occurs[vehicle_id] = True



    def reset_collision(self, vehicle):
        """Reset collision flag for a vehicle in each new episodie"""
        if getattr(self, "closing", False):
            return
        if vehicle.id in self.collision_occurs:
            self.collision_occurs[vehicle.id] = False
            
    
    def __lidar_buffer(self, vehicle, measure):
        """Process lidar data"""
        try:
            raw_data = measure.raw_data
            data = np.frombuffer(raw_data, dtype=np.float32)
            data_lidar = np.reshape(data, (-1, 4))
            if len(data_lidar) < 1000:
                padding = np.zeros((1000 - len(data_lidar), 4), dtype=np.float32)
                data_lidar = np.vstack([data_lidar, padding])
            else:
                data_lidar = data_lidar[:1000]
        
            self.sensors_data[vehicle]['lidar_data'] = data_lidar
        except Exception as e:
            print(f"Error processing lidar data: {e}")
  
            
    
    def __save_camera_data(self, vehicle, measure):
        """Process semantic segmentation data: extract class IDs (R channel of BGRA)."""
        try:
            raw = np.frombuffer(measure.raw_data, dtype=np.uint8)
            raw = np.reshape(raw, (measure.height, measure.width, 4))
            # semantic_segmentation: class ID en canal R (índice 2 en BGRA)
            class_ids = raw[:, :, 2]
            if class_ids.shape != (128, 128):
                class_ids = cv2.resize(class_ids, (128, 128), interpolation=cv2.INTER_NEAREST)
            self.sensors_data[vehicle]['camera_data'] = class_ids.astype(np.uint8)
        except Exception as e:
            print(f"Error processing camera data: {e}")

    def save_seg_debug(self, vehicle, path):
        """Guarda la última segmentación del vehículo coloreada para depuración visual."""
        vehicle_id = vehicle.id if hasattr(vehicle, 'id') else vehicle
        data = self.sensors_data.get(vehicle_id, {}).get('camera_data')
        if data is None:
            return False
        data_clean = data.copy()
        data_clean[105:, :] = 0
        H, W = data_clean.shape
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        for k, color in SEG_PALETTE.items():
            img[data_clean == k] = color
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True

       
    
    def destroy_actors(self):
        """Destructor"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        print(f"RSS before: {process.memory_info().rss / 1024**2:.1f} MB")
        print("WARNING: destroy_actors() called!")
        print("sensors:", sum(len(v) for v in self.sensors.values()))
        print("queues:", len(self.camera_queues))
        print("Sensor data:", len(self.sensors_data))
        

        self.closing = True
        for actor_id, sensors in self.sensors.items():
            for sensor in sensors.values():
                try:
                    #if sensor.is_alive:
                    sensor.stop()
                except:
                    pass
        time.sleep(0.5)
        for actor_id, sensors in self.sensors.items():
            for sensor in sensors.values():
                try:
                    #if sensor.is_alive:
                    sensor.destroy()
                except:
                    pass
        self.camera_queues.clear()
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
                #if actor.is_alive:
                actor.set_autopilot(False)
                actor.destroy()
                print("Actors destroyed")
            except Exception as e:
                print(f"Error destroying actors {e}")
        self.vehicles_npcs_list.clear()

        for actor in self.vehicles_marl_list:
            try:
                #if actor.is_alive:
                actor.destroy()
                print("Actors destroyed")
            except Exception as e:
                print(f"Error destroying actors {e}")
        self.vehicles_marl_list.clear()

        print("WARNING: destroy_actors() called!")
        print("sensors:", sum(len(v) for v in self.sensors.values()))
        print("queues:", len(self.camera_queues))
        print("Sensor data:", len(self.sensors_data))
        print(f"RSS after: {process.memory_info().rss / 1024**2:.1f} MB")

        try:
            if self.world is not None and getattr(self, "_original_settings", None) is not None:
                self.world.apply_settings(self._original_settings())
            if getattr(self, "traffic_manager", None) is not None:
                self.traffic_manager.set_synchronous_mode(False)
        except Exception as e:
            print(f"Error restoring world settings: {e}")

