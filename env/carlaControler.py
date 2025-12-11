import carla
import time
import random
import numpy as np


class CarlaControler():
    """Class to connect with CARLA server, set the weather parameters, maps, cars and other simulator configurations"""
    def __init__(self):

        self.client = None
        self.world = None
        self.sensors = {}
        self.sensors_data = {}
        self.vehicles_npcs_list = []
        self.vehicles_marl_list = []
        self.people_list = []

        try:
            #CONEXION CON EL SERVIDOR
            print("Connecting to CARLA server...")
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)

            #SETEAMOS EL MAPA QUE QUEREMOS USAR
            print("Loading world...")
            self.world = self.client.load_world("Town10HD")
            self.client.set_timeout(10.0)

            #SETEAMOS EL CLIMA
            print("Setting weather...")
            self.weather = self.world.get_weather()
            self.weather_values()
            self.world.set_weather(self.weather)
            time.sleep(5)

            #SETEAMOS VEHICULOS Y PERSONAS
            self.vehicles_npcs = 10 #numero de vehiculos que tendremos en el mapa
            print("Spawning vehicles...")
            self.spawn_vehicle()

            self.people = 10 #numero de personas que tendremos en el mapa
            print("Spawning pedestrians...")
            self.spawn_people()

            #Posiblidad de pasar un argumento según la vista que queramos tener del entorno(buscar si se podría seguir a nuestros vehiculos)
            print("Setting camera view...")
            spectator = self.world.get_spectator()
            self.map_view(spectator)
            time.sleep(10)



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
                                self.people_list.append(walker_actor)
                            else:
                                walker_actor.destroy()

                except:
                    print("Failed spawned pedestrian")
            
        except Exception as e:
            print(f"Error in spawn people {e}")
    


    def spawn_vehicle(self):
        """Set NPC and MARL vehicles in the map"""
        #la idea es meter todo tipo de vehiculos pero controlar 2
        if not self.world:
            print("World not initialized")
            return
        
        vehicles_MARL=['vehicle.ford.mustang', 'vehicle.tesla.model3']
        for vehicle in vehicles_MARL:
            try:
                blueprint_marl=self.world.get_blueprint_library().find(vehicle)
                if not blueprint_marl:
                    print("Vehicle MARL not found")
                    return
                location = random.choice(self.world.get_map().get_spawn_points())
                actor = self.world.try_spawn_actor(blueprint_marl, location)
                if actor:
                    self.vehicles_marl_list.append(actor)
                    self.initialize_sensors(actor)

            except Exception as e:
                print(f"Failed spawned MARL vehicles {e}")
        try:
            vehicles_npcs_blueprint = self.world.get_blueprint_library().filter('vehicle')
            if not vehicles_npcs_blueprint:
                print("Vehicle not found")
            for _ in range(self.vehicles_npcs):
                blueprint = random.choice(vehicles_npcs_blueprint)
                random_points = random.choice(self.world.get_map().get_spawn_points())
                actor = self.world.try_spawn_actor(blueprint, random_points)
                if actor:
                    actor.set_autopilot(True, self.client.get_trafficmanager().get_port())
                    self.vehicles_npcs_list.append(actor)
        except Exception as e:
            print(f"Failed spawned vehicles {e}")


    def initialize_sensors(self, actor):
        blueprint_librariy = self.world.get_blueprint_library()
        camera = blueprint_librariy.find('sensor.camera.rgb')
        lidar = blueprint_librariy.find('sensor.lidar.ray_cast')
        camera_transform = carla.Transform(
            carla.Location(x=2.0, z=1.0), 
            carla.Rotation(pitch=0.0)
        )
        lidar_transform = carla.Transform(
            carla.Location(x=0.0, z=2.5),
            carla.Rotation()
        )
        camera = self.world.spawn_actor(camera, camera_transform, attach_to=actor)
        lidar = self.world.spawn_actor(lidar, lidar_transform, attach_to=actor)

        #configurar cámara
        camera.set_attribute('image_size_x', '84')
        camera.set_attribute('image_size_y', '84')
        camera.set_attribute('fov', '90')

        #configurar lidar
        lidar.set_attribute('channels', '16')
        lidar.set_attribute('range', '30.0')
        lidar.set_attribute('points_per_second', '28000')
        lidar.set_attribute('rotation_frequency', '10')

        if actor in self.sensors:
            print(f"{actor} saved previously")
        else:
            self.sensors[actor] = {'camera':camera, 'lidar':lidar}
            self.sensors_data[actor] = {'camera_data': None, 'lidar_data': None}
            
            camera.listen(lambda data, v=actor: self.__save_camera_data(v, data))
            lidar.listen(lambda data, v=actor: self.__lidar_buffer(v,data))


    def get_sensor_data(self, vehicle):
        """Get latest sensor data for a vehicle"""
        if vehicle in self.sensors_data:
            return self.sensors_data[vehicle]
        return {'camera_data': None, 'lidar_data': None}
            
    
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
        """Process camera data"""

        try:
            raw_data = measure.raw_data
            data = np.frombuffer(raw_data, dtype=np.uint8)
            data_camera = np.reshape(data, (measure.height, measure.width, 4))
            data_camera = data_camera[:, :, :3] 
            data_camera = data_camera[:, :, [2, 1, 0]]
            self.sensors_data[vehicle]['camera_data'] = data_camera
        except Exception as e:
            print(f"Error processing camera data: {e}")


        

        
    
    def destroy_actors(self):
        """Destructor"""
        for actor in self.vehicles_npcs_list + self.vehicles_marl_list + self.people_list :
            try:
                if actor.is_alive:
                    actor.destroy()
                    print("Actors destroyed")
            except Exception as e:
                print(f"Error destroying actors {e}")

