import carla
import time
import random


class CarlaControler():
    """Class to connect with CARLA server, set the weather parameters, maps, cars and other simulator configurations"""
    def __init__(self):

        self.client=None
        self.world=None
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
                    print("Failed spawned pederestian")
            
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

    
    def destroy_actors(self):
        """Destructor"""
        for actor in self.vehicles_npcs_list + self.vehicles_marl_list + self.people_list :
            try:
                if actor.is_alive:
                    actor.destroy()
                    print("Actors destroyed")
            except Exception as e:
                print(f"Error destroying actors {e}")

prueba = CarlaControler()