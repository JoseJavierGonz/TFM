import carla
import gym
from gym import spaces
import time
import numpy as np
from carlaControler import CarlaControler


class envCARLA(gym.Env):
    """Class to create a gym env, where implement the steps, rewards and so on"""
    def __init__(self):
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
        self.CARLA = CarlaControler()

        for vehicle in self.CARLA.vehicles_marl_list:
            self.__agent.append(vehicle)




# RECOMPENSA POR VELOCIDAD E IR DENTRO DEL CARRIL 
    def step(self, action):
        cars=[]

        for i in range(len(action)):
            cars.append(action[i])
        i=0
        for vehicle in self.__agent:
            car = cars[i]
            throttle = float(car[0])
            if throttle > 1.0:
                throttle = 1.0

            steer = float(car[1])
            if abs(steer) > 1.0:
                steer = 1.0 if steer > 0 else -1.0

            brake = float(car[2])
            if brake > 1.0:
                brake = 1.0

            move = carla.VehicleControl(throttle, steer, brake)
            vehicle.apply_control(move)
            i += 1
        
        observations = self.__get_obs()




    def __get_obs(self):
        observation = []
        for agent in self.__agent:
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
            
            observation.append(obs)



        return observation

    
        