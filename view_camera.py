#!/usr/bin/env python3
"""
view_camera_pygame.py
======================
Visor fluido en Pygame para agentes MARL en CARLA.
"""

import argparse
import sys
import time
import weakref

import carla
import numpy as np

try:
    import pygame
except ImportError:
    sys.exit("ERROR: pygame no está instalado. Instálalo con 'pip install pygame'")


def parse_args():
    parser = argparse.ArgumentParser(description="Visor fluido en Pygame para agente MARL")
    parser.add_argument("--agent", type=int, default=0, choices=[0, 1], help="Índice de agente (0 o 1)")
    parser.add_argument("--host", default="carla-engine", help="Host del servidor CARLA")
    parser.add_argument("--port", default=2000, type=int, help="Puerto del servidor CARLA")
    parser.add_argument("--width", default=640, type=int, help="Ancho de la ventana")
    parser.add_argument("--height", default=640, type=int, help="Alto de la ventana")
    parser.add_argument("--fov", default=90, type=float, help="FOV de la cámara")
    return parser.parse_args()


class AgentCameraViewer:
    def __init__(self, world, vehicle, width, height, fov):
        self.world = world
        self.vehicle = vehicle
        self.width = width
        self.height = height

        self.sensor_width = width // 2
        self.sensor_height = height // 2

        self.surface = None
        self.sensor = None
        self._frame_counter = 0

        bp_lib = world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.sensor_width))
        camera_bp.set_attribute("image_size_y", str(self.sensor_height))
        camera_bp.set_attribute("fov", str(fov))

        transform = carla.Transform(
            carla.Location(x=-6.0, z=3.0),
            carla.Rotation(pitch=-15.0)
        )

        self.sensor = world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: AgentCameraViewer._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if self is None:
            return

        self._frame_counter += 1
        if self._frame_counter % 2 != 0:
            return

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]      
        array = array[:, :, ::-1]    

        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


def find_marl_vehicle(world, agent_idx):
    target_role = f"marl_agent_{agent_idx}"
    actors = world.get_actors().filter("vehicle.*")
    for actor in actors:
        if actor.attributes.get("role_name") == target_role:
            return actor
    return None


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
    except Exception as e:
        sys.exit(f"No se pudo conectar con CARLA: {e}")

    pygame.init()
    pygame.font.init()

    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption(f"MARL Viewer - Agente {args.agent}")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    viewer = None
    running = True

    print(f"[viewer] Buscando agente 'marl_agent_{args.agent}'...")

    try:
        while running:
            clock.tick(30)

            world.wait_for_tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False

            if viewer is None or not viewer.vehicle.is_alive:
                if viewer is not None:
                    viewer.destroy()
                    viewer = None

                vehicle = find_marl_vehicle(world, args.agent)
                if vehicle is not None:
                    print(f"[viewer] Agente {args.agent} encontrado. Conectando cámara...")
                    viewer = AgentCameraViewer(
                        world=world,
                        vehicle=vehicle,
                        width=args.width,
                        height=args.height,
                        fov=args.fov
                    )

            if viewer is not None and viewer.surface is not None:
                scaled = pygame.transform.scale(viewer.surface, (args.width, args.height))
                display.blit(scaled, (0, 0))
                
                hud_text = font.render(f"Agente {args.agent} | {clock.get_fps():.1f} FPS", True, (255, 255, 0))
                display.blit(hud_text, (10, 10))
            else:
                display.fill((0, 0, 0))
                hud_text = font.render(f"Buscando 'marl_agent_{args.agent}'...", True, (255, 255, 255))
                display.blit(hud_text, (10, 10))

            pygame.display.flip()

    finally:
        print("[viewer] Cerrando visor...")
        if viewer is not None:
            viewer.destroy()
        pygame.quit()


if __name__ == "__main__":
    main()