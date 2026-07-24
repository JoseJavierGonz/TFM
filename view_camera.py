#!/usr/bin/env python3
"""
view_camera.py
==============
Standalone viewer for a MARL agent's RGB camera feed.

Connects to a running CARLA server (as a passive second client), attaches a
separate display camera to the chosen MARL vehicle, and either:
  - saves colorised frames to  <output_dir>/agent_<N>/frame_XXXXXXXX.png
  - shows a live window (--display, requires X11 / VNC)

NOTE on re-attach behaviour
---------------------------
On every episode *reset* the training script only TELEPORTS the vehicle
(set_transform), so sensors remain attached — no re-attach is needed.
Re-attach IS triggered when the training script fully reconnects to CARLA
(envCARLA destroyed + re-created), which destroys the vehicles.  The viewer
detects `vehicle.is_alive == False`, destroys its camera, and re-searches for
the new vehicle.

NOTE on synchronous mode
------------------------
The training script drives world.tick().  This viewer must NOT call tick().
Sensor callbacks fire automatically on each training tick.

Usage
-----
    python view_camera.py [--agent 0|1] [--host carla-engine] [--port 2000]
                          [--output-dir camera_feed] [--interval 10]
                          [--display]
"""

import argparse
import os
import signal
import sys
import time
import threading
from queue import Empty, Queue

import numpy as np

try:
    import carla
except ImportError:
    sys.exit("ERROR: carla Python module not found. Activate the correct env.")

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("WARNING: cv2 not available — frames will be saved but not displayed.")


class CameraViewer:
    def __init__(
        self,
        host: str,
        port: int,
        agent_idx: int,
        output_dir: str,
        interval: int,
        display: bool,
        cam_width: int = 256,
        cam_height: int = 256,
    ):
        self.host       = host
        self.port       = port
        self.agent_idx  = agent_idx
        self.output_dir = os.path.join(output_dir, f"agent_{agent_idx}")
        self.interval   = interval
        self.display    = display and _HAS_CV2
        self.cam_width  = cam_width
        self.cam_height = cam_height

        self._stop    = threading.Event()
        self._queue: Queue = Queue(maxsize=2)

        self.client  = None
        self.world   = None
        self.vehicle = None
        self.camera  = None
        self._frame_count = 0

        os.makedirs(self.output_dir, exist_ok=True)

    # ── CARLA connection ───────────────────────────────────────────────────────

    def _connect(self) -> bool:
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world  = self.client.get_world()
            print(f"[viewer] Connected to CARLA at {self.host}:{self.port}")
            return True
        except Exception as exc:
            print(f"[viewer] Connection error: {exc}")
            return False

    # ── Vehicle discovery ──────────────────────────────────────────────────────

    def _find_vehicle(self):
        """Return the MARL vehicle for this agent index by role_name tag, or None."""
        target_role = f"marl_agent_{self.agent_idx}"
        actors = self.world.get_actors().filter("vehicle.*")
        for actor in actors:
            if actor.attributes.get("role_name") == target_role:
                print(f"[viewer] Found {actor.type_id}  role={target_role}  id={actor.id}")
                return actor
        print(f"[viewer] No vehicle with role_name '{target_role}' found.")
        return None

    # ── Sensor management ─────────────────────────────────────────────────────

    def _attach_camera(self) -> bool:
        if self.vehicle is None:
            return False
        try:
            bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(self.cam_width))
            bp.set_attribute("image_size_y", str(self.cam_height))
            bp.set_attribute("fov", "90")

            transform = carla.Transform(
                carla.Location(x=2.0, z=1.5),
                carla.Rotation(pitch=-5.0),
            )
            self.camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
            self.camera.listen(self._camera_callback)
            print(
                f"[viewer] Camera attached to {self.vehicle.type_id} "
                f"(vehicle id={self.vehicle.id})"
            )
            return True
        except Exception as exc:
            print(f"[viewer] Failed to attach camera: {exc}")
            return False

    def _detach_camera(self):
        """Stop and destroy the viewer camera sensor."""
        if self.camera is None:
            return
        try:
            self.camera.stop()
            self.camera.destroy()
            print("[viewer] Camera detached.")
        except Exception as exc:
            print(f"[viewer] Error detaching camera: {exc}")
        finally:
            self.camera = None

    # ── Sensor callback (fires in CARLA's internal thread on each tick) ────────

    def _camera_callback(self, image):
        try:
            raw = np.frombuffer(image.raw_data, dtype=np.uint8)
            raw = raw.reshape((image.height, image.width, 4))
            # [FIX] .copy(): raw_data buffer is freed/reused after the callback
            # returns, so a bare view would read garbage in the consumer thread;
            # copy also yields a C-contiguous array required by cv2.imwrite.
            bgr = raw[:, :, :3].copy()  # BGRA → BGR (drop alpha)

            # Drop oldest frame if queue is full so we never block the training tick
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put_nowait((int(image.frame), bgr))
        except Exception as exc:
            print(f"[viewer] Callback error: {exc}")

    # ── Main display / save loop ───────────────────────────────────────────────

    def _loop(self):
        """Process frames until stopped or vehicle is gone."""
        win = f"Agent {self.agent_idx} — RGB Camera (press q to quit)"
        if self.display:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 512, 512)

        try:
            while not self._stop.is_set():
                # Detect vehicle destruction (training restart)
                try:
                    if not self.vehicle.is_alive:
                        print("[viewer] Vehicle destroyed — will reattach after restart.")
                        return
                except Exception:
                    return

                try:
                    frame_id, bgr = self._queue.get(timeout=5.0)
                except Empty:
                    print(
                        "[viewer] No frame for 5 s "
                        "(waiting for training ticks or vehicle respawn…)"
                    )
                    continue

                # Save every `interval` frames
                if self._frame_count % self.interval == 0:
                    path = os.path.join(
                        self.output_dir, f"frame_{frame_id:08d}.png"
                    )
                    if _HAS_CV2:
                        cv2.imwrite(path, bgr)

                # Live display
                if self.display:
                    cv2.imshow(win, bgr)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        self._stop.set()
                        break

                self._frame_count += 1

        except Exception as exc:
            print(f"[viewer] Loop error: {exc}")
        finally:
            if self.display and _HAS_CV2:
                cv2.destroyAllWindows()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self):
        """
        Main loop: connect → find vehicle → attach camera → process frames.
        Reconnects automatically if the CARLA server restarts or the vehicle
        is destroyed (training script reconnects).
        """
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        print(f"[viewer] Saving frames to: {self.output_dir}")
        print(f"[viewer] Save interval: every {self.interval} frames")

        while not self._stop.is_set():
            # 1. Connect to CARLA
            if not self._connect():
                print("[viewer] Retrying connection in 5 s…")
                time.sleep(5)
                continue

            # 2. Find the MARL vehicle
            self.vehicle = self._find_vehicle()
            if self.vehicle is None:
                print("[viewer] Vehicle not found. Retrying in 5 s…")
                time.sleep(5)
                continue

            # 3. Attach viewer camera
            if not self._attach_camera():
                time.sleep(5)
                continue

            # 4. Process frames (blocks until vehicle dies or stop requested)
            self._loop()

            # 5. Cleanup before reconnect attempt
            self._detach_camera()
            if not self._stop.is_set():
                print("[viewer] Will reconnect in 5 s…")
                time.sleep(5)

        print("[viewer] Stopped.")

    def _handle_signal(self, signum, frame):
        print("\n[viewer] Caught signal — stopping…")
        self._stop.set()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Watch a MARL agent's RGB camera during CARLA training."
    )
    parser.add_argument(
        "--agent", type=int, default=0, choices=[0, 1],
        help="Agent index: 0 = Ford Mustang, 1 = Tesla Model3 (default: 0)",
    )
    parser.add_argument(
        "--host", default="carla-engine",
        help="CARLA server hostname (default: carla-engine)",
    )
    parser.add_argument(
        "--port", type=int, default=2000,
        help="CARLA server port (default: 2000)",
    )
    parser.add_argument(
        "--output-dir", default="camera_feed",
        help="Root directory for saved frames (default: camera_feed/)",
    )
    parser.add_argument(
        "--interval", type=int, default=10,
        help="Save one frame every N callbacks (default: 10)",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show live OpenCV window (requires X11 forwarding or VNC)",
    )
    args = parser.parse_args()

    viewer = CameraViewer(
        host       = args.host,
        port       = args.port,
        agent_idx  = args.agent,
        output_dir = args.output_dir,
        interval   = args.interval,
        display    = args.display,
    )
    viewer.run()


if __name__ == "__main__":
    main()
