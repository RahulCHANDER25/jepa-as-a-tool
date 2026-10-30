"""The 3D table shared by the MuJoCo worlds: a table with a low rim, lights, a camera looking down
from the front, and a kinematic pusher.

Table coordinates are the 2D worlds' (px, y downward) at 1 px = 1 mm: X = x / 1000, Y = -y / 1000,
the table surface at Z = 0. A screen angle theta (clockwise, y downward) is a yaw of -theta.
"""

import math
import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from jat.worlds.base import PUSHER_RADIUS, TABLE

MM = 0.001
RIM_INSET = 5.0  # mm from the table edge to the rim's inner face, like PushT's walls
RIM_HEIGHT = 0.02
PUSHER_HALF_HEIGHT = 0.015
PUSHER_Z = PUSHER_HALF_HEIGHT + 0.001  # 1 mm above the table: it never scrapes the surface
PARKED = (0.256, -0.256, 5.0)  # far above the table, out of the camera's view
RENDER_SIZE = 256  # the frame V-JEPA 2 sees

# MuJoCo picks its OpenGL backend when imported: EGL on a headless GPU machine (Kaggle), GLFW
# when there is a display. MUJOCO_GL set by the user wins.
if "MUJOCO_GL" not in os.environ:
    has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    os.environ["MUJOCO_GL"] = "glfw" if has_display else "egl"


def to_world(x: float, y: float) -> tuple[float, float]:
    return x * MM, -y * MM


def from_world(X: float, Y: float) -> tuple[float, float]:
    return X / MM, -Y / MM


def yaw_quat(theta: float) -> np.ndarray:
    """Quaternion (w, x, y, z) of a T at screen angle theta."""
    half = -theta / 2
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def screen_angle(quat) -> float:
    w, x, y, z = quat
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return -yaw


@dataclass(frozen=True)
class Look:
    """Everything about the scene that changes the picture but not the physics."""

    table_rgb1: str = "0.62 0.50 0.38"
    table_rgb2: str = "0.58 0.46 0.35"
    rim_rgba: str = "0.30 0.22 0.16 1"
    floor_rgb: str = "0.20 0.22 0.25"
    object_rgba: str = "0.25 0.42 0.70 1"
    second_rgba: str = "0.80 0.15 0.15 1"  # a second object (billiards: the red ball)
    light_dir: str = "0.3 0.5 -1"
    elevation: float = 62.0  # camera, degrees above the table
    camera_shift: float = 0.0  # camera moved sideways, m
    distance: float = 0.77  # camera to the table center, m


DEFAULT_LOOK = Look()
# World 4: another table, another light, another color, the camera a bit moved. Never trained on.
NEW_LOOK = Look(
    table_rgb1="0.78 0.80 0.80",
    table_rgb2="0.70 0.73 0.74",
    rim_rgba="0.15 0.16 0.18 1",
    floor_rgb="0.45 0.35 0.30",
    object_rgba="0.85 0.55 0.15 1",
    second_rgba="0.15 0.55 0.30 1",
    light_dir="-0.5 0.3 -1",
    elevation=57.0,
    camera_shift=0.03,
    distance=0.80,
)


def _camera(look: Look) -> str:
    cx, cy = to_world(TABLE / 2, TABLE / 2)
    e = math.radians(look.elevation)
    pos = (cx + look.camera_shift, cy - look.distance * math.cos(e), look.distance * math.sin(e))
    up = (0.0, math.sin(e), math.cos(e))
    return (
        f'<camera name="main" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
        f'xyaxes="1 0 0 {up[0]:.4f} {up[1]:.4f} {up[2]:.4f}" fovy="45"/>'
    )


def _rim(height: float) -> str:
    half = TABLE / 2 * MM
    thick = 0.01
    inner = half - RIM_INSET * MM  # rim inner faces sit RIM_INSET from each edge
    c = to_world(TABLE / 2, TABLE / 2)
    parts = []
    for name, (dx, dy, sx, sy) in {
        "rim_left": (-(inner + thick), 0, thick, half + thick),
        "rim_right": (inner + thick, 0, thick, half + thick),
        "rim_far": (0, inner + thick, half + thick, thick),
        "rim_near": (0, -(inner + thick), half + thick, thick),
    }.items():
        parts.append(
            f'<geom name="{name}" class="rim" type="box" '
            f'pos="{c[0] + dx:.4f} {c[1] + dy:.4f} {height / 2}" '
            f'size="{sx:.4f} {sy:.4f} {height / 2}"/>'
        )
    return "\n    ".join(parts)


def scene_xml(
    objects: str, look: Look, timestep: float, rim_class: str = "", rim_height: float = RIM_HEIGHT
) -> str:
    """A full MJCF model: the table, `objects` (bodies placed on it) and the parked pusher.
    `rim_class` holds extra attributes for the rim geoms (billiards: bouncy cushions); the rim
    must stand taller than any ball's radius, or balls jump over it."""
    c = to_world(TABLE / 2, TABLE / 2)
    half = TABLE / 2 * MM
    return f"""
<mujoco model="table3d">
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"/>
  <visual>
    <global offwidth="512" offheight="512"/>
    <quality shadowsize="2048" offsamples="4"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="table" type="2d" builtin="checker" rgb1="{look.table_rgb1}"
             rgb2="{look.table_rgb2}" width="512" height="512"/>
    <material name="table" texture="table" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <default>
    <default class="rim">
      <geom rgba="{look.rim_rgba}" {rim_class}/>
    </default>
  </default>
  <worldbody>
    <light directional="true" castshadow="true" pos="0 0 2" dir="{look.light_dir}"
           diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    {_camera(look)}
    <geom name="floor" type="plane" pos="{c[0]} {c[1]} -0.05" size="3 3 0.1"
          rgba="{look.floor_rgb} 1" contype="0" conaffinity="0"/>
    <geom name="table" type="box" pos="{c[0]} {c[1]} -0.01" size="{half} {half} 0.01"
          material="table"/>
    {_rim(rim_height)}
    {objects}
    <body name="pusher" mocap="true" pos="{PARKED[0]} {PARKED[1]} {PARKED[2]}">
      <geom name="pusher" type="cylinder" size="{PUSHER_RADIUS * MM} {PUSHER_HALF_HEIGHT}"
            rgba="0.9 0.9 0.9 1" friction="0.3 0.005 0.0001"
            solref="0.004 1" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


class Scene:
    """One compiled model with per-thread simulation data and one shared renderer.

    Physics is plain CPU MuJoCo and runs anywhere. Rendering needs an OpenGL context: set
    MUJOCO_GL=egl on a headless GPU machine (Kaggle), glfw on a desktop.
    """

    def __init__(self, xml: str):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(xml)
        self._local = threading.local()
        self._render_lock = threading.Lock()
        self._renderer = None
        self._render_data = None

    @property
    def data(self):
        """This thread's MjData."""
        if not hasattr(self._local, "data"):
            self._local.data = self.mujoco.MjData(self.model)
        return self._local.data

    def render(self, place, size: int) -> np.ndarray:
        """uint8 [size, size, 3]. `place(data)` sets the objects' qpos; the pusher stays parked.
        Always drawn at RENDER_SIZE, then resized: a second GL context breaks the first."""
        with self._render_lock:  # one GL context, made current by whichever thread renders
            if self._renderer is None:
                self._renderer = self.mujoco.Renderer(self.model, RENDER_SIZE, RENDER_SIZE)
                self._render_data = self.mujoco.MjData(self.model)
            data = self._render_data
            self.mujoco.mj_resetData(self.model, data)
            place(data)
            data.mocap_pos[0] = PARKED
            self.mujoco.mj_forward(self.model, data)
            self._renderer.update_scene(data, camera="main")
            frame = self._renderer.render().copy()
        if size != RENDER_SIZE:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        return frame
