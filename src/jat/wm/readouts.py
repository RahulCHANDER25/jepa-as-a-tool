"""What a translator head reads out of a latent, per kind of world state.

A world declares its readout (`World.readout`); the translator gets one head of `size` outputs
per world, trained with the readout's loss and judged with its errors.

- t_pose: the T worlds. Output (x, y scaled to [-1, 1], cos theta, sin theta); the loss compares
  the T's aim points, so it has one unit and no angle wrap-around.
- points: n objects without an angle (billiards balls). Output: their (x, y) scaled to [-1, 1].
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from jat.worlds import get_world
from jat.worlds.base import TABLE
from jat.worlds.tblock import shape as geo

HALF = TABLE / 2
_LANDMARKS = torch.tensor(list(geo.LANDMARKS.values()), dtype=torch.float32)  # [4, 2] local px


def pose_to_output(poses: torch.Tensor) -> torch.Tensor:
    """[B, 3] (x px, y px, theta) -> [B, 4]."""
    x, y, theta = poses.unbind(-1)
    return torch.stack([x / HALF - 1, y / HALF - 1, torch.cos(theta), torch.sin(theta)], -1)


def output_to_pose(out: torch.Tensor) -> torch.Tensor:
    """[B, 4] -> [B, 3] (x px, y px, theta)."""
    return torch.stack(
        [(out[:, 0] + 1) * HALF, (out[:, 1] + 1) * HALF, out[:, 3].atan2(out[:, 2])], -1
    )


def output_keypoints(out: torch.Tensor) -> torch.Tensor:
    """[B, 4] -> the 4 aim landmarks [B, 4, 2], in table px / HALF. Differentiable."""
    cs = F.normalize(out[:, 2:], dim=-1)
    c, s = cs[:, :1], cs[:, 1:]
    local = _LANDMARKS.to(out) / HALF  # [4, 2]
    lx, ly = local[:, 0], local[:, 1]
    world_x = out[:, :1] + c * lx - s * ly
    world_y = out[:, 1:2] + s * lx + c * ly
    return torch.stack([world_x, world_y], -1)


def keypoint_loss(out: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Distance between predicted and true landmarks."""
    target = output_keypoints(pose_to_output(poses))
    return (
        F.smooth_l1_loss(output_keypoints(out), target, beta=0.02)
        + 0.01 * (out[:, 2:].norm(dim=-1) - 1).pow(2).mean()
    )


class TPose:
    size = 4

    encode = staticmethod(pose_to_output)
    decode = staticmethod(output_to_pose)
    loss = staticmethod(keypoint_loss)

    @staticmethod
    def errors(pred: torch.Tensor, true: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """(center distance px [N], angle difference deg [N])."""

        def centers(poses):
            x, y, theta = poses.unbind(-1)
            cx, cy = geo.CENTER
            c, s = torch.cos(theta), torch.sin(theta)
            return torch.stack([x + c * cx - s * cy, y + s * cx + c * cy], -1)

        distance = (centers(pred) - centers(true)).norm(dim=-1)
        turn = torch.remainder(pred[:, 2] - true[:, 2] + math.pi, 2 * math.pi) - math.pi
        return distance.cpu().numpy(), turn.abs().rad2deg().cpu().numpy()


class Points:
    def __init__(self, n: int):
        self.size = 2 * n

    @staticmethod
    def encode(states: torch.Tensor) -> torch.Tensor:
        return states / HALF - 1

    @staticmethod
    def decode(out: torch.Tensor) -> torch.Tensor:
        return (out + 1) * HALF

    def loss(self, out: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(out, self.encode(states), beta=0.02)

    @staticmethod
    def errors(pred: torch.Tensor, true: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """(the worst object's distance px [N], zeros: points have no angle)."""
        distance = (pred - true).reshape(len(pred), -1, 2).norm(dim=-1).amax(-1)
        return distance.cpu().numpy(), np.zeros(len(pred))


def readout(world: str) -> TPose | Points:
    kind = get_world(world).readout
    if kind == "t_pose":
        return TPose()
    if kind.startswith("points:"):
        return Points(int(kind.split(":")[1]))
    raise ValueError(f"unknown readout {kind!r} of world {world!r}")
