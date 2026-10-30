"""Offline check of the world model, no LLM: how far is the pose the LLM would read from the
true pose, after 0, 1, 2, ... imagined pushes?

Horizon 0 is the translator alone on a real frame: can V-JEPA 2 latents show where the T is?
"""

import math

import numpy as np
import torch

from jat.wm.dataset import LatentData
from jat.wm.models import Predictor, Translator, output_to_pose
from jat.worlds.tblock import shape as geo
from jat.worlds.tblock.world import ANGLE_TOLERANCE, POSITION_TOLERANCE


def pose_errors(pred: torch.Tensor, true: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized geo.pose_error: (center distance px [N], angle difference deg [N])."""

    def centers(poses):
        x, y, theta = poses.unbind(-1)
        cx, cy = geo.CENTER
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.stack([x + c * cx - s * cy, y + s * cx + c * cy], -1)

    distance = (centers(pred) - centers(true)).norm(dim=-1)
    turn = torch.remainder(pred[:, 2] - true[:, 2] + math.pi, 2 * math.pi) - math.pi
    return distance.cpu().numpy(), turn.abs().rad2deg().cpu().numpy()


def summarize_errors(px: np.ndarray, deg: np.ndarray) -> dict:
    return {
        "n": len(px),
        "px_median": float(np.median(px)),
        "px_p90": float(np.percentile(px, 90)),
        "deg_median": float(np.median(deg)),
        "deg_p90": float(np.percentile(deg, 90)),
        # Share of poses that would count as "at the goal" if the goal were the true pose
        "within_tolerance": float(np.mean((px < POSITION_TOLERANCE) & (deg < ANGLE_TOLERANCE))),
    }


@torch.no_grad()
def translator_errors(data: LatentData, translator: Translator, split: int) -> dict:
    """Translator on every real frame of a split."""
    episodes = data.episodes(split)
    px, deg = [], []
    for t in range(data.n_steps + 1):
        steps = torch.full_like(episodes, t)
        pred = output_to_pose(translator(data.latent(episodes, steps)))
        a, b = pose_errors(pred, data.poses[episodes, t])
        px.append(a)
        deg.append(b)
    return summarize_errors(np.concatenate(px), np.concatenate(deg))


@torch.no_grad()
def rollout_errors(
    data: LatentData, predictor: Predictor, translator: Translator, split: int, batch: int = 256
) -> list[dict]:
    """One row per horizon 0..T: imagined pose vs true pose, plus the "nothing moves" baseline
    (the start pose) for scale."""
    episodes = data.episodes(split)
    n_rows = data.n_steps + 1
    # errors[guess][t] = list of (px, deg) arrays, one per chunk
    errors = {"model": [[] for _ in range(n_rows)], "no_move": [[] for _ in range(n_rows)]}
    for chunk in episodes.split(batch):
        z = data.latent(chunk, torch.zeros_like(chunk))
        for t in range(n_rows):
            if t > 0:
                z = predictor(z, data.pushes[chunk, t - 1])
            true = data.poses[chunk, t]
            errors["model"][t].append(pose_errors(output_to_pose(translator(z)), true))
            errors["no_move"][t].append(pose_errors(data.poses[chunk, 0], true))

    def summary(guess: str, t: int) -> dict:
        px, deg = zip(*errors[guess][t], strict=True)
        return summarize_errors(np.concatenate(px), np.concatenate(deg))

    rows = []
    for t in range(n_rows):
        baseline = summary("no_move", t)
        rows.append(
            {
                "horizon": t,
                **summary("model", t),
                "no_move_px_median": baseline["px_median"],
                "no_move_deg_median": baseline["deg_median"],
            }
        )
    return rows


def format_rows(rows: list[dict]) -> str:
    lines = [
        "pushes  median px  p90 px  median °  p90 °  within 20px/15°   | no-move px  no-move °",
    ]
    for r in rows:
        lines.append(
            f"{r['horizon']:>6}  {r['px_median']:>9.1f}  {r['px_p90']:>6.1f}  {r['deg_median']:>8.1f}"
            f"  {r['deg_p90']:>5.1f}  {r['within_tolerance']:>15.0%}   | {r['no_move_px_median']:>10.1f}"
            f"  {r['no_move_deg_median']:>9.1f}"
        )
    return "\n".join(lines)
