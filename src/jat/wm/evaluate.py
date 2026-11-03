"""Offline check of the world model, no LLM: how far is the state the LLM would read from the
true state, after 0, 1, 2, ... imagined pushes?

Horizon 0 is the translator alone on a real frame: can V-JEPA 2 latents show the world's state?
Errors come from the world's readout: px (T center, or the worst ball) and degrees (T only).
"""

import numpy as np
import torch

from jat.wm.dataset import LatentData
from jat.wm.models import Predictor, Translator
from jat.worlds.tblock.world import ANGLE_TOLERANCE, POSITION_TOLERANCE


def summarize_errors(px: np.ndarray, deg: np.ndarray) -> dict:
    return {
        "n": len(px),
        "px_median": float(np.median(px)),
        "px_p90": float(np.percentile(px, 90)),
        "deg_median": float(np.median(deg)),
        "deg_p90": float(np.percentile(deg, 90)),
        # Share of states that would count as "at the goal" if the goal were the true state
        "within_tolerance": float(np.mean((px < POSITION_TOLERANCE) & (deg < ANGLE_TOLERANCE))),
    }


@torch.no_grad()
def translator_errors(
    data: LatentData, translator: Translator, split: int, head: str | None = None
) -> dict:
    """Translator on every real frame of a split. `head`: read with another world's head
    (zero-shot on an evaluation-only world)."""
    head = head or data.world
    episodes = data.episodes(split)
    px, deg = [], []
    for t in range(data.n_steps + 1):
        steps = torch.full_like(episodes, t)
        pred = data.readout.decode(translator(data.latent(episodes, steps), head))
        a, b = data.readout.errors(pred, data.poses[episodes, t])
        px.append(a)
        deg.append(b)
    return summarize_errors(np.concatenate(px), np.concatenate(deg))


@torch.no_grad()
def rollout_errors(
    data: LatentData,
    predictor: Predictor,
    translator: Translator,
    split: int,
    batch: int = 256,
    head: str | None = None,
) -> list[dict]:
    """One row per horizon 0..T: imagined state vs true state, plus the "nothing moves" baseline
    (the start state) for scale."""
    head = head or data.world
    errors_of = data.readout.errors
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
            pred = data.readout.decode(translator(z, head))
            errors["model"][t].append(errors_of(pred, true))
            errors["no_move"][t].append(errors_of(data.poses[chunk, 0], true))

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
