"""Success rates with confidence intervals, grouped by task family and setting."""

import math
from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from jat.agent.episode_log import EpisodeLog


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval: stays inside [0, 1] and is sane for small n."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z**2 / n
    middle = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, middle - half), min(1.0, middle + half)


class Row(BaseModel):
    family: str
    setting: str
    n: int
    success_rate: float
    ci_low: float
    ci_high: float
    position_error: float  # median px
    angle_error: float  # median degrees
    real_pushes: float  # mean
    imagined_pushes: float  # mean
    tokens: float  # mean input + output


def summarize(logs: list[EpisodeLog]) -> list[Row]:
    groups: dict[tuple[str, str], list[EpisodeLog]] = defaultdict(list)
    for log in logs:
        groups[(log.task.family, log.setting)].append(log)
    rows = []
    for (family, setting), group in sorted(groups.items()):
        wins = sum(log.outcome.success for log in group)
        low, high = wilson_interval(wins, len(group))
        rows.append(
            Row(
                family=family,
                setting=setting,
                n=len(group),
                success_rate=wins / len(group),
                ci_low=low,
                ci_high=high,
                position_error=float(np.median([log.outcome.position_error for log in group])),
                angle_error=float(np.median([log.outcome.angle_error for log in group])),
                real_pushes=float(np.mean([_count(log, "real") - 1 for log in group])),
                imagined_pushes=float(np.mean([_count(log, "imagined") for log in group])),
                tokens=float(np.mean([_tokens(log) for log in group])),
            )
        )
    return rows


def _count(log: EpisodeLog, kind: str) -> int:
    return sum(frame.kind == kind for frame in log.frames)


def _tokens(log: EpisodeLog) -> int:
    return sum((t.input_tokens or 0) + (t.output_tokens or 0) for t in log.turns)


def format_table(rows: list[Row]) -> str:
    header = f"{'family':<10} {'setting':<12} {'n':>4} {'success':>8} {'95% CI':>13} {'pos px':>7} {'ang °':>6} {'pushes':>6} {'imag':>6} {'tokens':>7}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.family:<10} {r.setting:<12} {r.n:>4} {r.success_rate:>8.0%} "
            f"{f'[{r.ci_low:.0%}, {r.ci_high:.0%}]':>13} {r.position_error:>7.0f} {r.angle_error:>6.0f} "
            f"{r.real_pushes:>6.1f} {r.imagined_pushes:>6.1f} {r.tokens:>7.0f}"
        )
    return "\n".join(lines)
