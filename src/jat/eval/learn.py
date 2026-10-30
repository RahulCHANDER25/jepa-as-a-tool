"""Learning from use: every push the agent imagined during an evaluation, with the true result
the logs already hold (the real physics replayed along the same branch), becomes a one-push
training episode. These are the pushes the agent actually asks about: aimed close to the object,
often near misses, where the world model is weakest.

Train on the logs of one task set, then evaluate on another task seed, or the test leaks.
"""

from pathlib import Path

import numpy as np

from jat.eval.run import load_logs
from jat.wm.data import Rollouts, random_split
from jat.worlds import get_world


def transitions(
    runs: list[Path], val_share: float = 0.05, test_share: float = 0.05, limit: int | None = None
) -> Rollouts:
    """(true state, push, true next state) for every imagined and real push in the runs' logs,
    each pair once. `limit`: keep a random subset of that many (dry runs)."""
    seen, world_names = set(), set()
    states, pushes, moved = [], [], []
    for run in runs:
        for log in load_logs(run):
            world = get_world(log.task.world)
            world_names.add(world.name)
            true = {f.id: f.true_pose for f in log.frames}
            for frame in log.frames:
                if frame.push is None:
                    continue
                before = true[frame.parent]
                key = (tuple(np.round(before, 6)), *frame.push.start, *frame.push.end)
                if key in seen:
                    continue
                seen.add(key)
                states.append([before, frame.true_pose])
                pushes.append([[*frame.push.start, *frame.push.end]])
                moved.append([world.moved(before, frame.true_pose)])
    if not states:
        raise ValueError(f"no logged pushes in {', '.join(map(str, runs))}")
    if len(world_names) != 1:
        raise ValueError(f"logs from one world only, got {sorted(world_names)}")
    if limit is not None and limit < len(states):
        keep = np.sort(np.random.default_rng(0).choice(len(states), limit, replace=False))
        states, pushes, moved = ([a[i] for i in keep] for a in (states, pushes, moved))
    n = len(states)
    split = random_split(n, 0, val_share, test_share)
    return Rollouts(
        poses=np.array(states, dtype=float),
        pushes=np.array(pushes, dtype=float),
        moved=np.array(moved, dtype=bool),
        jammed=np.zeros((n, 1), dtype=bool),
        split=split,
        world=world_names.pop(),
    )
