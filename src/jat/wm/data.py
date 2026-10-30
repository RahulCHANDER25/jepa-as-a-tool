"""Training data for the world model: random push episodes in a world's real physics.

One dataset = one folder:
  rollouts.npz   world name, states, pushes and flags of every episode (written by `generate`)
  latents.npy    V-JEPA 2 tokens of every pose, fp16 [E, T+1, P, D] (written by `jat encode`)
  latents.json   how the latents were made: encoder id, pooling, shape
Episodes all have the same length, so plain dense arrays are enough.
"""

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from jat.worlds import get_world
from jat.worlds.base import uniform_push

TRAIN, VAL, TEST = 0, 1, 2
# Mixed into every episode seed. Tasks seed their rng with [seed, index]; without this,
# episode i of dataset seed 0 would start exactly where task i of task seed 0 starts.
_STREAM = 7_000_001


@dataclass
class Rollouts:
    poses: np.ndarray  # [E, T+1, S] float: the world state (T worlds: x, y, theta)
    pushes: np.ndarray  # [E, T, 4] float: start x, y, end x, y in table px
    moved: np.ndarray  # [E, T] bool: the push moved the T
    jammed: np.ndarray  # [E, T] bool: the push stopped early, T pinned against a wall
    split: np.ndarray  # [E] uint8: TRAIN, VAL or TEST
    world: str = "pusht"

    @property
    def n_episodes(self) -> int:
        return len(self.poses)

    @property
    def n_steps(self) -> int:
        return self.pushes.shape[1]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "rollouts.npz", **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, folder: Path) -> "Rollouts":
        with np.load(folder / "rollouts.npz") as arrays:
            values = {f.name: arrays[f.name] for f in fields(cls) if f.name in arrays}
        # Datasets made before there were several worlds have no name: they are PushT.
        values["world"] = str(values.get("world", "pusht"))
        return cls(**values)


def random_split(n: int, seed, val_share: float, test_share: float) -> np.ndarray:
    """TRAIN / VAL / TEST per episode; at least one VAL and one TEST episode when n >= 3, so
    tiny datasets (dry runs) can still be evaluated."""
    split = np.random.default_rng(seed).choice(
        [TRAIN, VAL, TEST], size=n, p=[1 - val_share - test_share, val_share, test_share]
    )
    if n >= 3:
        for k, which in enumerate((VAL, TEST)):
            if not (split == which).any():
                split[n - 1 - k] = which
    return split.astype(np.uint8)


def _episode(args: tuple[str, int, int, int, float]) -> tuple[np.ndarray, ...]:
    world_name, seed, index, n_steps, aimed_share = args
    world = get_world(world_name)
    rng = np.random.default_rng([_STREAM, seed, index])
    state = world.random_start(rng)
    states, pushes, moved, jammed = [state], [], [], []
    for _ in range(n_steps):
        push = None
        if rng.random() < aimed_share:
            try:
                push = world.aimed_push(state, rng)
            except RuntimeError:  # object squeezed in a corner: no aimed push fits
                pass
        if push is None:
            push = uniform_push(world, state, rng)
        result = world.push(state, push)
        state = np.array(result.state)
        states.append(state)
        pushes.append([*push.start, *push.end])
        moved.append(result.moved)
        jammed.append(result.jammed)
    return np.array(states), np.array(pushes), np.array(moved), np.array(jammed)


def generate(
    n_episodes: int,
    n_steps: int = 6,
    seed: int = 0,
    aimed_share: float = 0.8,
    val_share: float = 0.05,
    test_share: float = 0.05,
    workers: int = 4,
    world: str = "pusht",
) -> Rollouts:
    """Episodes of `n_steps` random pushes. Aimed pushes hit the T; uniform ones mostly miss,
    which teaches the model that a missed push changes nothing."""
    jobs = [(world, seed, i, n_steps, aimed_share) for i in range(n_episodes)]
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as pool:
        results = pool.map(_episode, jobs, chunksize=16) if pool else map(_episode, jobs)
        episodes = []
        for i, episode in enumerate(results, 1):
            episodes.append(episode)
            if i % max(1, n_episodes // 10) == 0:
                print(f"generated {i}/{n_episodes} episodes", flush=True)
    poses, pushes, moved, jammed = (np.stack(a) for a in zip(*episodes, strict=True))
    split = random_split(n_episodes, [_STREAM, seed], val_share, test_share)
    return Rollouts(poses, pushes, moved, jammed, split, world)
