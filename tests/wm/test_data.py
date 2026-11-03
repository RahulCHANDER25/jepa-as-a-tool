import numpy as np

from jat.wm.data import Rollouts, generate
from jat.worlds import get_world
from jat.worlds.base import Push


def test_episodes_are_reproducible_and_follow_the_physics():
    a, b = generate(3, n_steps=2, seed=5, workers=1), generate(3, n_steps=2, seed=5, workers=1)
    np.testing.assert_array_equal(a.poses, b.poses)
    push = Push(start=a.pushes[1, 0, :2].tolist(), end=a.pushes[1, 0, 2:].tolist())
    result = get_world("pusht").push(a.poses[1, 0], push)
    np.testing.assert_allclose(result.state, a.poses[1, 1], atol=1e-9)


def test_dataset_seeds_give_different_episodes():
    a, b = generate(2, n_steps=1, seed=0, workers=1), generate(2, n_steps=1, seed=1, workers=1)
    assert not np.allclose(a.poses, b.poses)


def test_rollouts_round_trip_on_disk(tmp_path):
    rollouts = generate(2, n_steps=1, workers=1)
    rollouts.save(tmp_path)
    loaded = Rollouts.load(tmp_path)
    np.testing.assert_array_equal(loaded.pushes, rollouts.pushes)
    assert loaded.n_episodes == 2 and loaded.n_steps == 1 and loaded.world == "pusht"


def test_rollouts_without_a_world_name_load_as_pusht(tmp_path):
    rollouts = generate(1, n_steps=1, workers=1)
    arrays = {k: v for k, v in vars(rollouts).items() if k != "world"}
    np.savez(tmp_path / "rollouts.npz", **arrays)
    assert Rollouts.load(tmp_path).world == "pusht"
