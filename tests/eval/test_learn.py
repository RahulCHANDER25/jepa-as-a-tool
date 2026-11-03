import numpy as np

from jat.eval.learn import transitions
from jat.eval.run import run_grid
from jat.worlds import get_world

PUSHT = get_world("pusht")


def test_logged_pushes_become_one_push_episodes_of_the_true_physics(tmp_path):
    tasks = [PUSHT.sample_task("rotate", 0, i) for i in range(2)]
    logs, _ = run_grid(tasks, ["random"], [0], None, tmp_path, workers=1)
    rollouts = transitions([tmp_path])
    n_pushes = sum(len(log.frames) - 1 for log in logs)
    assert rollouts.n_episodes == n_pushes and rollouts.n_steps == 1
    assert rollouts.world == "pusht"
    state, push = rollouts.poses[0, 0], rollouts.pushes[0, 0]
    result = PUSHT.push(state, type(tasks[0].ref_plan[0])(start=push[:2], end=push[2:]))
    np.testing.assert_allclose(result.state, rollouts.poses[0, 1], atol=1e-6)
