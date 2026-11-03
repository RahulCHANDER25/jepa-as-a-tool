import numpy as np

from jat.tool.backends import OracleBackend
from jat.tool.session import Budget, Session
from jat.worlds import get_world
from jat.worlds.base import Push

PUSHT = get_world("pusht")
TASK = PUSHT.sample_task("rotate", seed=0, index=0)
A, B = TASK.ref_plan


def _session(**budget) -> Session:
    return Session(TASK, OracleBackend(PUSHT), simulate_enabled=True, budget=Budget(**budget))


def test_simulate_from_imagined_frame_equals_longer_plan():
    session = _session()
    session.simulate("R0", [[A]])  # creates I1 = R0 + A
    session.simulate("I1", [[B]])  # creates I2 = I1 + B
    session.simulate("R0", [[A, B]])  # creates I3, I4
    frames = session.frames
    np.testing.assert_allclose(frames.get("I2").pose, frames.get("I4").pose, atol=1e-9)
    np.testing.assert_allclose(frames.get("I4").pose, TASK.goal, atol=1e-9)


def test_unknown_frame_id_returns_an_error_text():
    text = _session().simulate("I9", [[A]])
    assert "Unknown frame 'I9'" in text and "R0" in text


def test_simulate_over_budget_runs_nothing():
    session = _session(imagined_per_push=3)
    text = session.simulate("R0", [[A, B], [A, B]])
    assert "needs 4 imagined pushes; 3 are left" in text
    assert list(session.frames.frames) == ["R0"]


def test_invalid_push_stops_only_its_plan():
    session = _session()
    inside_t = Push(start=TASK.ref_plan[0].end, end=TASK.ref_plan[0].start)  # likely ends inside
    outside = Push(start=[500.0, 500.0], end=[600.0, 500.0])
    text = session.simulate("R0", [[outside], [A]])
    assert "rejected" in text and "I1 = R0 + push" in text
    del inside_t


def test_act_moves_the_real_table_and_ends_at_the_push_budget():
    session = _session()
    first = session.act(A)
    assert "Current real frame: R1" in first and "Real pushes left: 2." in first
    session.act(B)
    assert session.frames.current == "R2" and not session.finished
    session.act(Push(start=[30.0, 30.0], end=[60.0, 30.0]))  # misses, still uses a push
    assert session.stop_reason == "pushes_exhausted"
    assert session.act(A) == "The episode is over."


def test_rejected_real_push_uses_no_push():
    session = _session()
    text = session.act(Push(start=[5.0, 5.0], end=[50.0, 5.0]))
    assert "no push was used" in text and session.pushes_left == TASK.max_pushes


def test_real_push_resets_the_imagination_budget():
    session = _session(imagined_per_push=2)
    session.simulate("R0", [[A, B]])
    assert session.imagined_left == 0
    session.act(A)
    assert session.imagined_left == 2
