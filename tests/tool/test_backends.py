import numpy as np
from torch import nn

from jat.tool.backends import WorldModelBackend
from jat.tool.session import Session
from jat.wm.models import Predictor, WorldModel
from jat.worlds import get_world

PUSHT = get_world("pusht")


class ExactTranslator(nn.Module):
    """Reads the fake encoder's latents perfectly: the pose output sits in token 0."""

    def forward(self, z):
        return z[:, 0, :4]


def _backend(fake_encoder) -> WorldModelBackend:
    # Untrained predictor = "nothing moves", so every imagined pose equals its parent's.
    predictor = Predictor(fake_encoder.n_tokens, fake_encoder.dim, width=16, depth=1, heads=2)
    return WorldModelBackend(WorldModel(PUSHT, fake_encoder, predictor.eval(), ExactTranslator()))


def test_world_model_backend_reads_the_real_frame_through_the_translator(fake_encoder):
    task = PUSHT.sample_task("rotate", 0, 0)
    session = Session(task, _backend(fake_encoder), simulate_enabled=True)
    np.testing.assert_allclose(session.frames.get("R0").pose, task.start, atol=1e-3)


def test_world_model_backend_runs_batched_plans(fake_encoder):
    task = PUSHT.sample_task("rotate", 0, 0)
    session = Session(task, _backend(fake_encoder), simulate_enabled=True)
    text = session.simulate("R0", [list(task.ref_plan), [task.ref_plan[0]]])
    assert "I1 = R0" in text and "I2 = R0" in text and "I3 = I1" in text
    assert "the T does not move" in text
    np.testing.assert_allclose(session.frames.get("I3").pose, task.start, atol=1e-3)
