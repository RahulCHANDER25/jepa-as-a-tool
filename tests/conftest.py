import json
import os

# Plain CLI output in tests: on GitHub Actions typer colors it, and the color codes split
# option names like "--wm". Set before typer is first imported.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

import numpy as np
import pytest
import torch

from jat.wm.data import TEST, VAL, generate
from jat.wm.readouts import readout


class FakeEncoder:
    """Stands in for V-JEPA 2 (a 1.3 GB download): every token holds the world's readout of the
    state, so a translator can read it exactly."""

    n_tokens, dim = 4, 8

    def encode_states(self, world, states) -> torch.Tensor:
        out = readout(world.name).encode(torch.tensor(np.asarray(states), dtype=torch.float32))
        z = torch.zeros(len(out), self.n_tokens, self.dim)
        z[:, :, : out.shape[1]] = out[:, None, :]
        return z


def make_fake_dataset(folder, world: str, episodes: int = 24, steps: int = 3):
    """A dataset folder with short random episodes and fake latents; 4 VAL, 4 TEST episodes."""
    from jat.worlds import get_world

    rollouts = generate(episodes, n_steps=steps, workers=1, world=world)
    rollouts.split[:] = 0
    rollouts.split[:4], rollouts.split[4:8] = VAL, TEST
    rollouts.save(folder)
    encoder = FakeEncoder()
    states = rollouts.poses.reshape(-1, rollouts.poses.shape[-1])
    z = encoder.encode_states(get_world(world), states)
    shape = [episodes, steps + 1, encoder.n_tokens, encoder.dim]
    np.save(folder / "latents.npy", z.reshape(shape).numpy().astype(np.float16))
    meta = {"world": world, "encoder": "fake", "pool": 1, "shape": shape}
    (folder / "latents.json").write_text(json.dumps(meta))
    return folder


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def fake_dataset(tmp_path):
    return make_fake_dataset(tmp_path / "pusht", "pusht")


@pytest.fixture
def fake_billiards(tmp_path):
    return make_fake_dataset(tmp_path / "billiards", "billiards")


@pytest.fixture(scope="session")
def can_render():
    """Skip a test when this machine has no OpenGL for MuJoCo (CI has no GPU and no display)."""
    from jat.worlds import get_world

    try:
        get_world("tabletop").render([256.0, 200.0, 0.0], size=32)
    except Exception as error:  # noqa: BLE001 (any GL failure means "cannot render here")
        pytest.skip(f"MuJoCo cannot render here: {type(error).__name__}")
