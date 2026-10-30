import json
import os

# Plain CLI output in tests: on GitHub Actions typer colors it, and the color codes split
# option names like "--wm". Set before typer is first imported.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

import numpy as np
import pytest
import torch

from jat.wm.data import TEST, VAL, generate
from jat.wm.models import pose_to_output


class FakeEncoder:
    """Stands in for V-JEPA 2 (a 1.3 GB download): every token holds the pose output."""

    n_tokens, dim = 4, 8

    def encode_poses(self, poses) -> torch.Tensor:
        out = pose_to_output(torch.tensor(np.asarray(poses), dtype=torch.float32))
        z = torch.zeros(len(out), self.n_tokens, self.dim)
        z[:, :, :4] = out[:, None, :]
        return z


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def fake_dataset(tmp_path, fake_encoder):
    """A dataset folder with 24 short episodes and fake latents."""
    rollouts = generate(24, n_steps=3, workers=1)
    rollouts.split[:] = 0
    rollouts.split[:4], rollouts.split[4:8] = VAL, TEST
    rollouts.save(tmp_path)
    z = fake_encoder.encode_poses(rollouts.poses.reshape(-1, 3))
    np.save(tmp_path / "latents.npy", z.reshape(24, 4, 4, 8).numpy().astype(np.float16))
    meta = {"encoder": "fake", "pool": 1, "shape": [24, 4, 4, 8]}
    (tmp_path / "latents.json").write_text(json.dumps(meta))
    return tmp_path
