import math

import torch

from jat.wm.encoder import pool_tokens
from jat.wm.models import (
    Predictor,
    Translator,
    keypoint_loss,
    load_module,
    output_to_pose,
    pose_to_output,
    save_module,
)

POSES = torch.tensor([[256.0, 200.0, 0.0], [100.0, 400.0, 3.0], [300.0, 50.0, -1.2]])


def test_pose_survives_the_translator_encoding():
    back = output_to_pose(pose_to_output(POSES))
    torch.testing.assert_close(back, POSES)


def test_keypoint_loss_is_zero_only_at_the_true_pose():
    exact = pose_to_output(POSES)
    assert keypoint_loss(exact, POSES).item() < 1e-6
    turned = POSES + torch.tensor([0.0, 0.0, math.radians(10)])
    assert keypoint_loss(pose_to_output(turned), POSES).item() > 1e-4


def test_untrained_predictor_says_nothing_moves():
    z = torch.randn(2, 4, 8)
    torch.testing.assert_close(Predictor(4, 8, width=16, depth=1, heads=2)(z, torch.zeros(2, 4)), z)


def test_saved_module_gives_the_same_outputs(tmp_path):
    model = Translator(4, 8).eval()
    save_module(model, tmp_path / "t.pt", note="x")
    loaded, extra = load_module(Translator, tmp_path / "t.pt")
    z = torch.randn(3, 4, 8)
    torch.testing.assert_close(loaded(z), model(z))
    assert extra == {"note": "x"}


def test_pooling_keeps_a_square_grid_of_normalized_tokens():
    tokens = pool_tokens(torch.randn(2, 256, 32) * 5, pool=2)
    assert tokens.shape == (2, 64, 32)
    torch.testing.assert_close(tokens.mean(-1), torch.zeros(2, 64), atol=1e-5, rtol=0)
