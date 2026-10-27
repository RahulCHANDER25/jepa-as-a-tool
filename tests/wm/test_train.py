import json

import torch

from jat.wm.dataset import LatentData
from jat.wm.train import train_predictor, train_translator

CPU = torch.device("cpu")


def test_translator_learns_to_read_the_pose(fake_dataset):
    result = train_translator(LatentData(fake_dataset, CPU), fake_dataset, steps=400, batch=64)
    # The fake latents hold the pose itself; an untrained reader is off by ~200 px.
    assert result["test"]["px_median"] < 25
    assert (fake_dataset / "translator.pt").exists()


def test_predictor_training_writes_a_checkpoint_and_a_horizon_table(fake_dataset):
    data = LatentData(fake_dataset, CPU)
    train_translator(data, fake_dataset, steps=20, batch=16, log_every=10)
    rows = train_predictor(data, fake_dataset, steps=20, batch=8, rollout=2, log_every=10)
    assert [r["horizon"] for r in rows] == [0, 1, 2, 3]
    saved = json.loads((fake_dataset / "wm_eval.json").read_text())
    assert len(saved["history"]) == 2 and (fake_dataset / "predictor.pt").exists()
    first = saved["history"][0]
    assert first["latent"] != first["keypoint"]  # each loss is logged on its own


def test_latents_too_big_for_ram_are_read_from_disk(fake_dataset, monkeypatch):
    monkeypatch.setattr("jat.wm.dataset.available_ram", lambda: 0)
    on_disk, in_ram = LatentData(fake_dataset, CPU), None
    monkeypatch.undo()
    in_ram = LatentData(fake_dataset, CPU)
    assert not on_disk.in_ram and in_ram.in_ram
    e, t = torch.tensor([0, 5]), torch.tensor([1, 3])
    torch.testing.assert_close(on_disk.latent(e, t), in_ram.latent(e, t))
