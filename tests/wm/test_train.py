import json

import torch

from jat.wm.dataset import LatentData
from jat.wm.models import Translator, load_module
from jat.wm.train import train_predictor, train_translator, transfer_test

CPU = torch.device("cpu")


def test_translator_learns_to_read_the_pose(fake_dataset):
    result = train_translator([LatentData(fake_dataset, CPU)], fake_dataset, steps=400, batch=64)
    # The fake latents hold the pose itself; an untrained reader is off by ~200 px.
    assert result["pusht"]["test"]["px_median"] < 25
    assert (fake_dataset / "translator.pt").exists()


def test_one_translator_reads_two_worlds(fake_dataset, fake_billiards, tmp_path):
    datasets = [LatentData(fake_dataset, CPU), LatentData(fake_billiards, CPU)]
    result = train_translator(datasets, tmp_path / "shared", steps=600, batch=64, log_every=300)
    assert set(result) == {"pusht", "billiards"}
    assert result["pusht"]["test"]["px_median"] < 30
    assert result["billiards"]["test"]["px_median"] < 30


def test_heads_on_a_frozen_trunk_leave_the_trunk_unchanged(fake_dataset, fake_billiards, tmp_path):
    train_translator([LatentData(fake_dataset, CPU)], tmp_path / "a", steps=20, log_every=10)
    trunk_from = tmp_path / "a" / "translator.pt"
    train_translator(
        [LatentData(fake_billiards, CPU)],
        tmp_path / "b",
        steps=20,
        log_every=10,
        trunk_from=trunk_from,
    )
    a, _ = load_module(Translator, trunk_from)
    b, _ = load_module(Translator, tmp_path / "b" / "translator.pt")
    torch.testing.assert_close(a.squeeze.weight, b.squeeze.weight)
    assert list(b.heads) == ["billiards"]


def test_transfer_test_writes_one_row_per_budget(fake_dataset, fake_billiards, tmp_path):
    train_translator([LatentData(fake_dataset, CPU)], tmp_path / "a", steps=20, log_every=10)
    rows = transfer_test(
        LatentData(fake_billiards, CPU),
        tmp_path / "a" / "translator.pt",
        tmp_path / "t",
        budgets=[4, 100],
        steps=20,
    )
    assert [r["episodes"] for r in rows] == [4, 16]
    assert {"head_only", "scratch"} <= set(rows[0])


def test_predictor_training_writes_a_checkpoint_and_a_horizon_table(fake_dataset):
    data = LatentData(fake_dataset, CPU)
    train_translator([data], fake_dataset, steps=20, batch=16, log_every=10)
    rows = train_predictor(data, fake_dataset, steps=20, batch=8, rollout=2, log_every=10)
    assert [r["horizon"] for r in rows] == [0, 1, 2, 3]
    saved = json.loads((fake_dataset / "wm_eval.json").read_text())
    assert len(saved["history"]) == 2 and (fake_dataset / "predictor.pt").exists()
    first = saved["history"][0]
    assert first["latent"] != first["readout"]  # each loss is logged on its own


def test_fine_tune_starts_from_a_predictor_and_mixes_an_extra_dataset(fake_dataset, tmp_path):
    from tests.conftest import make_fake_dataset

    data = LatentData(fake_dataset, CPU)
    extra = LatentData(make_fake_dataset(tmp_path / "agent", "pusht", steps=1), CPU)
    train_translator([data], fake_dataset, steps=20, batch=16, log_every=10)
    train_predictor(data, fake_dataset, steps=10, batch=8, log_every=10)
    out = tmp_path / "ft"
    train_predictor(
        data,
        out,
        steps=10,
        batch=8,
        log_every=5,
        init=fake_dataset / "predictor.pt",
        translator_path=fake_dataset / "translator.pt",
        extra=extra,
    )
    saved = json.loads((out / "wm_eval.json").read_text())
    assert "extra_test" in saved and (out / "predictor.pt").exists()


def test_latents_too_big_for_ram_are_read_from_disk(fake_dataset, monkeypatch):
    monkeypatch.setattr("jat.wm.dataset.available_ram", lambda: 0)
    on_disk, in_ram = LatentData(fake_dataset, CPU), None
    monkeypatch.undo()
    in_ram = LatentData(fake_dataset, CPU)
    assert not on_disk.in_ram and in_ram.in_ram
    e, t = torch.tensor([0, 5]), torch.tensor([1, 3])
    torch.testing.assert_close(on_disk.latent(e, t), in_ram.latent(e, t))
