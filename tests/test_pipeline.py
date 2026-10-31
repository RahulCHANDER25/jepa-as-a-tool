from jat.pipeline import job_env


def test_a_job_on_the_second_gpu_sees_it_as_its_only_device(monkeypatch):
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    env = job_env(1)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "MUJOCO_EGL_DEVICE_ID" not in env  # EGL numbers the visible GPUs from 0


def test_the_old_billiards_is_deleted_once_only(tmp_path):
    from jat.pipeline import Pipeline

    p = Pipeline(tmp_path)
    keep = p.runs / "tabletop" / "predictor.pt"
    for path in (p.runs / "billiards" / "predictor.pt", keep):
        path.parent.mkdir(parents=True)
        path.touch()
    p.reset_billiards()
    assert not (p.runs / "billiards").exists() and keep.exists()
    (p.runs / "billiards").mkdir()  # the rebuilt billiards
    p.reset_billiards()
    assert (p.runs / "billiards").exists()
