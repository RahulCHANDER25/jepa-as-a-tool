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


def test_a_resumed_rebuild_only_reruns_the_evaluations(tmp_path, monkeypatch):
    from jat.pipeline import Pipeline

    p = Pipeline(tmp_path, eval_v2=tmp_path)
    for path in ("billiards/wm_eval.json", "transfer/transfer.json", "pusht-ft/wm_eval.json",
                 "pusht-ft/wm_eval_agent_ft.json", "billiards-physics-2"):  # fmt: skip
        (p.runs / path).parent.mkdir(parents=True, exist_ok=True)
        (p.runs / path).touch()
    evals, jobs = [], []
    monkeypatch.setattr(p, "_eval", lambda name, *args, **kw: evals.append(name))
    monkeypatch.setattr(p, "_run", lambda batch, **kw: jobs.extend(batch))
    p.reset_billiards()
    p.stage_billiards_data()
    assert (p.runs / "billiards").exists() and not jobs  # no new data: the models are kept
    assert {"eval-billiards-base", "eval-billiards-wm", "eval-pusht-ft-wm"} <= set(evals)
