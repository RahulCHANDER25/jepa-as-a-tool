from jat.pipeline import job_env


def test_a_job_on_the_second_gpu_sees_it_as_its_only_device(monkeypatch):
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    env = job_env(1)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "MUJOCO_EGL_DEVICE_ID" not in env  # EGL numbers the visible GPUs from 0


def test_a_resumed_run_with_a_checked_world_model_only_evaluates(tmp_path, monkeypatch):
    from jat.pipeline import NEW_LOOK, Pipeline

    p = Pipeline(tmp_path)
    for name in ("translator.pt", "wm_eval.json", f"wm_eval_{NEW_LOOK}.json"):
        p.wm.mkdir(parents=True, exist_ok=True)
        (p.wm / name).touch()
    evals, jobs = [], []
    monkeypatch.setattr(p, "_eval", lambda name, *args, **kw: evals.append(name))
    monkeypatch.setattr(p, "_run", lambda batch, **kw: jobs.extend(batch))
    for stage in (p.stage_baseline_evals, p.stage_data, p.stage_encode, p.stage_translator,
                  p.stage_predictor, p.stage_wm_evals, p.stage_check_new_look):  # fmt: skip
        stage()
    assert not jobs  # no data, no training: the output holds the trained model
    assert len(evals) == 8 and "eval-billiards-wm" in evals
