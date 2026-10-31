from jat.pipeline import job_env


def test_a_job_on_the_second_gpu_sees_it_as_its_only_device(monkeypatch):
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")
    env = job_env(1)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "MUJOCO_EGL_DEVICE_ID" not in env  # EGL numbers the visible GPUs from 0
