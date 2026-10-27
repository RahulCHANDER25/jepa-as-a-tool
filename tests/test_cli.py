from typer.testing import CliRunner

from jat.cli import app

runner = CliRunner()


def test_help_lists_every_pipeline_stage():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("gen", "encode", "train", "wm-eval", "eval", "export-web"):
        assert command in result.output


def test_gen_writes_a_dataset(tmp_path):
    result = runner.invoke(
        app, ["gen", str(tmp_path), "--episodes", "2", "--steps", "1", "--workers", "1"]
    )
    assert result.exit_code == 0 and (tmp_path / "rollouts.npz").exists()


def test_wm_tool_without_a_world_model_is_refused(tmp_path):
    result = runner.invoke(app, ["eval", "--out", str(tmp_path), "--settings", "wm_tool"])
    assert result.exit_code != 0 and "--wm" in result.output


def test_unimplemented_stage_exits_with_error():
    result = runner.invoke(app, ["export-web"])
    assert result.exit_code == 1
