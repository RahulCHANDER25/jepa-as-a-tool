from typer.testing import CliRunner

from jat.cli import app

runner = CliRunner()


def test_help_lists_every_pipeline_stage():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("gen", "encode", "train", "eval", "export-web"):
        assert command in result.output


def test_unimplemented_stage_exits_with_error():
    result = runner.invoke(app, ["gen"])
    assert result.exit_code == 1
