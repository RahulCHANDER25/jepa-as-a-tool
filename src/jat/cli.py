"""Command-line entry point: one subcommand per pipeline stage."""

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

app = typer.Typer(no_args_is_help=True, help="jepa-as-a-tool pipeline.")


@app.callback()
def _load_env() -> None:
    load_dotenv()


def _not_ready(milestone: str) -> None:
    typer.echo(f"Not implemented yet (planned for {milestone}).", err=True)
    raise typer.Exit(code=1)


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@app.command()
def tasks(
    family: Annotated[str, typer.Argument(help="translate | rotate | pose")],
    n: int = 5,
    seed: int = 0,
) -> None:
    """Print sampled tasks: start pose, goal, and the reference plan length."""
    from jat.env import geometry as geo
    from jat.env.tasks import sample_tasks

    for task in sample_tasks(family, n, seed):
        typer.echo(f"{task.id}  ({len(task.ref_plan)}-push reference plan)\n  {task.text}")
        typer.echo("  start: " + geo.pose_text(task.start, task.goal).replace("\n", "\n  "))


@app.command()
def gen() -> None:
    """Generate random-push rollouts in PushT."""
    _not_ready("M3")


@app.command()
def encode() -> None:
    """Encode rendered frames with V-JEPA 2 (Modal GPU)."""
    _not_ready("M3")


@app.command()
def train() -> None:
    """Train the translator or the predictor (Modal GPU)."""
    _not_ready("M4")


@app.command(name="eval")
def eval_(
    out: Annotated[Path, typer.Option(help="Run folder; episodes already in it are skipped.")],
    families: str = "rotate",
    n: int = 30,
    task_seed: int = 0,
    settings: str = "no_tool,oracle_tool,random",
    seeds: int = 1,
    provider: str = "baseten",
    temperature: float = 0.7,
    workers: int = 8,
) -> None:
    """Run agent episodes over tasks and settings, then print the success table."""
    from jat.agent.episode_log import AgentInfo
    from jat.agent.llm import chat_model
    from jat.agent.prompts import PROMPT_VERSION
    from jat.env.tasks import sample_tasks
    from jat.eval.metrics import format_table, summarize
    from jat.eval.run import run_grid

    task_list = [t for f in _csv(families) for t in sample_tasks(f, n, task_seed)]
    setting_list = _csv(settings)

    def model_factory():
        model = chat_model(provider, temperature)
        return model, AgentInfo(
            provider=provider,
            model=model.model_name,
            prompt_version=PROMPT_VERSION,
            temperature=temperature,
        )

    needs_llm = any(s != "random" for s in setting_list)
    if needs_llm:
        model_factory()  # fail now on missing keys, not inside every worker
    logs, failures = run_grid(
        task_list,
        setting_list,
        list(range(seeds)),
        model_factory if needs_llm else None,
        out,
        workers,
    )
    typer.echo("\n" + format_table(summarize(logs)))
    if failures:
        typer.echo(f"\n{failures} episodes failed; rerun the same command to retry them.", err=True)
        raise typer.Exit(code=1)


@app.command()
def report(run: Annotated[Path, typer.Argument(help="Run folder written by `jat eval`.")]) -> None:
    """Print the success table of a finished run."""
    from jat.eval.metrics import format_table, summarize
    from jat.eval.run import load_logs

    typer.echo(format_table(summarize(load_logs(run))))


@app.command(name="export-web")
def export_web() -> None:
    """Copy results and episode logs into the website data folder."""
    _not_ready("M7")
