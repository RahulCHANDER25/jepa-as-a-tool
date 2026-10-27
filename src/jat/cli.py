"""Command-line entry point: one subcommand per pipeline stage."""

import typer

app = typer.Typer(no_args_is_help=True, help="jepa-as-a-tool pipeline.")


def _not_ready(milestone: str) -> None:
    typer.echo(f"Not implemented yet (planned for {milestone}).", err=True)
    raise typer.Exit(code=1)


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
def eval_() -> None:
    """Run agent episodes over tasks and settings."""
    _not_ready("M2")


@app.command(name="export-web")
def export_web() -> None:
    """Copy results and episode logs into the website data folder."""
    _not_ready("M7")
