"""Command-line entry point: one subcommand per pipeline stage."""

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

app = typer.Typer(no_args_is_help=True, help="jepa-as-a-tool pipeline.")


@app.callback()
def _load_env() -> None:
    load_dotenv()


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
def gen(
    out: Annotated[Path, typer.Argument(help="Dataset folder, e.g. data/random-4k.")],
    episodes: int = 4000,
    steps: int = 6,
    seed: int = 0,
    workers: int = 4,
) -> None:
    """Generate random-push episodes in PushT (CPU)."""
    from jat.wm.data import generate

    rollouts = generate(episodes, steps, seed, workers=workers)
    rollouts.save(out)
    typer.echo(
        f"{episodes} episodes x {steps} pushes -> {out / 'rollouts.npz'}  "
        f"(T moved in {rollouts.moved.mean():.0%} of pushes, jammed in {rollouts.jammed.mean():.1%})"
    )


@app.command()
def encode(
    data: Annotated[Path, typer.Argument(help="Dataset folder written by `jat gen`.")],
    pool: int = typer.Option(
        2, help="Average 2x2 patches: 64 tokens instead of 256, 4x less disk."
    ),
    batch: int = 64,
) -> None:
    """Render every pose and encode it with V-JEPA 2 (GPU recommended)."""
    from jat.wm.encoder import Encoder, encode_dataset

    encoder = Encoder(pool=pool)
    typer.echo(f"V-JEPA 2 on {encoder.device}: {encoder.n_tokens} tokens x {encoder.dim} per frame")
    encode_dataset(data, encoder, batch)


train_app = typer.Typer(no_args_is_help=True, help="Train the translator, then the predictor.")
app.add_typer(train_app, name="train")


@train_app.command("translator")
def train_translator(
    data: Annotated[Path, typer.Argument(help="Dataset folder with latents.")],
    out: Annotated[Path, typer.Option(help="Run folder, e.g. runs/wm.")],
    steps: int = 5000,
    batch: int = 256,
    lr: float = 1e-3,
) -> None:
    """Latent -> pose. Its test error says whether V-JEPA 2 latents show where the T is."""
    from jat.wm import train
    from jat.wm.dataset import LatentData
    from jat.wm.encoder import default_device

    result = train.train_translator(LatentData(data, default_device()), out, steps, batch, lr)
    test = result["test"]
    typer.echo(
        f"\nTranslator on test frames: median {test['px_median']:.1f} px, {test['deg_median']:.1f}°"
        f" (90% under {test['px_p90']:.1f} px, {test['deg_p90']:.1f}°)"
    )


@train_app.command("predictor")
def train_predictor(
    data: Annotated[Path, typer.Argument(help="Dataset folder with latents.")],
    out: Annotated[Path, typer.Option(help="Run folder holding translator.pt.")],
    steps: int = 20000,
    batch: int = 64,
    rollout: int = typer.Option(3, help="Pushes chained per training sample."),
    lr: float = 3e-4,
    keypoint_weight: float = 1.0,
    idm_weight: float = 0.1,
) -> None:
    """(latent, push) -> next latent, trained over chains of pushes."""
    from jat.wm import train
    from jat.wm.dataset import LatentData
    from jat.wm.encoder import default_device

    train.train_predictor(
        LatentData(data, default_device()),
        out,
        steps=steps,
        batch=batch,
        rollout=rollout,
        lr=lr,
        keypoint_weight=keypoint_weight,
        idm_weight=idm_weight,
    )


@app.command(name="wm-eval")
def wm_eval(
    data: Annotated[Path, typer.Argument(help="Dataset folder with latents.")],
    wm: Annotated[Path, typer.Option(help="Run folder with translator.pt and predictor.pt.")],
    split: str = "test",
) -> None:
    """Imagined pose vs true pose after 0, 1, 2, ... pushes. No LLM."""
    from jat.wm.data import TEST, VAL
    from jat.wm.dataset import LatentData
    from jat.wm.encoder import default_device
    from jat.wm.evaluate import format_rows, rollout_errors
    from jat.wm.models import Predictor, Translator, load_module

    device = default_device()
    translator, _ = load_module(Translator, wm / "translator.pt", device)
    predictor, _ = load_module(Predictor, wm / "predictor.pt", device)
    rows = rollout_errors(
        LatentData(data, device), predictor, translator, {"val": VAL, "test": TEST}[split]
    )
    typer.echo(format_rows(rows))


@app.command(name="eval")
def eval_(
    out: Annotated[Path, typer.Option(help="Run folder; episodes already in it are skipped.")],
    families: str = "rotate",
    n: int = 30,
    task_seed: int = 0,
    settings: str = "no_tool,oracle_tool,random",
    wm: Annotated[
        Path | None, typer.Option(help="World model run folder; needed for wm_tool.")
    ] = None,
    seeds: int = 1,
    provider: str = "meta",
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
    world_model = None
    if "wm_tool" in setting_list:
        if wm is None:
            raise typer.BadParameter("wm_tool needs --wm <run folder>")
        from jat.wm.models import WorldModel

        world_model = WorldModel.load(wm)

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
        world_model,
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
def export_web(
    run: Annotated[Path, typer.Option(help="Run folder written by `jat eval`.")],
    wm: Annotated[
        list[str], typer.Option(help="name=folder of a world-model run, e.g. v2=runs/wm-v2.")
    ] = [],  # noqa: B006 (typer reads the default, never mutates it)
    out: Path = Path("web/public/data"),
) -> None:
    """Copy results and episode logs into the website data folder."""
    from jat.eval.export import export_web as export

    models = {}
    for item in wm:
        name, sep, folder = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"expected name=folder, got {item!r}")
        models[name] = Path(folder)
    n = export(run, models, out)
    typer.echo(f"{n} episodes, results and {len(models)} world-model tables -> {out}")
