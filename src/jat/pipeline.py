"""The main run: data of three worlds, V-JEPA 2 latents, one world model (one translator, one
predictor) trained on the three at once, and the agent's evaluations on them plus a fourth world
it never saw. notebooks/train_and_evaluate.ipynb runs it, one stage per cell.

Each stage skips what is already done, so running it again continues where the last attempt
stopped. Long steps run as `jat` subprocesses with their own log in <work>/logs; GPU steps are
pinned to a GPU; evaluations run in the background, those without the world model from the start.

`jat pipeline --size small --provider dry` runs everything at toy sizes with a scripted agent: a
dry run with no API and no GPU.
"""

import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

WORLDS = ("pusht", "tabletop", "billiards")  # the world model learns these, together
NEW_LOOK = "tabletop-new-look"  # evaluation only: the tabletop in another room, never trained on
FAMILIES = {  # the hardest task families of each world (tabletop translate: 94% alone)
    "pusht": "rotate,pose",
    "tabletop": "rotate,pose",
    "billiards": "shot,bank,carom",
    NEW_LOOK: "rotate,pose",
}


@dataclass(frozen=True)
class Sizes:
    episodes: int  # per 3D world
    pusht_episodes: int
    new_look_episodes: int  # offline check of the world model only
    steps: int  # pushes per episode
    translator_steps: int
    predictor_steps: int  # one world per step, in turn
    rollout: int
    tasks: int  # per family


SIZES = {
    "full": Sizes(
        episodes=5000,
        pusht_episodes=6000,
        new_look_episodes=500,
        steps=8,
        translator_steps=8000,
        predictor_steps=60000,
        rollout=4,
        tasks=100,
    ),
    "small": Sizes(
        episodes=12,
        pusht_episodes=12,
        new_look_episodes=8,
        steps=3,
        translator_steps=20,
        predictor_steps=10,
        rollout=2,
        tasks=1,
    ),
}


def job_env(gpu: int | None) -> dict[str, str]:
    """The environment of a job pinned to one GPU. The job sees only that GPU, as device 0, for
    CUDA and for MuJoCo's EGL rendering alike: never set MUJOCO_EGL_DEVICE_ID to the physical
    index, EGL would refuse it."""
    env = os.environ.copy()
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


# One job per GPU, started like the pipeline's own jobs: CUDA, V-JEPA 2's config, MuJoCo
# physics and rendering of both 3D worlds, the CLI.
_JOB_CHECK = """
from concurrent.futures import ThreadPoolExecutor
import torch
from jat.cli import app
from jat.wm.encoder import VJEPA2_ID
from jat.worlds import get_world
assert torch.cuda.is_available() or {cpu}, "no CUDA in this job"
device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
x = torch.ones(1024, device="cuda" if torch.cuda.is_available() else "cpu").sum().item()
tabletop = get_world("tabletop").render([256.0, 200.0, 0.0])
billiards = get_world("billiards").render([150.0, 256.0, 400.0, 100.0])
assert tabletop.std() > 5 and billiards.std() > 5, "blank frames"
with ThreadPoolExecutor(4) as pool:  # evaluation workers render from several threads
    frames = list(pool.map(lambda _: get_world("tabletop").render([256.0, 200.0, 0.0]), range(8)))
assert all((f == tabletop).all() for f in frames), "frames differ between threads"
print(f"{{device}}, renders ok ({{tabletop.shape[0]}} px, from 4 threads too)")
"""


def preflight(gpus: int = 0) -> None:
    """Checks everything the run needs before the run needs it: every jat module imports in
    this process, and a job on each GPU (started exactly like the pipeline's jobs) sees CUDA
    and renders both 3D worlds, from several threads too. Raises if anything fails."""
    import importlib
    import pkgutil

    import jat

    failures, lines = [], []
    for module in sorted(m.name for m in pkgutil.walk_packages(jat.__path__, "jat.")):
        try:
            importlib.import_module(module)
            lines.append(f"  ok      import {module}")
        except Exception as error:  # noqa: BLE001 (report every failure, then raise once)
            failures.append(module)
            lines.append(f"  FAILED  import {module}: {type(error).__name__}: {error}")
    for gpu in range(gpus) if gpus else [None]:
        name = f"job on GPU {gpu}" if gpu is not None else "job on CPU"
        code = _JOB_CHECK.format(cpu=gpu is None)
        run = subprocess.run(
            [sys.executable, "-c", code], env=job_env(gpu), capture_output=True, text=True
        )
        last = (run.stdout or run.stderr).strip().splitlines()[-1:] or ["no output"]
        if run.returncode != 0:
            failures.append(name)
        lines.append(f"  {'ok     ' if run.returncode == 0 else 'FAILED '} {name}: {last[0]}")
    print("\n".join(lines))
    if failures:
        raise RuntimeError(f"preflight failed: {', '.join(failures)}")
    print("Everything the run needs works.")


class Pipeline:
    def __init__(
        self,
        work: Path,
        size: str = "full",
        gpus: int = 0,
        provider: str = "meta",
        eval_workers: int = 8,
    ):
        self.work = work.resolve()  # absolute: the jat subprocesses run inside `work`
        self.sizes = SIZES[size]
        self.gpus, self.provider, self.eval_workers = gpus, provider, eval_workers
        self.data, self.runs, self.results = (self.work / d for d in ("data", "runs", "results"))
        self.logs = self.work / "logs"
        self.wm = self.runs / "wm"
        for folder in (self.data, self.runs, self.results, self.logs):
            folder.mkdir(parents=True, exist_ok=True)
        self.background: dict[str, subprocess.Popen] = {}

    # Running jat commands

    def _start(self, name: str, args: list[str], gpu: int | None = None) -> subprocess.Popen:
        env = job_env(gpu)
        log = open(self.logs / f"{name}.log", "a")  # noqa: SIM115 (closed with the process)
        log.write(f"\n$ jat {' '.join(args)}\n")
        log.flush()
        cmd = [sys.executable, "-c", "from jat.cli import app; app()", *args]
        print(
            f"[{time.strftime('%H:%M:%S')}] start {name}"
            + (f" (GPU {gpu})" if gpu is not None else "")
        )
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=self.work)

    def _tail(self, name: str, lines: int = 12) -> str:
        text = (self.logs / f"{name}.log").read_text().splitlines()
        return "\n".join(f"  | {line}" for line in text[-lines:])

    def _wait(self, jobs: dict[str, subprocess.Popen]) -> None:
        failed = []
        for name, process in jobs.items():
            code = process.wait()
            print(f"[{time.strftime('%H:%M:%S')}] {'done' if code == 0 else 'FAILED'} {name}")
            print(self._tail(name))
            if code != 0:
                failed.append(name)
        if failed:
            raise RuntimeError(f"failed: {', '.join(failed)}; see {self.logs}")

    def _run(self, jobs: list[tuple[str, list[str]]], gpu_jobs: bool = False) -> None:
        """Run jobs, in parallel over the GPUs when `gpu_jobs` and there are several."""
        if not jobs:
            return
        lanes = max(1, self.gpus) if gpu_jobs else 1
        pending = list(jobs)
        while pending:
            batch, pending = pending[:lanes], pending[lanes:]
            gpus = range(len(batch)) if gpu_jobs and self.gpus else [None] * len(batch)
            self._wait({n: self._start(n, a, g) for (n, a), g in zip(batch, gpus, strict=True)})

    def _eval(
        self, name: str, world: str, settings: str, extra: list[str] = (), folder: str = ""
    ) -> None:
        """An evaluation in the background, once per name. Runs with the same folder add up:
        one results folder per world (`folder` overrides it)."""
        if name in self.background:
            return
        folder = folder or world
        args = [
            "eval", "--out", str(self.results / folder), "--world", world, "--families",
            FAMILIES[folder], "--n", str(self.sizes.tasks),
            "--settings", settings, "--provider", self.provider,
            "--workers", str(self.eval_workers), *extra,
        ]  # fmt: skip
        self.background[name] = self._start(name, args)

    # Stages

    def stage_baseline_evals(self) -> None:
        """The agent alone and with the real physics in every world, in the background: they
        need no world model, so they run while it trains."""
        for world in (*WORLDS, NEW_LOOK):
            self._eval(f"eval-{world}-base", world, "no_tool,oracle_tool")

    def stage_data(self) -> None:
        """Random-push episodes of every world (CPU). Not needed again once the world model is
        trained and checked: a resumed run's output has runs, not datasets."""
        if self._checked():
            return
        s = self.sizes
        episodes = {
            "pusht": s.pusht_episodes,
            "tabletop": s.episodes,
            "billiards": s.episodes,
            NEW_LOOK: s.new_look_episodes,
        }
        self._run(
            [
                self._gen_job(world, n)
                for world, n in episodes.items()
                if not (self.data / world / "rollouts.npz").exists()
            ]
        )

    def _gen_job(self, world: str, episodes: int) -> tuple[str, list[str]]:
        args = ["gen", str(self.data / world), "--world", world, "--episodes", str(episodes)]
        return f"gen-{world}", [*args, "--steps", str(self.sizes.steps)]

    def stage_encode(self) -> None:
        """V-JEPA 2 latents of every dataset, spread over the GPUs."""
        jobs = [
            (f"encode-{d.name}", ["encode", str(d)])
            for d in sorted(self.data.iterdir())
            if (d / "rollouts.npz").exists() and not (d / "latents.json").exists()
        ]
        self._run(jobs, gpu_jobs=True)

    def _datasets(self) -> list[str]:
        return [str(self.data / w) for w in WORLDS]

    def stage_translator(self) -> None:
        """One translator for the three worlds: a shared trunk, one head per world."""
        if (self.wm / "translator.pt").exists():
            return
        steps = ["--steps", str(self.sizes.translator_steps)]
        args = ["train", "translator", *self._datasets(), "--out", str(self.wm), *steps]
        self._run([("translator", args)], gpu_jobs=True)

    def stage_predictor(self) -> None:
        """One predictor for the three worlds, read through that translator."""
        if (self.wm / "wm_eval.json").exists():
            return
        s = self.sizes
        args = [
            "train", "predictor", *self._datasets(), "--out", str(self.wm),
            "--steps", str(s.predictor_steps), "--rollout", str(s.rollout),
        ]  # fmt: skip
        self._run([("predictor", args)], gpu_jobs=True)

    def stage_wm_evals(self) -> None:
        """The agent with the world model in every world, in the background."""
        for world in (*WORLDS, NEW_LOOK):
            self._eval(f"eval-{world}-wm", world, "wm_tool", ["--wm", str(self.wm)])

    def _checked(self) -> bool:
        return (self.wm / f"wm_eval_{NEW_LOOK}.json").exists()

    def stage_check_new_look(self) -> None:
        """The world model against the real physics in the room it never saw (the trained
        worlds' test errors come with the predictor, in wm_eval.json)."""
        if self._checked():
            return
        args = ["wm-eval", str(self.data / NEW_LOOK), "--wm", str(self.wm), "--head", "tabletop"]
        self._run([(f"wm-eval-{NEW_LOOK}", args)], gpu_jobs=True)

    def stage_finish(self) -> Path:
        """Wait for every evaluation, write the reports, pack runs + results + logs."""
        for name, process in list(self.background.items()):
            if process.wait() != 0:  # episodes on disk are skipped: this reruns the failed ones
                print(f"[{time.strftime('%H:%M:%S')}] {name} had failures; one retry")
                args = process.args[3:]
                self.background[name] = process = self._start(name, args)
            code = process.wait()
            print(f"[{time.strftime('%H:%M:%S')}] {'done' if code == 0 else 'FAILED'} {name}")
            print(self._tail(name, 6))
        for folder in sorted(self.results.iterdir()):
            if (folder / "episodes").is_dir():
                self._run([(f"report-{folder.name}", ["report", str(folder)])])
                shutil.copy(self.logs / f"report-{folder.name}.log", folder / "report.txt")
        archive = self.work / "results.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for top in (self.runs, self.results, self.logs):
                for path in sorted(top.rglob("*")):
                    if path.is_file():
                        z.write(path, path.relative_to(self.work))
        print(f"{archive} ({archive.stat().st_size / 1e6:.0f} MB)")
        return archive

    def run_all(self) -> Path:
        self.stage_baseline_evals()
        self.stage_data()
        self.stage_encode()
        self.stage_translator()
        self.stage_predictor()
        self.stage_wm_evals()
        self.stage_check_new_look()
        return self.stage_finish()
