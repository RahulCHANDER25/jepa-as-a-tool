"""The overnight run: data, V-JEPA 2 latents, translators (specialists, the shared one, the
transfer test), predictors, learning from use, and every evaluation, as stages.

Each step skips itself when its output exists, so running a stage again continues where the
last attempt stopped. Long steps run as `jat` subprocesses with their own log in <work>/logs;
GPU steps spread over the GPUs; evaluations run in the background as soon as what they need
exists (episodes without the world model start right away).

notebooks/overnight.ipynb runs one stage per cell. `jat pipeline --size small --provider dry`
runs everything at toy sizes with a scripted agent: a dry run with no API and no GPU.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

NEW_WORLDS = ("tabletop", "billiards")
NEW_LOOK = "tabletop-new-look"
FAMILIES = {
    "tabletop": "translate,rotate,pose",
    "billiards": "shot,bank,carom",
    NEW_LOOK: "rotate,pose",
    "pusht-ft": "pose",
}
FT_TASK_SEED = 1  # the fine-tune learned from the task seed 0 logs: evaluate on other tasks
GATE3_SLACK = 1.25  # the shared translator passes within 25% of each specialist


@dataclass(frozen=True)
class Sizes:
    episodes: int  # per new world
    steps: int  # pushes per episode
    new_look_episodes: int  # offline zero-shot check only
    pusht_episodes: int  # only if the PushT v2 data is not attached
    translator_steps: int
    shared_steps: int
    transfer_budgets: str
    transfer_steps: int
    predictor_steps: int
    rollout: int
    finetune_steps: int
    tasks: int  # per family
    learn_limit: int | None


SIZES = {
    "full": Sizes(
        episodes=5000,
        steps=8,
        new_look_episodes=500,
        pusht_episodes=6000,
        translator_steps=6000,
        shared_steps=8000,
        transfer_budgets="250,1000,4000",
        transfer_steps=3000,
        predictor_steps=25000,
        rollout=4,
        finetune_steps=8000,
        tasks=50,
        learn_limit=None,
    ),
    "small": Sizes(
        episodes=12,
        steps=3,
        new_look_episodes=8,
        pusht_episodes=12,
        translator_steps=20,
        shared_steps=20,
        transfer_budgets="4,8",
        transfer_steps=10,
        predictor_steps=10,
        rollout=2,
        finetune_steps=10,
        tasks=1,
        learn_limit=40,
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
print(f"{{device}}, renders ok ({{tabletop.shape[0]}} px)")
"""


def preflight(gpus: int = 0) -> None:
    """Checks everything the run needs before the run needs it: every jat module imports in
    this process, and a job on each GPU (started exactly like the pipeline's jobs) sees CUDA
    and renders both 3D worlds. Raises if anything fails."""
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
        wm_v2: Path | None = None,
        eval_v2: Path | None = None,
        pusht_data: Path | None = None,
        gpus: int = 0,
        provider: str = "meta",
        eval_workers: int = 6,
    ):
        self.work = work.resolve()
        self.sizes = SIZES[size]
        # Absolute: the jat subprocesses run inside `work`
        self.wm_v2, self.eval_v2, self.pusht_data = (
            p.resolve() if p is not None else None for p in (wm_v2, eval_v2, pusht_data)
        )
        self.gpus, self.provider, self.eval_workers = gpus, provider, eval_workers
        self.data, self.runs, self.results = (self.work / d for d in ("data", "runs", "results"))
        self.logs = self.work / "logs"
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

    def stage_data(self) -> None:
        """Random-push episodes of the new worlds (CPU), PushT's (attached or generated), and
        the agent's own imagined pushes from the PushT evaluation logs."""
        s = self.sizes
        jobs = []
        for world, episodes in (
            *((w, s.episodes) for w in NEW_WORLDS),
            (NEW_LOOK, s.new_look_episodes),
        ):
            if not (self.data / world / "rollouts.npz").exists():
                args = ["gen", str(self.data / world), "--world", world, "--episodes"]
                jobs.append((f"gen-{world}", [*args, str(episodes), "--steps", str(s.steps)]))
        pusht = self.data / "pusht"
        if not pusht.exists():
            if self.pusht_data is not None:
                pusht.symlink_to(self.pusht_data.resolve(), target_is_directory=True)
                print(f"PushT data: {self.pusht_data}")
            else:
                args = ["gen", str(pusht), "--episodes", str(s.pusht_episodes)]
                jobs.append(("gen-pusht", [*args, "--steps", str(s.steps)]))
        agent = self.data / "pusht-agent"
        if self.eval_v2 is not None and not (agent / "rollouts.npz").exists():
            args = ["learn-data", str(agent), "--run", str(self.eval_v2)]
            jobs.append(
                ("learn-data", args + (["--limit", str(s.learn_limit)] if s.learn_limit else []))
            )
        self._run(jobs)

    def stage_baseline_evals(self) -> None:
        """Episodes that need no new world model, in the background: alone and with the real
        physics in every new world; PushT pose tasks on a new task seed, alone and with v2."""
        for world in (*NEW_WORLDS, NEW_LOOK):
            self._eval(f"eval-{world}-base", world, "no_tool,oracle_tool")
        if self.wm_v2 is not None:
            extra = ["--task-seed", str(FT_TASK_SEED), "--wm", str(self.wm_v2)]
            self._eval("eval-pusht-ft-base", "pusht", "no_tool,wm_tool", extra, "pusht-ft")

    def stage_encode(self) -> None:
        """V-JEPA 2 latents of every dataset, spread over the GPUs."""
        jobs = [
            (f"encode-{d.name}", ["encode", str(d)])
            for d in sorted(self.data.iterdir())
            if (d / "rollouts.npz").exists() and not (d / "latents.json").exists()
        ]
        self._run(jobs, gpu_jobs=True)

    def stage_translators(self) -> dict:
        """Specialists, the shared translator (3 worlds), a trunk on 2 worlds and the transfer
        test on billiards; Gate 3 picks the translator each new world's agent reads with."""
        s, d, r = self.sizes, self.data, self.runs
        steps = ["--steps", str(s.translator_steps)]
        specialists = [
            (
                f"translator-{w}",
                ["train", "translator", str(d / w), "--out", str(r / f"{w}-spec"), *steps],
            )
            for w in NEW_WORLDS
            if not (r / f"{w}-spec" / "translator.pt").exists()
        ]
        self._run(specialists, gpu_jobs=True)
        shared_steps = ["--steps", str(s.shared_steps)]
        jobs = []
        if not (r / "shared" / "translator.pt").exists():
            data = [str(d / "pusht"), *(str(d / w) for w in NEW_WORLDS)]
            jobs.append(
                (
                    "translator-shared",
                    ["train", "translator", *data, "--out", str(r / "shared"), *shared_steps],
                )
            )
        if not (r / "trunk2" / "translator.pt").exists():
            data = [str(d / "pusht"), str(d / "tabletop")]
            jobs.append(
                (
                    "translator-trunk2",
                    ["train", "translator", *data, "--out", str(r / "trunk2"), *shared_steps],
                )
            )
        self._run(jobs)  # one at a time: each may hold PushT's latents in RAM
        if not (r / "transfer" / "transfer.json").exists():
            args = [
                "transfer", str(d / "billiards"), "--trunk-from", str(r / "trunk2" / "translator.pt"),
                "--out", str(r / "transfer"), "--budgets", s.transfer_budgets,
                "--steps", str(s.transfer_steps),
            ]  # fmt: skip
            self._run([("transfer", args)])
        return self._gate3()

    def _gate3(self) -> dict:
        import torch

        from jat.wm.data import TEST
        from jat.wm.dataset import LatentData
        from jat.wm.evaluate import translator_errors
        from jat.wm.models import Translator, load_module

        r = self.runs
        shared = json.loads((r / "shared" / "translator_eval.json").read_text())
        table, gates = {}, {}
        for world in NEW_WORLDS:
            spec = json.loads((r / f"{world}-spec" / "translator_eval.json").read_text())[world][
                "test"
            ]
            both = shared[world]["test"]
            ok = all(both[k] <= GATE3_SLACK * spec[k] + 0.5 for k in ("px_median", "deg_median"))
            gates[world] = "shared" if ok else "specialist"
            table[world] = {"specialist": spec, "shared": both}
            chosen = r / ("shared" if ok else f"{world}-spec") / "translator.pt"
            (r / world).mkdir(exist_ok=True)
            shutil.copy(chosen, r / world / "translator.pt")
        table["pusht"] = {"shared": shared["pusht"]["test"]}
        # Zero-shot: the new look read by the tabletop head, never trained on it
        look = LatentData(self.data / NEW_LOOK, torch.device("cpu"))
        for name in ("tabletop-spec", "shared"):
            translator, _ = load_module(Translator, r / name / "translator.pt")
            table.setdefault(NEW_LOOK, {})[name] = translator_errors(
                look, translator, TEST, "tabletop"
            )
        transfer = json.loads((r / "transfer" / "transfer.json").read_text())
        summary = {"gate3": gates, "translators": table, "transfer": transfer}
        (self.results / "translators.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps({"gate3": gates}, indent=1))
        for world, rows in table.items():
            for name, e in rows.items():
                print(f"{world:>18} {name:>14}: {e['px_median']:.1f} px, {e['deg_median']:.1f}°")
        return summary

    def stage_predictors(self) -> None:
        """Predictors of the new worlds, one per GPU, then PushT fine-tuned on the agent's own
        pushes. Each world's evaluation with its world model starts as soon as it is trained."""
        s, d, r = self.sizes, self.data, self.runs
        common = ["--steps", str(s.predictor_steps), "--rollout", str(s.rollout)]
        jobs = {
            w: ["train", "predictor", str(d / w), "--out", str(r / w), *common]
            for w in NEW_WORLDS
            if not (r / w / "wm_eval.json").exists()
        }
        ft = r / "pusht-ft"
        if (
            self.wm_v2 is not None
            and (d / "pusht-agent").exists()
            and not (ft / "wm_eval.json").exists()
        ):
            ft.mkdir(exist_ok=True)
            shutil.copy(self.wm_v2 / "translator.pt", ft / "translator.pt")
            jobs["pusht-ft"] = [
                "train", "predictor", str(d / "pusht"), "--out", str(ft),
                "--init", str(self.wm_v2 / "predictor.pt"), "--extra", str(d / "pusht-agent"),
                "--steps", str(s.finetune_steps), "--rollout", str(min(s.rollout + 1, s.steps)),
                "--lr", "1e-4",
            ]  # fmt: skip
        lanes = max(1, self.gpus)
        free = list(range(self.gpus)) if self.gpus else [None]
        running: dict[str, tuple[subprocess.Popen, int | None]] = {}
        queue = list(jobs)
        self._evals_with_models()
        while queue or running:
            while queue and len(running) < lanes:
                name = queue.pop(0)
                gpu = free.pop(0)
                running[name] = (self._start(f"predictor-{name}", jobs[name], gpu), gpu)
            time.sleep(5)
            for name, (process, gpu) in list(running.items()):
                if process.poll() is None:
                    continue
                del running[name]
                free.append(gpu)
                self._wait({f"predictor-{name}": process})
                self._evals_with_models()
        for world in NEW_WORLDS:
            if not (r / world / f"wm_eval_{world}.json").exists():
                self._run(
                    [(f"wm-eval-{world}", ["wm-eval", str(d / world), "--wm", str(r / world)])]
                )
        look = r / "tabletop" / f"wm_eval_{NEW_LOOK}.json"
        if not look.exists():
            args = ["wm-eval", str(d / NEW_LOOK), "--wm", str(r / "tabletop"), "--head", "tabletop"]
            self._run([(f"wm-eval-{NEW_LOOK}", args)])

    def _evals_with_models(self) -> None:
        r = self.runs
        for world in NEW_WORLDS:
            if (r / world / "wm_eval.json").exists():
                self._eval(f"eval-{world}-wm", world, "wm_tool", ["--wm", str(r / world)])
        if (r / "tabletop" / "wm_eval.json").exists():
            self._eval(f"eval-{NEW_LOOK}-wm", NEW_LOOK, "wm_tool", ["--wm", str(r / "tabletop")])
        if (r / "pusht-ft" / "wm_eval.json").exists():
            extra = ["--task-seed", str(FT_TASK_SEED), "--wm", f"ft={r / 'pusht-ft'}"]
            self._eval("eval-pusht-ft-wm", "pusht", "wm_tool@ft", extra, "pusht-ft")

    def stage_finish(self) -> Path:
        """Wait for every evaluation, write the reports, pack runs + results + logs."""
        self._evals_with_models()
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
        archive = self.work / "overnight.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for top in (self.runs, self.results, self.logs):
                for path in sorted(top.rglob("*")):
                    if path.is_file():
                        z.write(path, path.relative_to(self.work))
        print(f"{archive} ({archive.stat().st_size / 1e6:.0f} MB)")
        return archive

    def run_all(self) -> Path:
        self.stage_data()
        self.stage_baseline_evals()
        self.stage_encode()
        self.stage_translators()
        self.stage_predictors()
        return self.stage_finish()
