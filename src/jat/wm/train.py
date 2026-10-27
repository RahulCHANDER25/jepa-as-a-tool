"""Training. The translator first (it is also the probe: can V-JEPA 2 latents show the pose?),
then the predictor, which reads its own predictions through the frozen translator.

Both write into one run folder:
  translator.pt, translator_eval.json, predictor.pt, wm_eval.json
"""

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from jat.wm.data import TEST, TRAIN, VAL
from jat.wm.dataset import LatentData
from jat.wm.evaluate import format_rows, rollout_errors, translator_errors
from jat.wm.models import (
    InverseDynamics,
    Predictor,
    Translator,
    keypoint_loss,
    load_module,
    save_module,
)


def _schedule(optimizer: torch.optim.Optimizer, steps: int, warmup: int):
    """Linear warmup, then cosine down to 5% of the base rate."""

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _log(step: int, steps: int, started: float, text: str) -> None:
    elapsed = time.time() - started
    eta = elapsed / max(step, 1) * (steps - step)
    print(
        f"[{step:>6}/{steps}] {text}  ({elapsed / 60:.1f} min, ~{eta / 60:.0f} min left)",
        flush=True,
    )


def train_translator(
    data: LatentData,
    out: Path,
    steps: int = 5000,
    batch: int = 256,
    lr: float = 1e-3,
    log_every: int = 500,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    model = Translator(data.n_tokens, data.dim).to(data.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = _schedule(optimizer, steps, warmup=steps // 20)
    episodes = data.episodes(TRAIN)
    started = time.time()

    for step in range(1, steps + 1):
        e = episodes[torch.randint(len(episodes), (batch,), device=data.device)]
        t = torch.randint(data.n_steps + 1, (batch,), device=data.device)
        loss = keypoint_loss(model(data.latent(e, t)), data.poses[e, t])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % log_every == 0 or step == steps:
            model.eval()
            val = translator_errors(data, model, VAL)
            model.train()
            _log(
                step,
                steps,
                started,
                f"loss {loss.item():.4f}  val {val['px_median']:.1f} px, {val['deg_median']:.1f}°",
            )

    model.eval()
    result = {
        "val": translator_errors(data, model, VAL),
        "test": translator_errors(data, model, TEST),
    }
    save_module(model, out / "translator.pt", latents=data.meta, metrics=result)
    (out / "translator_eval.json").write_text(json.dumps(result, indent=1))
    return result


def train_predictor(
    data: LatentData,
    out: Path,
    steps: int = 20000,
    batch: int = 64,
    rollout: int = 3,
    lr: float = 3e-4,
    keypoint_weight: float = 1.0,
    idm_weight: float = 0.1,
    log_every: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """Each sample: start from a real latent, apply `rollout` pushes in a row, feeding back the
    predictor's own output. Losses per step: latent L1 to the true latent, landmarks read by the
    frozen translator, and inverse dynamics on pushes that moved the T."""
    torch.manual_seed(seed)
    translator, _ = load_module(Translator, out / "translator.pt", data.device)
    translator.requires_grad_(False)
    predictor = Predictor(data.n_tokens, data.dim).to(data.device)
    idm = InverseDynamics(data.n_tokens, data.dim).to(data.device)
    params = [*predictor.parameters(), *idm.parameters()]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    scheduler = _schedule(optimizer, steps, warmup=steps // 20)
    cuda = data.device.type == "cuda"
    # fp16 autocast: T4 and P100 (Kaggle, Colab free) have no fast bf16
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if cuda else nullcontext
    episodes = data.episodes(TRAIN)
    rollout = min(rollout, data.n_steps)
    best, history, started = math.inf, [], time.time()

    for step in range(1, steps + 1):
        e = episodes[torch.randint(len(episodes), (batch,), device=data.device)]
        t0 = torch.randint(data.n_steps - rollout + 1, (batch,), device=data.device)
        z = data.latent(e, t0)
        # One tensor per key, summed out of place: `+=` on a shared zero would alias them all
        losses = {k: torch.zeros((), device=data.device) for k in ("latent", "keypoint", "idm")}
        with autocast():
            for k in range(rollout):
                push, moved = data.pushes[e, t0 + k], data.moved[e, t0 + k]
                target = data.latent(e, t0 + k + 1)
                z_next = predictor(z, push)
                losses["latent"] = losses["latent"] + F.l1_loss(z_next, target) / rollout
                losses["keypoint"] = losses["keypoint"] + (
                    keypoint_loss(translator(z_next).float(), data.poses[e, t0 + k + 1]) / rollout
                )
                if k == 0 and moved.any():  # a push that missed says nothing about the push
                    guess_true = idm(z[moved], target[moved])
                    guess_pred = idm(z[moved], z_next[moved])
                    losses["idm"] = F.mse_loss(guess_true.float(), push[moved]) + F.mse_loss(
                        guess_pred.float(), push[moved]
                    )
                z = z_next
            loss = (
                losses["latent"] + keypoint_weight * losses["keypoint"] + idm_weight * losses["idm"]
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % log_every == 0 or step == steps:
            predictor.eval()
            rows = rollout_errors(data, predictor, translator, VAL)
            predictor.train()
            # Selection score: median px + median degrees, summed over the trained horizons
            score = sum(r["px_median"] + r["deg_median"] for r in rows[1 : rollout + 1])
            history.append(
                {"step": step, "score": score, **{k: float(v.detach()) for k, v in losses.items()}}
            )
            mark = ""
            if score < best:
                best, mark = score, "  * saved"
                save_module(predictor, out / "predictor.pt", step=step)
            parts = "  ".join(f"{k} {v.item():.4f}" for k, v in losses.items())
            r1 = rows[1]
            _log(
                step,
                steps,
                started,
                f"{parts}  val 1 push: {r1['px_median']:.1f} px, {r1['deg_median']:.1f}°{mark}",
            )

    predictor, _ = load_module(Predictor, out / "predictor.pt", data.device)
    rows = rollout_errors(data, predictor, translator, TEST)
    (out / "wm_eval.json").write_text(json.dumps({"test": rows, "history": history}, indent=1))
    print("\nTest set, imagined pose vs true pose:\n" + format_rows(rows))
    return rows
