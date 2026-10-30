"""Training. The translator first (it is also the probe: can V-JEPA 2 latents show the state?),
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
from jat.wm.models import InverseDynamics, Predictor, Translator, load_module, save_module


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


def _latents_meta(datasets: list[LatentData]) -> dict:
    """The encoder settings shared by all datasets; a translator only reads one kind of latent."""
    metas = {(d.meta["encoder"], d.meta["pool"], d.n_tokens, d.dim) for d in datasets}
    if len(metas) != 1:
        raise ValueError(f"datasets were encoded differently: {sorted(metas)}")
    encoder, pool, _, _ = metas.pop()
    return {"encoder": encoder, "pool": pool}


def train_translator(
    datasets: list[LatentData],
    out: Path,
    steps: int = 5000,
    batch: int = 256,
    lr: float = 1e-3,
    log_every: int = 500,
    seed: int = 0,
    trunk_from: Path | None = None,
) -> dict:
    """One trunk, one head per dataset's world; each batch takes an equal share of every
    dataset. `trunk_from`: start from that translator's trunk and keep it frozen (only the
    heads learn). Returns {world: {"val": errors, "test": errors}}."""
    torch.manual_seed(seed)
    device = datasets[0].device
    meta = _latents_meta(datasets)
    worlds = [d.world for d in datasets]
    if len(set(worlds)) != len(worlds):
        raise ValueError(f"one dataset per world, got {worlds}")
    heads = {d.world: d.readout.size for d in datasets}
    model = Translator(datasets[0].n_tokens, datasets[0].dim, heads).to(device)
    params = list(model.parameters())
    if trunk_from is not None:
        source, _ = load_module(Translator, trunk_from, device)
        trunk = {k: v for k, v in source.state_dict().items() if not k.startswith("heads.")}
        model.load_state_dict(trunk, strict=False)
        for p in model.trunk_parameters():
            p.requires_grad_(False)
        params = list(model.heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = _schedule(optimizer, steps, warmup=steps // 20)
    train_episodes = [d.episodes(TRAIN) for d in datasets]
    share = max(1, batch // len(datasets))
    started = time.time()

    for step in range(1, steps + 1):
        loss = torch.zeros((), device=device)
        for data, episodes in zip(datasets, train_episodes, strict=True):
            e = episodes[torch.randint(len(episodes), (share,), device=device)]
            t = torch.randint(data.n_steps + 1, (share,), device=device)
            out_k = model(data.latent(e, t), data.world)
            loss = loss + data.readout.loss(out_k, data.poses[e, t]) / len(datasets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % log_every == 0 or step == steps:
            model.eval()
            vals = [translator_errors(d, model, VAL) for d in datasets]
            model.train()
            parts = "  ".join(
                f"{d.world} {v['px_median']:.1f} px, {v['deg_median']:.1f}°"
                for d, v in zip(datasets, vals, strict=True)
            )
            _log(step, steps, started, f"loss {loss.item():.4f}  val {parts}")

    model.eval()
    result = {
        d.world: {
            "val": translator_errors(d, model, VAL),
            "test": translator_errors(d, model, TEST),
        }
        for d in datasets
    }
    save_module(model, out / "translator.pt", latents=meta, worlds=worlds, metrics=result)
    (out / "translator_eval.json").write_text(json.dumps(result, indent=1))
    return result


def train_predictor(
    data: LatentData,
    out: Path,
    steps: int = 20000,
    batch: int = 64,
    rollout: int = 3,
    lr: float = 3e-4,
    readout_weight: float = 1.0,
    idm_weight: float = 0.1,
    log_every: int = 1000,
    seed: int = 0,
    translator_path: Path | None = None,
    init: Path | None = None,
    extra: LatentData | None = None,
    extra_share: float = 0.5,
) -> list[dict]:
    """Each sample: start from a real latent, apply `rollout` pushes in a row, feeding back the
    predictor's own output. Losses per step: latent L1 to the true latent, the world's readout
    through the frozen translator, and inverse dynamics on pushes that moved something.

    `translator_path`: the translator to read through (default: out/translator.pt).
    `init`: start from this predictor checkpoint (fine-tuning).
    `extra`: a second dataset of the same world (e.g. the agent's own imagined pushes),
    drawn for a share `extra_share` of the batches; its validation error joins the score."""
    torch.manual_seed(seed)
    translator, _ = load_module(Translator, translator_path or out / "translator.pt", data.device)
    translator.requires_grad_(False)
    if data.world not in translator.heads:
        raise ValueError(f"the translator has no {data.world!r} head: {list(translator.heads)}")
    if extra is not None and extra.world != data.world:
        raise ValueError(f"extra data is {extra.world!r}, main data is {data.world!r}")
    predictor = Predictor(data.n_tokens, data.dim).to(data.device)
    if init is not None:
        start, _ = load_module(Predictor, init, data.device)
        predictor.load_state_dict(start.state_dict())
    idm = InverseDynamics(data.n_tokens, data.dim).to(data.device)
    params = [*predictor.parameters(), *idm.parameters()]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    scheduler = _schedule(optimizer, steps, warmup=steps // 20)
    cuda = data.device.type == "cuda"
    # fp16 autocast: T4 and P100 (Kaggle, Colab free) have no fast bf16
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if cuda else nullcontext
    sources = [(data, data.episodes(TRAIN))]
    if extra is not None:
        sources.append((extra, extra.episodes(TRAIN)))
    head, readout = data.world, data.readout
    best, history, started = math.inf, [], time.time()
    gen = torch.Generator().manual_seed(seed)

    for step in range(1, steps + 1):
        use_extra = extra is not None and torch.rand((), generator=gen).item() < extra_share
        src, episodes = sources[1] if use_extra else sources[0]
        k_steps = min(rollout, src.n_steps)
        e = episodes[torch.randint(len(episodes), (batch,), device=src.device)]
        t0 = torch.randint(src.n_steps - k_steps + 1, (batch,), device=src.device)
        z = src.latent(e, t0)
        # One tensor per key, summed out of place: `+=` on a shared zero would alias them all
        losses = {k: torch.zeros((), device=data.device) for k in ("latent", "readout", "idm")}
        with autocast():
            for k in range(k_steps):
                push, moved = src.pushes[e, t0 + k], src.moved[e, t0 + k]
                target = src.latent(e, t0 + k + 1)
                z_next = predictor(z, push)
                losses["latent"] = losses["latent"] + F.l1_loss(z_next, target) / k_steps
                losses["readout"] = losses["readout"] + (
                    readout.loss(translator(z_next, head).float(), src.poses[e, t0 + k + 1])
                    / k_steps
                )
                if k == 0 and moved.any():  # a push that missed says nothing about the push
                    guess_true = idm(z[moved], target[moved])
                    guess_pred = idm(z[moved], z_next[moved])
                    losses["idm"] = F.mse_loss(guess_true.float(), push[moved]) + F.mse_loss(
                        guess_pred.float(), push[moved]
                    )
                z = z_next
            loss = (
                losses["latent"] + readout_weight * losses["readout"] + idm_weight * losses["idm"]
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
            extra_rows = rollout_errors(extra, predictor, translator, VAL) if extra else []
            predictor.train()
            # Selection score: median px + median degrees, summed over the trained horizons
            score = sum(r["px_median"] + r["deg_median"] for r in rows[1 : rollout + 1])
            score += sum(r["px_median"] + r["deg_median"] for r in extra_rows[1:])
            history.append(
                {"step": step, "score": score, **{k: float(v.detach()) for k, v in losses.items()}}
            )
            mark = ""
            if score < best:
                best, mark = score, "  * saved"
                save_module(predictor, out / "predictor.pt", step=step, world=data.world)
            parts = "  ".join(f"{k} {v.item():.4f}" for k, v in losses.items())
            r1 = rows[1]
            note = f", agent pushes {extra_rows[1]['px_median']:.1f} px" if extra_rows else ""
            _log(
                step,
                steps,
                started,
                f"{parts}  val 1 push: {r1['px_median']:.1f} px, {r1['deg_median']:.1f}°{note}{mark}",
            )

    predictor, _ = load_module(Predictor, out / "predictor.pt", data.device)
    rows = rollout_errors(data, predictor, translator, TEST)
    result = {"test": rows, "history": history}
    if extra is not None:
        result["extra_test"] = rollout_errors(extra, predictor, translator, TEST)
    (out / "wm_eval.json").write_text(json.dumps(result, indent=1))
    print("\nTest set, imagined state vs true state:\n" + format_rows(rows))
    if extra is not None:
        print("\nExtra data (test):\n" + format_rows(result["extra_test"]))
    return rows


def transfer_test(
    data: LatentData, trunk_from: Path, out: Path, budgets: list[int], steps: int = 3000
) -> list[dict]:
    """Is a trained trunk worth anything to a new world? For each label budget (training
    episodes of `data`), a head on the frozen trunk of `trunk_from` against a translator
    trained from scratch on the same episodes. Errors on the full test split."""
    rows = []
    for budget in budgets:
        data.max_train = budget
        row = {"episodes": min(budget, len(data.episodes(TRAIN)))}
        for name, trunk in (("head_only", trunk_from), ("scratch", None)):
            print(f"\n== {data.world}: {name}, {row['episodes']} training episodes ==")
            result = train_translator(
                [data], out / f"{name}-{budget}", steps=steps, trunk_from=trunk, log_every=steps
            )
            test = result[data.world]["test"]
            row[name] = {"px_median": test["px_median"], "deg_median": test["deg_median"]}
        rows.append(row)
    data.max_train = None
    (out / "transfer.json").write_text(json.dumps(rows, indent=1))
    print("\nepisodes   head on frozen trunk   from scratch   (median test error)")
    for r in rows:
        h, s = r["head_only"], r["scratch"]
        print(
            f"{r['episodes']:>8}   {h['px_median']:>6.1f} px {h['deg_median']:>5.1f}°"
            f"     {s['px_median']:>6.1f} px {s['deg_median']:>5.1f}°"
        )
    return rows
