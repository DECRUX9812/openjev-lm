"""openjev/train_v2.py -- memory-safe, resumable CPU LoRA trainer (parent-owned).

Hardened after the 02:35 host OOM (8/8 GB swap; one oversized python process killed every
Hermes child on the box):

  * default batch 6 / max-len 192 keeps activations ~1 GB instead of ~2 GB
  * checkpoints every 5 steps AND saves the AdamW state, so a kill costs at most 5 steps
  * --time-budget-min exits cleanly at a wall-clock budget; run_training.sh relaunches with
    --resume until the step target is reached (small resumable steps that land on disk)
  * --max-rss-gb aborts the step loop and checkpoints if RSS crosses the cap (OOM guard)
  * exit code 10 = "more steps wanted", so the supervisor loops instead of guessing
  * model-heavy work runs under `flock /tmp/jev-model.lock` (see run_training.sh) so this box
    never holds two models at once

LoRA layout comes from train_lora.py so adapters stay format-compatible with the eval
harnesses: LoRALinear around q/k/v/o_proj, r=16, alpha=32, delta = B(A(x)) * (alpha/r).

usage:
  python openjev/train_v2.py --run-name openjev-v2 --steps 200 --time-budget-min 25
  python openjev/train_v2.py --run-name openjev-v2 --steps 200 --resume
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_lora import apply_lora, load_rows, make_batch  # noqa: E402

ROOT = HERE.parent
DATA = HERE / "data"
RUNS = HERE / "runs"
DEFAULT_MODEL = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct")


def rss_gb() -> float:
    """Resident set size of this process in GB (Linux)."""
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 1e9
    except Exception:
        return 0.0


def save(run_dir: Path, model, opt, step: int, extra: dict) -> None:
    lora = {k: v for k, v in model.state_dict().items() if "A.weight" in k or "B.weight" in k}
    torch.save({"format": 2, "lora": lora, "opt": opt.state_dict(), "step": step},
               run_dir / "adapter.pt")
    (run_dir / "progress.json").write_text(json.dumps({"step": step, **extra}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--run-name", default="openjev-v2")
    ap.add_argument("--steps", type=int, default=200, help="steps for THIS invocation")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also keep adapter_step<N>.pt every N steps (for a learning curve)")
    ap.add_argument("--time-budget-min", type=float, default=0.0, help="0 = no limit")
    ap.add_argument("--max-rss-gb", type=float, default=7.0, help="abort + checkpoint above this")
    ap.add_argument("--target-total", type=int, default=0, help="absolute step target (LR schedule length)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--train-file", default=str(DATA / "train.jsonl"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt, prog = run_dir / "adapter.pt", run_dir / "progress.json"
    ledger = (run_dir / "train_log.jsonl").open("a")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.train()
    n_lora = apply_lora(model, args.rank, args.alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    if n_train > 50e6:      # 0.5B base + LoRA should be ~2.2M; anything larger means the freeze is gone
        sys.exit(f"REFUSING TO TRAIN: {n_train/1e6:.0f}M trainable params -- apply_lora() did not "
                 f"freeze the base weights (this is what OOM-killed the host at 02:35)")
    print(f"[load] {time.time()-t0:.1f}s | LoRA layers={n_lora} r={args.rank} | "
          f"trainable={n_train/1e6:.2f}M | rss={rss_gb():.2f}GB", flush=True)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train = load_rows(Path(args.train_file), tok, args.max_len)
    lens = sorted(len(r["ids"]) for r in train)
    print(f"[data] {len(train)} rows | mean tokens {sum(len(r['ids']) for r in train)/len(train):.0f} "
          f"p95={lens[int(.95*len(lens))]} cap={args.max_len} | {Path(args.train_file).name}", flush=True)

    start_step = 0
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    if args.resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and state.get("format") == 2:
            model.load_state_dict(state["lora"], strict=False)
            opt.load_state_dict(state["opt"])
            start_step = int(state["step"])
        elif isinstance(state, dict) and "model" in state:      # train_lora.py checkpoint shape
            model.load_state_dict(state["model"], strict=False)
            if state.get("opt"):
                try:
                    opt.load_state_dict(state["opt"])
                except Exception:  # noqa: BLE001  (optimizer state is a bonus, not required)
                    pass
            start_step = int(state.get("step") or (json.loads(prog.read_text()).get("step", 0) if prog.exists() else 0))
        else:                                     # legacy flat LoRA-only checkpoint
            model.load_state_dict(state, strict=False)
            start_step = json.loads(prog.read_text()).get("step", 0) if prog.exists() else 0
        print(f"[resume] step {start_step} | rss={rss_gb():.2f}GB", flush=True)

    total = args.target_total or (start_step + args.steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(total, 2),
                                                pct_start=0.05, anneal_strategy="cos")
    for _ in range(min(start_step, total - 1)):   # fast-forward the schedule on resume
        sched.step()

    rng = torch.Generator().manual_seed(args.seed + start_step)
    perm = torch.randperm(len(train), generator=rng).tolist()
    cursor, step = 0, start_step
    losses, t_start = [], time.time()
    deadline = t_start + args.time_budget_min * 60 if args.time_budget_min else None
    stop_reason = "target-steps"

    while step < total:
        if cursor + args.batch > len(perm):
            perm = torch.randperm(len(train), generator=rng).tolist(); cursor = 0
        batch_rows = [train[i] for i in perm[cursor:cursor + args.batch]]
        cursor += args.batch
        step += 1

        opt.zero_grad(set_to_none=True)
        t_step = time.time()
        input_ids, attn, labels = make_batch(batch_rows, pad_id)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        dt = time.time() - t_step
        losses.append(out.loss.item())
        mem = rss_gb()

        ledger.write(json.dumps({"step": step, "loss": round(out.loss.item(), 4),
                                 "lr": f"{sched.get_last_lr()[0]:.2e}", "step_s": round(dt, 1),
                                 "tokens": int(attn.sum()), "rss_gb": round(mem, 2),
                                 "elapsed_min": round((time.time()-t_start)/60, 1)}) + "\n")
        ledger.flush()
        print(f"[step {step}/{total}] loss={out.loss.item():.4f} {dt:.0f}s rss={mem:.2f}GB "
              f"eta={(total-step)*dt/60:.0f}min", flush=True)

        if step % args.save_every == 0 or step == total:
            save(run_dir, model, opt, step, {"loss": out.loss.item(), "train_file": args.train_file,
                                             "batch": args.batch, "max_len": args.max_len,
                                             "rank": args.rank, "target_total": total,
                                             "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            print(f"  [save] step {step}", flush=True)
            if args.snapshot_every and step % args.snapshot_every == 0:
                import shutil
                shutil.copy(ckpt, run_dir / f"adapter_step{step}.pt")
                print(f"  [snapshot] adapter_step{step}.pt", flush=True)
        if mem > args.max_rss_gb:
            stop_reason = "rss-guard"
            save(run_dir, model, opt, step, {"loss": out.loss.item(), "train_file": args.train_file,
                                             "stopped": "rss-guard", "batch": args.batch})
            print(f"[guard] rss {mem:.2f}GB > {args.max_rss_gb}GB -- checkpointed at step {step}", flush=True)
            break
        if deadline and time.time() > deadline and step < total:
            stop_reason = "time-budget"
            save(run_dir, model, opt, step, {"loss": out.loss.item(), "train_file": args.train_file,
                                             "stopped": "time-budget", "batch": args.batch})
            print(f"[budget] hit {args.time_budget_min}min at step {step}/{total} -- checkpointed", flush=True)
            break

    cfg = {"model": args.model, "run_name": args.run_name, "steps_done": step, "target_total": total,
           "batch": args.batch, "lr": args.lr, "rank": args.rank, "alpha": args.alpha,
           "max_len": args.max_len, "trainable_params": n_train, "train_file": args.train_file,
           "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
           "mean_loss_last10": (sum(losses[-10:]) / min(10, len(losses))) if losses else None,
           "wall_min": round((time.time()-t_start)/60, 1), "stop_reason": stop_reason}
    (run_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    print(f"[chunk-done] {json.dumps(cfg)}", flush=True)
    sys.exit(0 if step >= total else 10)          # 10 = more steps wanted -> supervisor loops


if __name__ == "__main__":
    main()
