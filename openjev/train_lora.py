"""openjev/train_lora.py — LoRA SFT of a small instruct model on Jev-style typed decisions (CPU).

Manual LoRA (no peft/Trainer dependency): wraps q/k/v/o projections, trains only the low-rank
deltas, masks prompt tokens out of the loss, samples length-bucketed batches (padding waste is
pure waste on an op-overhead-bound CPU), logs a JSONL ledger, checkpoints + resumable, and stops
cleanly on a wall-clock budget.

usage:
  python train_lora.py --smoke
  python train_lora.py --steps 400 --batch 8 --accum 1 --max-minutes 240 --run-name run1
  python train_lora.py --run-name run1 --resume openjev/runs/run1/adapter.pt --steps 400
"""
from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
RUNS = HERE / "runs"

DEFAULT_MODEL = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.scale = r, alpha / r
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.B(self.drop(self.A(x))) * self.scale


def apply_lora(model, r: int, alpha: int, dropout: float = 0.05) -> int:
    # freeze EVERY base parameter first (only the low-rank deltas stay trainable)
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name in TARGET_MODULES and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha, dropout))
                n += 1
    return n


def load_rows(path: Path, tok, max_len: int):
    rows = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        p = tok(rec["prompt"], add_special_tokens=False)["input_ids"]
        t = tok(rec["target"], add_special_tokens=False)["input_ids"]
        ids = (p + t)[:max_len]
        rows.append({"id": rec.get("id"), "ids": ids, "pmask": min(len(p), len(ids))})
    return rows


def make_batch(rows, pad_id):
    L = max(len(r["ids"]) for r in rows)
    input_ids, attn, labels = [], [], []
    for r in rows:
        pad = L - len(r["ids"])
        input_ids.append(r["ids"] + [pad_id] * pad)
        attn.append([1] * len(r["ids"]) + [0] * pad)
        labels.append([-100] * r["pmask"] + r["ids"][r["pmask"]:] + [-100] * pad)
    return torch.tensor(input_ids), torch.tensor(attn), torch.tensor(labels)


def buckets(rows, size, rng):
    """Length-bucketed batches: similar-length rows share a batch -> less padding."""
    order = sorted(rows, key=lambda r: len(r["ids"]))
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    rng.shuffle(chunks)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--run-name", default="run1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.train()
    n_lora = apply_lora(model, args.rank, args.alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"[load] {time.time()-t0:.1f}s | LoRA layers={n_lora} r={args.rank} alpha={args.alpha} | "
          f"trainable={n_train/1e6:.2f}M params", flush=True)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train = load_rows(DATA / "train.jsonl", tok, args.max_len)
    lens = [len(r["ids"]) for r in train]
    print(f"[data] {len(train)} rows | tokens: mean={sum(lens)/len(lens):.0f} max={max(lens)} "
          f"p90={sorted(lens)[int(len(lens)*0.9)]}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    resume_from = args.resume
    step_offset = 0
    if resume_from and Path(resume_from).exists():
        payload = torch.load(resume_from, map_location="cpu")
        sd = payload.get("model", payload)
        model.load_state_dict(sd, strict=False)
        if isinstance(payload, dict) and "opt" in payload:
            opt.load_state_dict(payload["opt"])
            step_offset = int(payload.get("step", 0))
        print(f"[resume] {resume_from} @ step {step_offset}", flush=True)

    steps = 3 if args.smoke else args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps + step_offset, 1),
                                                pct_start=0.05, anneal_strategy="cos")
    for _ in range(step_offset):
        sched.step()
    ledger = (run_dir / "train_log.jsonl").open("a")
    rng = random.Random(7)
    wall_budget = args.max_minutes * 60 if args.max_minutes else None

    losses, t_start, tok_seen = [], time.time(), 0
    stream = None
    for step in range(1, steps + 1):
        if stream is None:
            stream = buckets(train, args.batch * args.accum, rng)
        if not stream:
            stream = buckets(train, args.batch * args.accum, rng)
        batch_rows = stream.pop()
        while len(batch_rows) < args.batch * args.accum and stream:
            batch_rows += stream.pop()
        opt.zero_grad(set_to_none=True)
        step_loss, t_step, step_tokens = 0.0, time.time(), 0
        for m in range(args.accum):
            chunk = batch_rows[m * args.batch:(m + 1) * args.batch]
            if not chunk:
                continue
            input_ids, attn, labels = make_batch(chunk, pad_id)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
            (out.loss / args.accum).backward()
            step_loss += out.loss.item() / args.accum
            step_tokens += int((labels != -100).sum())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if not args.smoke:
            sched.step()
        tok_seen += step_tokens
        dt = time.time() - t_step
        losses.append(step_loss)
        rec = {"step": step + step_offset, "loss": round(step_loss, 4),
               "lr": f"{sched.get_last_lr()[0]:.2e}", "step_s": round(dt, 1),
               "tokens": step_tokens, "tok_per_s": round(step_tokens / dt, 1),
               "elapsed_s": round(time.time() - t_start, 1)}
        ledger.write(json.dumps(rec) + "\n"); ledger.flush()
        print(f"[step {step}/{steps}] loss={step_loss:.4f} {dt:.1f}s {rec['tok_per_s']} tok/s "
              f"lr={rec['lr']} elapsed={rec['elapsed_s']}s", flush=True)

        if not args.smoke and (step % args.ckpt_every == 0 or step == steps):
            state = {k: v for k, v in model.state_dict().items() if k.endswith(("A.weight", "B.weight"))}
            torch.save({"model": state, "opt": opt.state_dict(), "step": step + step_offset}, run_dir / "adapter.pt")
        if wall_budget and (time.time() - t_start) > wall_budget:
            print(f"[budget] wall budget {args.max_minutes} min reached at step {step}; stopping cleanly", flush=True)
            break

    if args.smoke:
        print(f"[smoke] ~{losses and rec['step_s']:.1f}s/step at batch={args.batch}x{args.accum} "
              f"-> {args.steps} steps ~= {args.steps * rec['step_s'] / 60:.0f} min", flush=True)
        return

    state = {k: v for k, v in model.state_dict().items() if k.endswith(("A.weight", "B.weight"))}
    torch.save({"model": state, "opt": opt.state_dict(), "step": step_offset + len(losses)}, run_dir / "adapter.pt")
    cfg = {"model": args.model, "rank": args.rank, "alpha": args.alpha, "steps_run": len(losses),
           "batch": args.batch, "accum": args.accum, "lr": args.lr, "max_len": args.max_len,
           "trainable_params": n_train, "target_modules": list(TARGET_MODULES),
           "final_loss": losses[-1], "mean_loss_last10": sum(losses[-10:]) / min(10, len(losses)),
           "tokens_seen": tok_seen, "tok_per_s_avg": round(tok_seen / (time.time() - t_start), 1),
           "wall_s": round(time.time() - t_start, 1), "data": "openjev/data/train.jsonl",
           "train_rows": len(train)}
    (run_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    print(f"[save] {run_dir/'adapter.pt'} | final loss {losses[-1]:.4f} | "
          f"{tok_seen/1000:.0f}k tokens in {cfg['wall_s']/60:.1f} min ({cfg['tok_per_s_avg']} tok/s)", flush=True)

    dev = load_rows(DATA / "dev.jsonl", tok, args.max_len)
    model.eval()
    with torch.no_grad():
        tot, n_tok = 0.0, 0
        for i in range(0, min(64, len(dev)), args.batch):
            chunk = dev[i:i + args.batch]
            input_ids, attn, labels = make_batch(chunk, pad_id)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
            n = int((labels != -100).sum())
            tot += out.loss.item() * n; n_tok += n
    print(f"[dev] target-token loss={tot/n_tok:.4f} ppl={math.exp(tot/n_tok):.2f} over {n_tok} tokens", flush=True)


if __name__ == "__main__":
    main()
