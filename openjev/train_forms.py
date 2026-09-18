"""openjev.train_forms — train the byte-level option scorer on choice JSONL.

Data format (one JSON object per line): {"context": str, "options": [str, ...],
"label": int}. Any generator that emits this shape works; cua-s1's `synth`
module is the one used for the published receipt.

    python -m openjev.train_forms train.jsonl --out runs/forms-s1 --epochs 2
    python -m openjev.train_forms train.jsonl --eval test.jsonl --out runs/forms-s1
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import forms_scorer as S

DEFAULT_CONFIG = {
    "encoder": "tinyx",
    "width": 128,
    "rank": 64,
    "context_tokens": 224,
    "option_tokens": 80,
    "layers": 2,
    "heads": 4,
    "dropout": 0.1,
}


class ChoiceData(Dataset):
    def __init__(self, path: str | Path):
        self.rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.rows.append(S.validate_example(json.loads(line)))
        if not self.rows:
            raise ValueError(f"no choice examples in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> S.ChoiceExample:
        return self.rows[i]


@torch.inference_mode()
def evaluate(model, data, collator, device, batch_size=256, shuffle_context=False) -> dict:
    model.eval()
    n = correct = 0
    nll = 0.0
    loader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch, shuffle_context=shuffle_context)
        labels = batch["labels"]
        nll += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(-1) == labels).sum())
        n += labels.numel()
    return {"n": n, "accuracy": correct / max(n, 1), "nll": nll / max(n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(prog="openjev.train_forms")
    ap.add_argument("train")
    ap.add_argument("--validation", default=None)
    ap.add_argument("--eval", dest="eval_only", default=None,
                    help="skip training; evaluate an existing --out checkpoint on this file")
    ap.add_argument("--shuffle-context", action="store_true",
                    help="with --eval, scores options against a ROLLED context (control)")
    ap.add_argument("--out", default="runs/forms-s1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = S.select_device(args.device)

    if args.eval_only:
        model, collator, cfg = S.load_checkpoint(args.out, device)
        res = evaluate(model, ChoiceData(args.eval_only), collator, device,
                       shuffle_context=args.shuffle_context)
        res["control"] = "shuffled_context" if args.shuffle_context else "none"
        print(json.dumps(res, sort_keys=True))
        return

    data = ChoiceData(args.train)
    val = ChoiceData(args.validation) if args.validation else None
    model, collator = S.make_system(DEFAULT_CONFIG, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, drop_last=True)
    print(f"params={S.parameter_count(model)} rows={len(data)} "
          f"steps/epoch={len(loader)} device={device}")

    started = time.perf_counter()
    best_nll, best_state = float("inf"), None
    for epoch in range(args.epochs):
        total, count = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = F.cross_entropy(model(batch), batch["labels"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * batch["labels"].numel()
            count += batch["labels"].numel()
        rec = {"epoch": epoch + 1, "train_nll": round(total / count, 5)}
        if val:
            vres = evaluate(model, val, collator, device)
            rec.update(val_accuracy=round(vres["accuracy"], 5), val_nll=round(vres["nll"], 5))
            if vres["nll"] < best_nll:
                best_nll, best_state = vres["nll"], S.trainable_state(model)
        print(json.dumps(rec, sort_keys=True), flush=True)
        model.train()

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    seconds = time.perf_counter() - started
    wp, cp = S.save_checkpoint(
        args.out, model, DEFAULT_CONFIG,
        {"epochs": args.epochs, "rows": len(data), "seconds": round(seconds, 1),
         "seed": args.seed, "best_val_nll": best_nll if val else None})
    print(f"saved {wp} + {cp} ({seconds:.0f}s)")


if __name__ == "__main__":
    main()
