#!/usr/bin/env python3
"""Score open-Jev predictions against gold — dependency-free, no torch, no installs.

    python3 verify/score.py --pred verify/predictions/lm_B_step400.jsonl
    python3 verify/score.py --self-test          # re-derives every headline number

Prediction files are JSONL with at least {"id": ..., "pred": "<bucket>"}.
Gold is data/eval_gold.jsonl ({"id": ..., "gold": ...}), hand-labelled and audited
(runs/VERIFY.md). Live-traffic receipts use their own gold column (Jev's own answer).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLD = ROOT / "data" / "eval_gold.jsonl"
PRED = HERE / "predictions"
LIVE = HERE / "live"

CLASSES = ["service_lead", "staff_role", "generic_job", "junk"]


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def metrics(pred_file, gold_file=GOLD, pred_key="pred", gold_key="gold"):
    gold = {r["id"]: r[gold_key] for r in load_jsonl(gold_file)}
    rows = [(r["id"], gold.get(r["id"]), r.get(pred_key)) for r in load_jsonl(pred_file)]
    rows = [(i, g, p) for i, g, p in rows if g and p]
    hits = sum(1 for _, g, p in rows if g == p)
    per = {c: [0, 0] for c in CLASSES}
    for _, g, p in rows:
        if g in per:
            per[g][1] += 1
            if p == g:
                per[g][0] += 1
    return {"n": len(rows), "correct": hits, "acc": (hits / len(rows)) if rows else 0.0, "per_class": per}


def agreement(file_a, file_b, key_a="pred", key_b="pred"):
    a = {r["id"]: r.get(key_a) for r in load_jsonl(file_a)}
    b = {r["id"]: r.get(key_b) for r in load_jsonl(file_b)}
    common = [i for i in a if i in b and a[i] and b[i]]
    same = sum(1 for i in common if a[i] == b[i])
    return {"n": len(common), "same": same}


def fmt(m):
    per = " · ".join(f"{c} {h}/{t}" for c, (h, t) in m["per_class"].items() if t)
    return f"{m['correct']}/{m['n']} = {100 * m['acc']:.1f}%  ({per})"


CHECKS = [
    # (label, kind, args, expected)
    ("LM arm, harness B (step-400 adapter)", "gold", ("lm_B_step400.jsonl",), (65, 70)),
    ("LM arm, harness A (step-386 snapshot)", "gold", ("lm_A_step386.jsonl",), (65, 70)),
    ("night-1 final adapter (420 steps)", "gold", ("night1_final420.jsonl",), (65, 70)),
    ("night-1 snapshot (step 405)", "gold", ("night1_snap0405.jsonl",), (63, 70)),
    ("untrained 0.5B base", "gold", ("untrained_0.5b.jsonl",), (51, 70)),
    ("viral Jev repro (as-is decoding)", "gold", ("viral_asis.jsonl",), (54, 70)),
    ("classifier arm (3 MB frozen encoder)", "gold", ("classifier.jsonl",), (66, 70)),
    ("Jev (hosted) reference answers", "gold", ("jev_reference.jsonl",), (68, 70)),
    ("harness A vs harness B: identical buckets", "agree", ("lm_A_step386.jsonl", "lm_B_step400.jsonl"), (70, 70)),
    ("live: classifier vs hosted Jev, 106 fresh postings", "live", ("fresh_106.jsonl", "classifier_pred", "hosted_bucket"), (106, 106)),
    ("live: classifier vs hosted Jev, 107 boundary postings", "live", ("boundary_107.jsonl", "classifier_pred", "hosted_bucket"), (104, 107)),
]


def self_test():
    ok = True
    print(f"{'check':58} {'result':34} expected   verdict")
    print("-" * 116)
    for label, kind, args, (exp_hit, exp_n) in CHECKS:
        if kind == "gold":
            m = metrics(PRED / args[0])
            got = (m["correct"], m["n"])
            res = fmt(m)
        elif kind == "agree":
            m = agreement(PRED / args[0], PRED / args[1])
            got = (m["same"], m["n"])
            res = f"{m['same']}/{m['n']} identical"
        else:
            fname, ka, kb = args
            m = agreement(LIVE / fname, LIVE / fname, ka, kb)
            got = (m["same"], m["n"])
            res = f"{m['same']}/{m['n']} agree"
        verdict = "PASS" if got == (exp_hit, exp_n) else "FAIL"
        ok = ok and verdict == "PASS"
        print(f"{label:58} {res:34} {exp_hit}/{exp_n:<8} {verdict}")
    print("-" * 116)
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED — the published numbers do not reproduce")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="predictions jsonl")
    ap.add_argument("--gold", default=str(GOLD), help="gold jsonl (default: data/eval_gold.jsonl)")
    ap.add_argument("--self-test", action="store_true", help="re-derive every headline number")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.pred:
        ap.error("give --pred FILE or --self-test")
    m = metrics(args.pred, args.gold)
    print(f"{args.pred}: {fmt(m)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
