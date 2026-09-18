"""openjev/dataset.py — build the final openjev training corpus.

Sources
  1. synthetic postings, teacher-labelled AND Jev-verified  (openjev/synth/data/jev_checked.jsonl)
  2. the real typesafe-lab leads corpus with Jev's answers  (typesafe-lab/runs/leads_corpus_full.json)

Rules
  * the 70 hand-labelled gold postings are NEVER trained on (they are the eval set)
  * synthetic rows are kept only where the teacher and Jev agree on the bucket (clean, cross-verified)
  * prompts use the COMPACT format — this box is op-overhead bound, every token costs wall-clock
  * target = Jev's own typed answers (the thing we are cloning), same JSON shape as inference expects
"""
from __future__ import annotations

import argparse, json, os, random
from collections import Counter
from pathlib import Path

# This file is imported by tools that only need prompt_for/target_for (eval harnesses,
# corpus verifier), so nothing here may touch the filesystem at import time.
ROOT = Path(os.environ.get("OPENJEV_ROOT", Path(__file__).resolve().parent.parent))
LAB = Path(os.environ.get("TYPESAFE_LAB", "/home/decrux/Code/typesafe-lab"))
OUT = ROOT / "openjev" / "data"

BOOLS = ["technical_need", "business_buyer", "small_firm_doable", "pay_stated", "evergreen_repost"]
KEYS = "bucket(service_lead|staff_role|generic_job|junk) " + " ".join(BOOLS) + " fit(0-4)"


def clip(s, n=90):
    return " ".join(str(s or "").split())[:n]


def prompt_for(title, employer, pay):
    return ("<|im_start|>system\nDecide.<|im_end|>\n"
            "<|im_start|>user\n"
            f"title: {clip(title)}\nemployer: {clip(employer)}\npay: {clip(pay, 40)}\n{KEYS}<|im_end|>\n"
            "<|im_start|>assistant\n")


def target_for(answers):
    obj = {"bucket": answers["bucket"]}
    for b in BOOLS:
        obj[b] = bool(answers[b])
    obj["fit"] = int(round(float(answers["fit"])))
    return json.dumps(obj, separators=(", ", ": ")) + "<|im_end|>\n"


def corpus_answers(row):
    """Jev's answers as stored in the leads corpus row."""
    a = row.get("answers") or {}
    bucket = a.get("bucket")
    if isinstance(bucket, dict):
        bucket = bucket.get("choice")
    out = {"bucket": bucket}
    for b in BOOLS:
        v = a.get(b)
        if isinstance(v, dict):
            v = v.get("noul", v.get("probability"))
        out[b] = float(v) >= 0.5
    fit = a.get("fit")
    if isinstance(fit, dict):
        fit = fit.get("score")
    out["fit"] = float(fit)
    return out


def synth_answers(row):
    """Jev's answers inside a jev_checked synthetic row (raw API shape)."""
    jev = row.get("jev") or {}
    out = {"bucket": (jev.get("bucket") or {}).get("choice")}
    for b in BOOLS:
        v = jev.get(b) or {}
        p = v.get("noul", v.get("probability"))
        out[b] = None if not isinstance(p, (int, float)) else float(p) >= 0.5
    fit = (jev.get("fit") or {}).get("score")
    out["fit"] = None if fit is None else float(fit)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default=str(ROOT / "openjev" / "synth" / "data" / "jev_checked.jsonl"))
    ap.add_argument("--corpus", default=str(LAB / "runs" / "leads_corpus_full.json"))
    ap.add_argument("--gold", default=str(LAB / "data" / "leads.gold.json"))
    ap.add_argument("--max-generic-share", type=float, default=0.42)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gold = json.load(open(args.gold))["labels"]
    gold_ids = set(gold)
    rows, stats = [], Counter()

    # ---- 1. synthetic (kept only when teacher == Jev on the bucket)
    spath = Path(args.synth)
    if spath.exists():
        for line in spath.read_text().splitlines():
            r = json.loads(line)
            ans = synth_answers(r)
            if ans["bucket"] != r["label"]["bucket"]:
                stats["synth_dropped_mismatch"] += 1
                continue
            if ans["bucket"] not in ("service_lead", "staff_role", "generic_job", "junk") \
                    or ans["fit"] is None or any(ans[b] is None for b in BOOLS):
                stats["synth_dropped_incomplete"] += 1
                continue
            rows.append({"src": "synth", "family": r.get("family"), "intent": r["label"]["bucket"],
                         "prompt": prompt_for(r["title"], r["employer"], r.get("pay") or ""),
                         "target": target_for(ans), "bucket": ans["bucket"]})
            stats["synth_kept"] += 1
    else:
        print(f"!! no synthetic file at {spath}")

    # ---- 2. real corpus minus the gold eval rows, with a cap on generic_job
    corpus = json.load(open(args.corpus))["rows"]
    real_nongeneric, real_generic, held_out = [], [], []
    for r in corpus:
        if r["id"] in gold_ids:
            held_out.append(r)
            continue
        try:
            ans = corpus_answers(r)
        except Exception:  # noqa: BLE001
            stats["real_dropped_parse"] += 1
            continue
        rec = {"src": "real", "id": r["id"], "legacy": r.get("legacy_category"),
               "prompt": prompt_for(r.get("title", ""), r.get("employer", ""), r.get("pay") or ""),
               "target": target_for(ans), "bucket": ans["bucket"]}
        (real_generic if ans["bucket"] == "generic_job" else real_nongeneric).append(rec)
    n_keep = int(len(real_nongeneric) * args.max_generic_share / (1 - args.max_generic_share))
    rng.shuffle(real_generic)
    rows += real_nongeneric + real_generic[:n_keep]
    stats["real_nongeneric"] = len(real_nongeneric)
    stats["real_generic_kept"] = min(n_keep, len(real_generic))
    stats["real_generic_total"] = len(real_generic)

    rng.shuffle(rows)
    dev_n = max(80, len(rows) // 12)
    dev, train = rows[:dev_n], rows[dev_n:]

    def dump(path, data):
        with Path(path).open("w") as fh:
            for rec in data:
                fh.write(json.dumps(rec) + "\n")
        print(f"{Path(path).name}: {len(data)} rows")

    dump(OUT / "train.jsonl", train)
    dump(OUT / "dev.jsonl", dev)

    ev = []
    for r in held_out:
        try:
            ans = corpus_answers(r)
        except Exception:  # noqa: BLE001
            continue
        ev.append({"id": r["id"], "title": r.get("title"), "gold": gold[r["id"]], "jev_bucket": ans["bucket"],
                   "prompt": prompt_for(r.get("title", ""), r.get("employer", ""), r.get("pay") or ""),
                   "target": target_for(ans)})
    dump(OUT / "eval_gold.jsonl", ev)

    print("\nstats:", dict(stats))
    print("train mix:", dict(Counter(r["bucket"] for r in train)))
    print("dev   mix:", dict(Counter(r["bucket"] for r in dev)))
    for name, data in (("train", train), ("dev", dev)):
        chars = sum(len(r["prompt"]) + len(r["target"]) for r in data)
        print(f"  {name}: avg {chars/len(data):.0f} chars/row (~{chars/len(data)/3.4:.0f} tokens)")
    manifest = {"train": len(train), "dev": len(dev), "eval_gold": len(ev), "stats": dict(stats),
                "prompt_format": "compact", "keys": KEYS}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("\n--- sample ---")
    print(train[0]["prompt"] + train[0]["target"])


if __name__ == "__main__":
    main()
