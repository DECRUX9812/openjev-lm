"""openjev/build_corpus.py -- assemble the final training corpus (parent-owned).

Merges three Jev-labelled sources into one balanced training set:
  * openjev/data/train.jsonl        real SaskJobs postings, Jev-labelled (2,361 rows)
  * openjev/data/synth_core.jsonl   balanced synthetic postings (S1)
  * openjev/data/synth_hard.jsonl   adversarial hard cases (S2)

Why a builder and not just `cat`: the real corpus is 98% generic_job (2,326) with 35
staff_role, 0 service_lead and 0 junk. Training on it teaches "everything is generic_job".
The builder caps the majority class, keeps every minority-class row, dedups by
(title, employer) across sources, and hard-fails on any collision with the 70 held-out
gold rows.

Outputs (overwriting the originals, with the previous versions kept as *.v1.bak):
  data/train.jsonl     balanced training rows
  data/dev.jsonl       held-out dev slice (real + synthetic), never trained
  data/eval_gold.jsonl untouched (the 70 hand-labelled rows -- the measuring stick)
  runs/corpus_report.json + runs/CORPUS.md  the recipe and what actually went in

usage:  python openjev/build_corpus.py [--cap-generic 900] [--dev-real 150] [--dev-synth 100]
"""
from __future__ import annotations

import argparse, json, random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNS = HERE / "runs"
GOLD = Path("/home/decrux/Code/typesafe-lab/data/leads.gold.json")
NOULS = ["technical_need", "business_buyer", "small_firm_doable", "pay_stated", "evergreen_repost"]


def norm(s: str) -> str:
    return " ".join(str(s or "").lower().split())


def key(rec: dict) -> tuple[str, str]:
    """(title, employer) identity used for dedup / leak checks."""
    body = rec.get("prompt", "")
    title = employer = ""
    for line in body.splitlines():
        if line.startswith("title: "):
            title = line[7:]
        elif line.startswith("employer: "):
            employer = line[10:]
    return norm(title), norm(employer)


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (missing, skipped) {path.name}")
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    print(f"  {path.name}: {len(out)} rows")
    return out


def gold_keys() -> set[tuple[str, str]]:
    labels = json.loads(GOLD.read_text())["labels"]
    sample = Path("/home/decrux/Code/typesafe-lab/data/leads.sample.jsonl")
    keys = set()
    for line in sample.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("id")) in {str(k) for k in labels}:
            keys.add((norm(row.get("title")), norm(row.get("employer"))))
    return keys


def bucket_of(rec: dict) -> str:
    try:
        return json.loads(rec["target"].replace("<|im_end|>", "").strip())["bucket"]
    except Exception:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-generic", type=int, default=900,
                    help="max real generic_job rows to keep (majority-class cap)")
    ap.add_argument("--dev-real", type=int, default=150)
    ap.add_argument("--dev-synth", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    print("sources:")
    real = load(DATA / "train.jsonl")
    real_dev_old = load(DATA / "dev.jsonl")
    core = load(DATA / "synth_core.jsonl")
    hard = load(DATA / "synth_hard.jsonl")
    gkeys = gold_keys()
    print(f"gold rows: {len(gkeys)} (title, employer) keys held out")

    rng = random.Random(args.seed)
    seen: dict[tuple[str, str], str] = {}
    dropped = Counter()
    kept: list[dict] = []

    def add(rec: dict, source: str, *, is_dev_candidate: bool = False) -> None:
        k = key(rec)
        if k in gkeys:
            dropped[f"gold_leak:{source}"] += 1
            return
        if k != ("", "") and k in seen:
            dropped[f"dupe_of_{seen[k]}:{source}"] += 1
            return
        seen[k] = source
        rec["source"] = source
        kept.append(rec)

    # 1) synthetic rows first: they are the balanced, boundary-teaching data
    for rec in core:
        add(rec, "synth_core")
    for rec in hard:
        add(rec, "synth_hard")

    # 2) real rows, majority class capped
    pool = real + real_dev_old
    rng.shuffle(pool)
    generic_kept = 0
    for rec in pool:
        if bucket_of(rec) == "generic_job":
            if generic_kept >= args.cap_generic:
                dropped["cap:real_generic_job"] += 1
                continue
            generic_kept += 1
        add(rec, "real")

    # 3) dev slice: held out of training, drawn from both worlds
    synth_rows = [r for r in kept if r["source"].startswith("synth")]
    real_rows = [r for r in kept if r["source"] == "real"]
    rng.shuffle(synth_rows)
    rng.shuffle(real_rows)
    dev = synth_rows[: args.dev_synth] + real_rows[: args.dev_real]
    dev_ids = {id(r) for r in dev}
    train = [r for r in kept if id(r) not in dev_ids]

    def dump(name: str, rows: list[dict]) -> None:
        path = DATA / f"{name}.jsonl"
        if path.exists():
            path.with_suffix(".v1.bak").write_bytes(path.read_bytes())
        with path.open("w") as fh:
            for rec in rows:
                fh.write(json.dumps({k: v for k, v in rec.items() if k != "source"}) + "\n")

    dump("train", train)
    dump("dev", dev)

    dist = lambda rows: dict(Counter(bucket_of(r) for r in rows))  # noqa: E731
    report = {
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_bucket_dist": dist(train),
        "dev_bucket_dist": dist(dev),
        "by_source": dict(Counter(r["source"] for r in train)),
        "dropped": dict(dropped),
        "dedup_policy": "first source wins: synth_core > synth_hard > real",
        "gold_rows_held_out": len(gkeys),
        "gold_leaks_in_train": sum(1 for r in train if key(r) in gkeys),
        "real_generic_cap": args.cap_generic,
    }
    (RUNS / "corpus_report.json").write_text(json.dumps(report, indent=1))
    lines = ["# openjev corpus recipe", "", "```json", json.dumps(report, indent=1), "```", ""]
    lines += ["| bucket | train | dev |", "|---|---|---|",
              *[f"| {b} | {report['train_bucket_dist'].get(b,0)} | {report['dev_bucket_dist'].get(b,0)} |"
                for b in ["generic_job", "staff_role", "service_lead", "junk"]]]
    (RUNS / "CORPUS.md").write_text("\n".join(lines) + "\n")
    print("\n" + json.dumps(report, indent=1))
    assert report["gold_leaks_in_train"] == 0, "gold leakage!"
    print(f"\nOK -> {DATA/'train.jsonl'} ({len(train)}) | {DATA/'dev.jsonl'} ({len(dev)})")


if __name__ == "__main__":
    main()
