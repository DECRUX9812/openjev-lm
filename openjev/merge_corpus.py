"""openjev/merge_corpus.py -- build a corpus where EVERY training row is Jev-labelled.

Distinct from (and non-colliding with) openjev/dataset.py, which the other agent owns:
  * dataset.py keeps a synthetic row only when the teacher label agrees with Jev, and it
    consumes only openjev/synth/data/jev_checked.jsonl.
  * this builder keeps every synthetic row that has a real Jev answer (teacher disagreement is
    informative boundary data, not noise), consumes any number of Jev-labelled files, and
    writes to its own filenames so nothing the other agent owns is overwritten.

Every prompt/target pair is regenerated through dataset.prompt_for / dataset.target_for, so
rows from different pipelines (old verbose prompt, compact prompt, raw API answer shapes) all
end up in one consistent format.

usage:
  python openjev/merge_corpus.py                 # default inputs, writes train_jev/dev_jev
  python openjev/merge_corpus.py --cap-generic 900 --dev-frac 0.12
"""
from __future__ import annotations

import argparse, json, random, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import BOOLS, prompt_for, target_for  # noqa: E402

ROOT = HERE.parent
DATA = HERE / "data"
RUNS = HERE / "runs"
LAB = Path("/home/decrux/Code/typesafe-lab")


def _p(v):
    if isinstance(v, dict):
        for k in ("noul", "probability", "prob"):
            if k in v:
                return float(v[k])
        return float(v.get("choice", 0) or 0)
    return float(v)


def norm_answer(d: dict | None) -> dict | None:
    """Any Jev answer dialect -> {bucket, fit, 5 bools} or None if unusable."""
    if not d:
        return None
    b = d.get("bucket")
    if isinstance(b, dict):
        b = b.get("choice")
    if b not in ("service_lead", "staff_role", "generic_job", "junk"):
        return None
    out = {"bucket": b}
    for k in BOOLS:
        if k not in d or d[k] is None:
            return None
        out[k] = _p(d[k]) >= 0.5
    fit = d.get("fit")
    if isinstance(fit, dict):
        fit = fit.get("score")
    if fit is None:
        return None
    out["fit"] = int(round(min(4.0, max(0.0, float(fit)))))
    return out


def nkey(title: str, employer: str) -> tuple[str, str]:
    return (" ".join(str(title or "").lower().split()), " ".join(str(employer or "").lower().split()))


def row_from(title: str, employer: str, pay: str, ans: dict, *, source: str, ident: str,
             family: str = "", teacher: str | None = None, rank=None) -> dict:
    rec = {"id": ident, "source": source, "prompt": prompt_for(title, employer, pay or ""),
           "target": target_for(ans), "jev": ans}
    if family:
        rec["family"] = family
    if teacher:
        rec["teacher_bucket"] = teacher
    if rank is not None:
        rec["rank"] = rank
    return rec


def load_jevlab(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ans = norm_answer(r.get("jev"))
            if ans is None:
                continue
            out.append(row_from(r.get("title", ""), r.get("employer", ""), r.get("pay", ""), ans,
                                source=r.get("source") or p.stem, ident=r.get("id") or f"{p.stem}-{len(out)}",
                                family=r.get("family") or r.get("stratum", ""), rank=r.get("rank")))
    return out


def load_checked(path: Path) -> list[dict]:
    """The other agent's jev_checked.jsonl (raw API answer shapes + teacher label)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ans = norm_answer(r.get("jev"))
        if ans is None:
            continue
        out.append(row_from(r.get("title", ""), r.get("employer", ""), r.get("pay", ""), ans,
                            source="synth_checked", ident=r.get("id") or f"checked-{len(out)}",
                            family=r.get("family", ""), teacher=(r.get("label") or {}).get("bucket")))
    return out


def load_real(corpus_path: Path, gold_ids: set) -> tuple[list[dict], Counter]:
    stats = Counter()
    rows = []
    for r in json.loads(corpus_path.read_text())["rows"]:
        if r["id"] in gold_ids:
            stats["gold_held_out"] += 1
            continue
        ans = norm_answer(r.get("answers"))
        if ans is None:
            stats["real_dropped_parse"] += 1
            continue
        rows.append(row_from(r.get("title", ""), r.get("employer", ""), r.get("pay", ""), ans,
                             source="real", ident=r["id"]))
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevlab-dir", action="append", default=None,
                    help="repeatable; each dir's *.jsonl is read")
    ap.add_argument("--max-generic-share", type=float, default=0.42,
                    help="after assembly, trim generic_job rows to at most this share")
    ap.add_argument("--checked", default=str(ROOT / "openjev" / "synth" / "data" / "jev_checked.jsonl"))
    ap.add_argument("--corpus", default=str(LAB / "runs" / "leads_corpus_full.json"))
    ap.add_argument("--gold", default=str(LAB / "data" / "leads.gold.json"))
    ap.add_argument("--eval-gold", default=str(DATA / "eval_gold.jsonl"))
    ap.add_argument("--cap-generic", type=int, default=900, help="max REAL generic_job rows kept")
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--train-out", default=str(DATA / "train_jev.jsonl"))
    ap.add_argument("--dev-out", default=str(DATA / "dev_jev.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    gold_ids = set(json.loads(Path(args.gold).read_text())["labels"])
    gold_keys = set()
    for r in json.loads(Path(args.corpus).read_text())["rows"]:
        if r["id"] in gold_ids:
            gold_keys.add(nkey(r.get("title", ""), r.get("employer", "")))
    for line in Path(args.eval_gold).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            gold_ids.add(rec["id"])
            gold_keys.add(nkey(rec.get("title", ""), rec.get("employer", "")))
    print(f"held out: {len(gold_ids)} gold ids, {len(gold_keys)} gold (title, employer) pairs")

    jevlab_dirs = args.jevlab_dir or [str(DATA / "jevlab")]
    here = HERE  # noqa: F841
    synth = []
    for d in jevlab_dirs:
        p = Path(d)
        files = sorted(p.glob("*.jsonl")) if p.exists() else []
        got = load_jevlab(files)
        synth += got
        print(f"  {d}: {len(got)} rows from {len(files)} file(s)")
    checked = load_checked(Path(args.checked))
    real, rstats = load_real(Path(args.corpus), gold_ids)
    print(f"loaded: jevlab {len(synth)} | checked {len(checked)} | real {len(real)}")

    rng = random.Random(args.seed)
    seen: dict[tuple[str, str], str] = {}
    kept, dropped = [], Counter()

    def add(rows, label):
        for rec in rows:
            k = nkey(*[ln.split(": ", 1)[1] for ln in rec["prompt"].splitlines() if ln.startswith(("title: ", "employer: "))][:2])
            if k in gold_keys:
                dropped[f"gold_pair:{label}"] += 1
                continue
            if k[0] and k in seen:
                dropped[f"dupe:{label}"] += 1
                continue
            seen[k] = label
            kept.append(rec)

    # synthetic first (balanced + boundary cases), then the other agent's verified rows, then real
    add(synth, "synth")
    add(checked, "checked")
    real_non, real_gen = [], []
    for rec in real:
        (real_gen if rec["jev"]["bucket"] == "generic_job" else real_non).append(rec)
    rng.shuffle(real_gen)
    kept_gen = min(args.cap_generic, len(real_gen))
    dropped["cap:real_generic"] += len(real_gen) - kept_gen
    add(real_non + real_gen[:kept_gen], "real")

    # ---- global generic trim: never let the majority class swallow the corpus
    non = [r for r in kept if r["jev"]["bucket"] != "generic_job"]
    gen = [r for r in kept if r["jev"]["bucket"] == "generic_job"]
    target_gen = int(len(non) * args.max_generic_share / (1 - args.max_generic_share))
    if len(gen) > target_gen:
        rng.shuffle(gen)
        # keep real generic rows first (they are real postings); synthetic generics go first out
        gen.sort(key=lambda r: 0 if r["source"] == "real" else 1)
        dropped["global_generic_trim"] += len(gen) - target_gen
        kept = non + gen[:target_gen]
    print(f"generic trim: {len(gen)} generic rows -> {min(len(gen), target_gen)} (non-generic {len(non)})")

    # ---- stratified dev split: every bucket represented in dev
    by_bucket: dict[str, list] = {}
    for r in kept:
        by_bucket.setdefault(r["jev"]["bucket"], []).append(r)
    dev, train = [], []
    for b, rows in by_bucket.items():
        rng.shuffle(rows)
        n = max(8, int(len(rows) * args.dev_frac))
        dev += rows[:n]
        train += rows[n:]
    rng.shuffle(dev)
    rng.shuffle(train)
    # never let a gold posting appear in either split
    def clean(rows):
        return [r for r in rows if r["id"] not in gold_ids]
    train, dev = clean(train), clean(dev)

    for path, rows in ((Path(args.train_out), train), (Path(args.dev_out), dev)):
        if path.exists():
            path.with_suffix(".bak").write_bytes(path.read_bytes())
        with path.open("w") as fh:
            for rec in rows:
                fh.write(json.dumps(rec) + "\n")

    dist = lambda rows: dict(Counter(r["jev"]["bucket"] for r in rows))  # noqa: E731
    report = {"train_rows": len(train), "dev_rows": len(dev), "train_dist": dist(train),
              "dev_dist": dist(dev), "by_source": dict(Counter(r["source"] for r in train)),
              "dropped": dict(dropped), "real_stats": dict(rstats),
              "real_generic_cap": args.cap_generic, "gold_held_out": len(gold_ids),
              "gold_leaks": sum(1 for r in train + dev if r["id"] in gold_ids)}
    (RUNS / "corpus_jev_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    assert report["gold_leaks"] == 0, "gold leakage in train/dev"


if __name__ == "__main__":
    main()
