#!/usr/bin/env python3
"""Build the published receipts: verify/predictions/*.jsonl and verify/live/*.jsonl.

Every file this writes is a per-row receipt for a number in the README/paper. Run it with the
venv that has fastembed if you want the classifier arm regenerated too:

    ~/.venvs/embed/bin/python verify/build_receipts.py

Sources (all on the machine that ran the work; paths documented so receipts can be traced):
  runs/eval_*.json                  - harness outputs (this repo)
  data/eval_gold.jsonl              - the 70 hand-labelled rows
  ~/Code/jev-repro-test/runs_local/leads_local_asis.jsonl   - viral artifact per-row outputs
  ~/Code/jev-mission/live-ab/*.json - live-traffic A/B receipts (classifier vs hosted Jev)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRED = HERE / "predictions"
LIVE = HERE / "live"
PRED.mkdir(exist_ok=True)
LIVE.mkdir(exist_ok=True)

RUNS = ROOT / "runs"
GOLD = ROOT / "data" / "eval_gold.jsonl"
LAB = Path.home() / "Code" / "jev-repro-test"
LIVEAB = Path.home() / "Code" / "jev-mission" / "live-ab"
CLASSIFIER = Path.home() / "Code" / "openjev"

gold_ids = [json.loads(l)["id"] for l in GOLD.read_text().splitlines() if l.strip()]
gold_set = set(gold_ids)


def w(path, rows):
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    print(f"{path.relative_to(ROOT)}  {len(rows)} rows")


def pred_of(row):
    for k in ("pred_bucket", "pred"):
        if k in row:
            return row[k]
    if isinstance(row.get("staged"), dict):
        return row["staged"].get("bucket")
    return None


# --- harness outputs -------------------------------------------------------
for src, dst in [
    ("eval_A_run-final.json", "lm_A_step386.jsonl"),
    ("eval_B_run-final.json", "lm_B_step400.jsonl"),
    ("eval_baseline.json", "untrained_0.5b.jsonl"),
    ("eval_snap0405.json", "night1_snap0405.jsonl"),
    ("eval_lora_final_420.json", "night1_final420.jsonl"),
]:
    d = json.loads((RUNS / src).read_text())
    rows = [{"id": r["id"], "pred": pred_of(r), "gold": r.get("gold"), "conf": r.get("bucket_conf")}
            for r in d["rows"]]
    w(PRED / dst, rows)

# --- Jev's own answers as the reference arm --------------------------------
rows = []
for line in GOLD.read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        rows.append({"id": r["id"], "pred": r["jev_bucket"], "gold": r["gold"], "title": r.get("title", "")})
w(PRED / "jev_reference.jsonl", rows)

# --- viral artifact (as-is decoding) ---------------------------------------
viral = []
for line in (LAB / "runs_local" / "leads_local_asis.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("id") in gold_set:
        viral.append({"id": r["id"], "pred": r.get("bucket"), "conf": r.get("bucket_confidence")})
if viral:
    w(PRED / "viral_asis.jsonl", sorted(viral, key=lambda r: r["id"]))

# --- classifier arm (regenerate from its repo if importable) ---------------
try:
    sys.path.insert(0, str(CLASSIFIER))
    from openjev.engine import OpenJev  # noqa: E402

    sample_path = Path.home() / "Code" / "typesafe-lab" / "data" / "leads.sample.jsonl"
    sample = [json.loads(l) for l in sample_path.read_text().splitlines() if l.strip()]
    want = [r for r in sample if r.get("id") in gold_set]
    jev = OpenJev()
    decisions = jev.decide(want)
    rows = []
    for r, d in zip(want, decisions):
        dd = d.to_dict() if hasattr(d, "to_dict") else d
        rows.append({"id": r["id"], "pred": dd.get("bucket"), "gold": dd.get("gold")})
    w(PRED / "classifier.jsonl", rows)
except Exception as exc:  # noqa: BLE001
    print(f"classifier.jsonl skipped: {exc}")

# --- live-traffic receipts (classifier vs hosted Jev, unseen postings) -----
def live_rows(src, dst, pred_key="classifier_pred"):
    d = json.loads((LIVEAB / src).read_text())
    rows = []
    for r in d["rows"]:
        h = r.get("hosted", {})
        l = r.get("local", {})
        rows.append({
            "id": r["id"], "title": r.get("title"), "employer": r.get("employer"),
            "hosted_bucket": h.get("bucket"), pred_key: l.get("bucket"),
            "hosted_tier": h.get("tier"), "local_tier": l.get("tier"),
            "agree": h.get("bucket") == l.get("bucket"),
        })
    w(LIVE / dst, rows)


live_rows("ab_rows.json", "fresh_106.jsonl")

# LM arm's decisions on the same 106 unseen postings (receipt from the fresh run)
lm_fresh = LIVE / "lm_fresh_106.json"
if lm_fresh.exists():
    lm = {str(r["id"]): r for r in json.loads(lm_fresh.read_text())["rows"]}
    rows = [json.loads(l) for l in (LIVE / "fresh_106.jsonl").read_text().splitlines() if l.strip()]
    for row in rows:
        hit = lm.get(str(row["id"]))
        if hit:
            row["lm_pred"] = hit.get("pred_bucket")
            row["lm_conf"] = hit.get("bucket_conf")
    w(LIVE / "fresh_106.jsonl", rows)
live_rows("hard_rows.json", "boundary_107.jsonl")

# --- dark sweep summary (2,844-row archive re-analysis) --------------------
sweep = json.loads((LIVEAB / "dark_sweep_rows.json").read_text())
if isinstance(sweep, dict):
    sweep = sweep.get("rows", [])
w(LIVE / "dark_sweep.jsonl", [{"id": r["id"], "pred": r.get("bucket"), "tier": r.get("tier"),
                                "conf": r.get("bucket_confidence")} for r in sweep])

# --- the human-readable live reports --------------------------------------
for name in ("ab_report.md", "hard_report.md", "dark_sweep_report.md", "REPORT-live-ab-2026-09-18.md"):
    src = LIVEAB / name
    if src.exists():
        (LIVE / name).write_text(src.read_text())
        print(f"verify/live/{name}")
