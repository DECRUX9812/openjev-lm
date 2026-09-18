"""openjev/label_jev.py -- put Jev's own answers on any posting JSONL (parent-owned).

Why this exists: synthetic postings in openjev/synth/ carry a TEACHER's label (a Claude
rubric guess) and are only spot-verified with Jev. That is fine for generation QA but wrong
for training: this experiment measures whether a small model can reproduce JEV'S judgment, so
every training row has to be labelled by Jev himself. This script takes any JSONL whose rows
carry title/employer/pay and returns the same rows with a normalized `jev` block, then builds
the training-format prompt/target FROM JEV'S ANSWERS.

Idempotent and resumable: rows that already carry a usable Jev answer are passed through
without an API call (answers already in the shared cache cost nothing either).

usage:
  python openjev/label_jev.py --in openjev/synth/data/labelled.jsonl \
                              --in openjev/synth_hard.jsonl \
                              --out-dir openjev/data/jevlab
  python openjev/label_jev.py --in-file ... --workers 8 --limit 100

output: <out-dir>/<input-stem>.jev.jsonl with rows
  {id, title, employer, pay, family|stratum, source, jev:{bucket,fit,5 bools,confidence,probs,raw},
   prompt, target}   <- prompt/target come from dataset.prompt_for/target_for
and prints per-file counts, bucket distribution, teacher agreement, and Jev spend.
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/decrux/Code/typesafe-lab")

from dataset import BOOLS, prompt_for, target_for            # noqa: E402
from typesafe_lab.cache import JsonCache                     # noqa: E402
from typesafe_lab.client import SystemOneClient              # noqa: E402
from typesafe_lab.leads import LEAD_QUESTIONS, lead_state    # noqa: E402

CACHE_PATH = HERE / "data" / "cache_jev_labels.json"


def _prob(v):
    """Accept a raw noul answer, a plain float, or a dict with several spellings."""
    if isinstance(v, dict):
        for k in ("noul", "probability", "prob"):
            if k in v:
                return float(v[k])
        return float(v.get("choice", 0) or 0)
    return float(v)


def normalize_jev(jev: dict) -> dict | None:
    """Turn any of the Jev answer dialects into {bucket, fit, bools, confidence, probs}."""
    if not jev:
        return None
    b = jev.get("bucket")
    if isinstance(b, dict):
        bucket, conf = b.get("choice"), b.get("confidence")
        probs = b.get("probabilities") or {}
    else:
        bucket, conf, probs = b, jev.get("confidence"), jev.get("probabilities") or {}
    if bucket not in ("service_lead", "staff_role", "generic_job", "junk"):
        return None
    out = {"bucket": bucket, "confidence": conf, "probs": probs}
    for k in BOOLS:
        out[k] = _prob(jev.get(k, 0.0)) >= 0.5
        out[k + "_p"] = round(_prob(jev.get(k, 0.0)), 4)
    fit = jev.get("fit")
    if isinstance(fit, dict):
        fit = fit.get("score", fit.get("choice", 0))
    out["fit"] = float(fit or 0)
    return out


def jev_block(client: SystemOneClient, row: dict) -> dict:
    resp = client.ask(lead_state(row), LEAD_QUESTIONS)
    return {k: (a.raw if hasattr(a, "raw") else a) for k, a in resp.answers.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="JSONL to label")
    ap.add_argument("--out-dir", default=str(HERE / "data" / "jevlab"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    client = SystemOneClient(cache=JsonCache(CACHE_PATH))
    t0 = time.time()

    for spec in args.inputs:
        src = Path(spec)
        rows = []
        for ln in src.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:          # torn append while another agent writes
                print(f"  ! skipped unparseable line in {src.name}", file=sys.stderr)
        out_path = out_dir / (src.stem + ".jev.jsonl")
        out, cache = [], Counter()
        for row in rows:
            norm = normalize_jev(row.get("jev") or {})
            if norm is None:                       # needs a real Jev call
                try:
                    row = {**row, "jev": jev_block(client, row)}
                    norm = normalize_jev(row["jev"])
                except Exception as exc:           # noqa: BLE001
                    cache["api_error"] += 1
                    print(f"  ! {row.get('title','?')[:40]}: {exc}", file=sys.stderr)
                    continue
            if norm is None:
                cache["unnormalizable"] += 1
                continue
            rec = {
                "id": row.get("id") or f"{src.stem}-{len(out):05d}",
                "title": row.get("title", ""), "employer": row.get("employer", ""),
                "pay": row.get("pay", ""),
                "family": row.get("family") or row.get("stratum") or "",
                "source": row.get("source") or (src.stem.replace(".jev", "")),
                "jev": norm,
                "prompt": prompt_for(row.get("title", ""), row.get("employer", ""), row.get("pay", "")),
                "target": target_for(norm),
            }
            if row.get("label"):                   # teacher label, kept as metadata
                rec["teacher"] = row["label"]
            out.append(rec)
            cache[norm["bucket"]] += 1
            if row.get("label", {}).get("bucket"):
                cache["teacher_agree" if row["label"]["bucket"] == norm["bucket"]
                      else "teacher_disagree"] += 1

        with out_path.open("w") as fh:
            for rec in out:
                fh.write(json.dumps(rec) + "\n")
        dist = {k: v for k, v in cache.items()
                if k in ("service_lead", "staff_role", "generic_job", "junk")}
        agree, dis = cache["teacher_agree"], cache["teacher_disagree"]
        u = client.usage
        print(f"[{src.name}] {len(out)}/{len(rows)} labelled -> {out_path.name} | buckets {dist} | "
              f"teacher agree {agree}/{agree+dis} | Jev {u.calls} calls ${u.cost_usd:.4f}")

    u = client.usage
    print(f"[done] {time.time()-t0:.0f}s | Jev total {u.calls} calls, "
          f"{u.input_tokens:,} in / {u.output_tokens:,} out, ${u.cost_usd:.4f}")


if __name__ == "__main__":
    main()
