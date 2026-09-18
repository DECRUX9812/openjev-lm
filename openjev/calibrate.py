"""openjev/calibrate.py -- temperature scaling for likelihood-scored fields.

Ported from kev's evaluate.test_temperature: the staged evaluator emits one
mean_logp per candidate value per field; softmax(mean_logp / T) turns those
into probabilities. One global T is fit on dev rows by minimizing NLL, then
ECE/NLL are reported on the gold holdout before and after scaling.

usage:
  python calibrate.py --adapter ../runs/run-final/adapter.pt \
      --fit ../data/dev_jev.jsonl --eval ../data/eval_gold.jsonl \
      --fit-rows 80    # writes <adapter-dir>/calibration.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_likelihood as E

FIELDS = ("bucket",) + E.BOOLS + ("fit",)


def collect(model, tok, rows, limit):
    """[(field, [mean_logp per cand], gold_idx)] — one staged pass per row."""
    out = []
    stats = {"boundary_fallbacks": 0}
    for i, row in enumerate(rows[:limit]):
        res = E.score_row(model, tok, row, stats, use_enum=False)
        try:
            target = json.loads(row["target"].split("<|im_end|>")[0])
        except Exception:
            continue
        for f in FIELDS:
            st = res["staged"]["per_field"][f]
            cands = st["candidates"]           # ranked order
            gold_v = str(target.get(f, "")).lower()
            if f in E.BOOLS:
                gold_v = "true" if gold_v == "true" else "false"
            gi = next((j for j, c in enumerate(cands) if str(c["value"]).lower() == gold_v), None)
            if gi is None:
                continue
            out.append((f, [c["mean_logp"] for c in cands], gi))
    return out


def nll_ece(samples, T=1.0):
    import math
    nll, ece_bins = 0.0, {}
    n = 0
    for f, logits, gi in samples:
        mx = max(logits)
        exps = [math.exp((l - mx) / T) for l in logits]
        tot = sum(exps)
        probs = [x / tot for x in exps]
        nll -= math.log(max(probs[gi], 1e-12))
        n += 1
        conf, corr = max(probs), probs.index(max(probs)) == gi
        b = min(int(conf * 10), 9)
        ece_bins.setdefault(b, []).append((conf, corr))
    ece = 0.0
    for b, xs in ece_bins.items():
        acc = sum(c for _, c in xs) / len(xs)
        avg = sum(c for c, _ in xs) / len(xs)
        ece += len(xs) / n * abs(acc - avg)
    return nll / n, ece


def fit_T(samples):
    lo, hi = 0.1, 8.0
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll_ece(samples, m1)[0] < nll_ece(samples, m2)[0]:
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=str(Path(__file__).resolve().parents[1] / "runs" / "run-final" / "adapter.pt"))
    ap.add_argument("--fit", default=str(Path(__file__).resolve().parents[1] / "data" / "dev_jev.jsonl"))
    ap.add_argument("--eval", dest="evalf", default=str(Path(__file__).resolve().parents[1] / "data" / "eval_gold.jsonl"))
    ap.add_argument("--fit-rows", type=int, default=80)
    ap.add_argument("--eval-rows", type=int, default=70)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    model, tok, _ = E.load_model(a.model, a.threads, want_lora=True)
    state, step = E.load_lora_state(a.adapter)
    E.apply_adapter(model, state)
    fit_rows = [json.loads(l) for l in open(a.fit)][: a.fit_rows]
    ev_rows = [json.loads(l) for l in open(a.evalf)][: a.eval_rows]

    t0 = time.time()
    fit_s = collect(model, tok, fit_rows, a.fit_rows)
    ev_s = collect(model, tok, ev_rows, a.eval_rows)
    T = fit_T(fit_s)
    nll0, ece0 = nll_ece(ev_s, 1.0)
    nll1, ece1 = nll_ece(ev_s, T)
    out_path = a.out or str(Path(a.adapter).resolve().parent / "calibration.json")
    res = {"temperature": round(T, 4), "fit_rows": len(fit_rows), "eval_rows": len(ev_rows),
           "field_decisions": len(ev_s), "nll_before": round(nll0, 4), "nll_after": round(nll1, 4),
           "ece_before": round(ece0, 4), "ece_after": round(ece1, 4),
           "adapter_step": step, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=2))
    Path(out_path).write_text(json.dumps(res, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
