"""openjev/eval_openjev.py -- the measuring instrument for openjev.

Scores a model (untrained base, or base + LoRA adapter) on the held-out gold postings with the
SAME method the published baseline used (parallel constrained decoding, RLCD-style):

  * the answer is the exact JSON shape `openjev/dataset.py:target_for()` writes
      '{"bucket": "<b>", "technical_need": <bool>, ..., "evergreen_repost": <bool>, "fit": <n>}<|im_end|>\n'
  * the JSON scaffolding is FORCED; at each of the 7 fields, only the allowed candidate
    continuations are scored, and the argmax is taken. No free text is ever generated, so
    schema validity is 100% by construction.
  * one batched forward per field (7 forwards per row). Decision positions match the published
    engine (Qwen-2.5-1B-RLCD `core/schema.py:compile_parallel_metadata`): the value is scored at
    the position right after the field's textual scaffold, e.g. bucket at  '{"bucket": "',
    booleans at '", "<field>":', fit at ', "fit":'.

Candidates (exactly as the trainer's targets spell them):
    bucket    -> service_lead | staff_role | generic_job | junk      (inside the opening quote)
    booleans  -> ' true' | ' false'                                  (space-prefixed, one token)
    fit       -> ' 0' | ' 1' | ' 2' | ' 3' | ' 4'                    (space-prefixed)
  Tokenizer check (Qwen2.5): booleans are single tokens; every bucket label is 2 tokens
  ('generic_job' -> ['generic','_job'], 'junk' -> ['j','unk']) and fit is 2 tokens (' ' + digit).
  So bucket/fit - and to stay uniform, ALL fields - are compared by FULL-STRING log-likelihood
  (sum of per-token logprobs) instead of a single-token logit; the softmax over those sums is the
  reported confidence. For single-token candidates this is identical to the single-token argmax.
  (`bucket_probs_firsttok*` keeps the published first-token variant for cross-checking.)

usage:
  python eval_openjev.py --model ../models/Qwen2.5-0.5B-Instruct --data data/eval_gold.jsonl --out runs/eval_baseline.json
  python eval_openjev.py --model ../models/Qwen2.5-0.5B-Instruct --adapter runs/openjev-v2/adapter.pt --out runs/eval_openjev-v2.json
  python eval_openjev.py --recompute runs/eval_baseline.json     # metrics from rows, no model

Resource rules for this box: run under the global model lock and niceness, e.g.
  flock -w 3600 /tmp/jev-model.lock -c 'nice -n 5 <venv>/python openjev/eval_openjev.py ...'
one model in memory at a time, CPU threads capped (default 4).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from train_lora import apply_lora  # noqa: E402  (module layout owned by the trainer; do not fork it)

BUCKETS = ["service_lead", "staff_role", "generic_job", "junk"]
BOOLS = ["technical_need", "business_buyer", "small_firm_doable", "pay_stated", "evergreen_repost"]
FIELDS = ["bucket"] + BOOLS + ["fit"]
EOS = "<|im_end|>\n"

DEFAULT_MODEL = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct")
DEFAULT_DATA = str(HERE / "data" / "eval_gold.jsonl")
JEV_SOURCES = [str(ROOT.parent / "typesafe-lab" / "runs" / "leads_corpus_full.json"),
               str(ROOT / "runs_local" / "leads_jev_live_nocache.json")]


# --------------------------------------------------------------------------------------- model

def load_model(model_path: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    info = {"adapter": None, "adapter_rank": None, "adapter_step": None, "adapter_params": None}
    if adapter:
        apath = Path(adapter)
        cfg = {}
        cfg_path = apath.parent / "adapter_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
        r, alpha = int(cfg.get("rank", 16)), int(cfg.get("alpha", 32))
        n_lora = apply_lora(model, r, alpha, dropout=0.0)  # same module layout the trainer used
        try:
            payload = torch.load(apath, map_location="cpu")
        except Exception:  # noqa: BLE001  (older payloads may need weights_only=False)
            payload = torch.load(apath, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "lora" in payload:      # train_v2.py format 2
            state, step = payload["lora"], payload.get("step")
        elif isinstance(payload, dict) and "model" in payload:   # train_lora.py format
            state, step = payload["model"], payload.get("step")
        else:                                                    # legacy flat LoRA state_dict
            state, step = payload, None
        missing = model.load_state_dict(state, strict=False)
        model_keys = model.state_dict()
        matched = sum(1 for k in state if k in model_keys)
        info.update(adapter=adapter, adapter_rank=r, adapter_step=step, adapter_params=len(state))
        print(f"[adapter] {adapter} | r={r} alpha={alpha} lora_layers={n_lora} "
              f"tensors={len(state)} matched={matched} step={step} "
              f"unexpected={len(missing.unexpected_keys)}", flush=True)
        if matched != len(state):
            print(f"[adapter][WARN] only {matched}/{len(state)} adapter tensors matched the model", flush=True)
    model.eval()
    model.requires_grad_(False)   # inference-only: no autograd graph, no memory surprise
    info["n_lora_layers"] = n_lora if adapter else 0
    return model, tok, info


def decide(model, tok, ctx_text: str, cand_texts: list[str]):
    """Score only `cand_texts` at the position after `ctx_text`; return argmax + softmax probs.

    Full-string log-likelihood: candidate tokens are appended teacher-forced, one row per
    candidate in a single batched forward; the sum of per-token logprobs is the candidate score.
    """
    ctx_ids = tok(ctx_text, add_special_tokens=False)["input_ids"]
    L = len(ctx_ids)
    cand_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in cand_texts]
    rows = [ctx_ids + t[:-1] for t in cand_ids]          # logits for token j sit at position L-1+j
    uniform = len({len(t) for t in cand_ids}) == 1
    if uniform:                                          # all our fields are uniform (k=1 or 2)
        K = len(cand_ids[0])
        inp = torch.tensor([r for r in rows], dtype=torch.long)
        att = torch.ones_like(inp)
        with torch.no_grad():
            logits = None
            for extra in ({"logits_to_keep": K}, {"num_logits_to_keep": K}):
                try:                     # transformers v5 renamed the kwarg (v4: num_logits_to_keep)
                    logits = model(input_ids=inp, attention_mask=att, use_cache=False, **extra).logits
                    break
                except TypeError:
                    continue
            if logits is None or logits.shape[1] != K:   # kwarg silently ignored, or older version
                logits = model(input_ids=inp, attention_mask=att, use_cache=False).logits[:, -K:]
        get = lambda i, j: logits[i, j]                  # noqa: E731  logits[i, j] -> dist for tok j
    else:                                                # defensive fallback (never hit today)
        maxlen = max(len(r) for r in rows)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        inp = torch.full((len(rows), maxlen), pad, dtype=torch.long)
        att = torch.zeros((len(rows), maxlen), dtype=torch.long)
        for i, r in enumerate(rows):
            inp[i, :len(r)] = torch.tensor(r)
            att[i, :len(r)] = 1
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=att, use_cache=False).logits
        get = lambda i, j: logits[i, L - 1 + j]          # noqa: E731
    scores = []
    for i, t in enumerate(cand_ids):
        lp = 0.0
        for j, tid in enumerate(t):
            lp += torch.log_softmax(get(i, j).float(), dim=-1)[tid].item()
        scores.append(lp)
    first = [torch.log_softmax(get(i, 0).float(), dim=-1)[t[0]].item() for i, t in enumerate(cand_ids)]
    p_full = torch.softmax(torch.tensor(scores), dim=0).tolist()
    p_first = torch.softmax(torch.tensor(first), dim=0).tolist()
    return scores.index(max(scores)), p_full, p_first, cand_ids


# ------------------------------------------------------------------------------------ Jev side

def _noul(v, row, name):
    if isinstance(v, dict):
        return v.get("noul", v.get("probability"))
    return row.get(name)


def load_jev(paths):
    """id -> Jev's typed answers (bucket choice, boolean majority, fit score)."""
    jev = {}
    for p in paths:
        try:
            data = json.loads(Path(p).read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[jev] cannot read {p}: {e}", flush=True)
            continue
        for row in data.get("rows", []):
            rid = str(row.get("id"))
            if rid in jev:
                continue
            a = row.get("answers") or {}
            b = a.get("bucket")
            bucket = b.get("choice") if isinstance(b, dict) else (b or row.get("bucket"))
            bools, ok = {}, True
            for name in BOOLS:
                pv = _noul(a.get(name), row, name)
                bools[name] = None if pv is None else (float(pv) >= 0.5)
                ok = ok and bools[name] is not None
            fv = a.get("fit")
            fit = fv.get("score") if isinstance(fv, dict) else row.get("fit")
            if bucket is None or not ok or fit is None:
                continue
            jev[rid] = {"bucket": bucket, "bools": bools, "fit": float(fit), "src": p}
    return jev


# ------------------------------------------------------------------------------------ metrics

def compute_metrics(rows):
    n = len(rows)
    per_gold = {b: {"n": 0, "correct": 0} for b in BUCKETS}
    per_pred = {b: {"n": 0, "correct": 0} for b in BUCKETS}
    confusion = {g: {p: 0 for p in BUCKETS} for g in BUCKETS}
    for r in rows:
        g, p = r["gold"], r["pred_bucket"]
        if g in confusion and p in confusion[g]:
            confusion[g][p] += 1
        if g in per_gold:
            per_gold[g]["n"] += 1
            per_gold[g]["correct"] += bool(r["correct"])
        if p in per_pred:
            per_pred[p]["n"] += 1
            per_pred[p]["correct"] += bool(r["correct"])
    for d in list(per_gold.values()) + list(per_pred.values()):
        d["rate"] = round(d["correct"] / d["n"], 4) if d["n"] else None

    bf = {}
    pairs = []
    for b in BOOLS:
        pp = [(r["bools"][b]["pred"], r["bools"][b]["jev"]) for r in rows if r["bools"][b]["jev"] is not None]
        pairs += pp
        bf[b] = {"n": len(pp), "agree": sum(1 for x, y in pp if x == y),
                 "agreement": round(sum(1 for x, y in pp if x == y) / len(pp), 4) if pp else None}
    deltas = [abs(r["fit"]["pred"] - r["fit"]["jev"]) for r in rows if r["fit"]["jev"] is not None]
    # calibration over the 4 bucket candidates
    ece, bins = 0.0, [[0, 0.0, 0.0] for _ in range(5)]  # n, sum_conf, sum_correct
    brier = 0.0
    for r in rows:
        c = r["bucket_conf"]
        j = min(4, int(c * 5))
        bins[j][0] += 1
        bins[j][1] += c
        bins[j][2] += bool(r["correct"])
        y = {b: (1.0 if b == r["gold"] else 0.0) for b in BUCKETS}
        brier += sum((r["bucket_probs"].get(b, 0.0) - y[b]) ** 2 for b in BUCKETS)
    for cnt, sc, sk in bins:
        if cnt:
            ece += cnt / n * abs(sk / cnt - sc / cnt)
    jev_agree = sum(1 for r in rows if r["pred_bucket"] == r["jev_bucket"] and r["jev_bucket"])
    return {
        "n": n,
        "bucket_accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "per_gold_bucket_recall": per_gold,
        "per_pred_bucket": per_pred,
        "confusion_gold_to_pred": confusion,
        "jev_bucket_agreement": round(jev_agree / n, 4),
        "mean_bucket_confidence": round(sum(r["bucket_conf"] for r in rows) / n, 4),
        "ece_5bin": round(ece, 4),
        "brier_4class": round(brier / n, 4),
        "bucket_accuracy_firsttok_variant": round(
            sum(1 for r in rows if r["bucket_firsttok"] == r["gold"]) / n, 4),
        "firsttok_label_differs_from_fullstring": sum(
            1 for r in rows if r["bucket_firsttok"] != r["pred_bucket"]),
        "boolean_agreement_vs_jev": bf,
        "boolean_agreement_overall": round(sum(1 for x, y in pairs if x == y) / len(pairs), 4) if pairs else None,
        "fit_mean_abs_delta_vs_jev": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "fit_exact_rate_vs_jev": round(
            sum(1 for d in deltas if d == 0) / len(deltas), 4) if deltas else None,
    }


# --------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=str(HERE / "runs" / "eval.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--recompute", default=None, help="recompute metrics from a finished --out json; no model loaded")
    args = ap.parse_args()

    if args.recompute:
        res = json.loads(Path(args.recompute).read_text())
        m = compute_metrics(res["rows"])
        print(json.dumps(m, indent=1))
        return

    torch.set_num_threads(args.threads)
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    jev = load_jev(JEV_SOURCES)
    print(f"[data] {len(rows)} rows from {args.data} | jev vectors for "
          f"{sum(1 for r in rows if str(r['id']) in jev)}/{len(rows)} ids", flush=True)

    t0 = time.time()
    model, tok, minfo = load_model(args.model, args.adapter)
    load_s = time.time() - t0

    cand_map = {"bucket": BUCKETS, BOOLS[0]: [" true", " false"], BOOLS[1]: [" true", " false"],
                BOOLS[2]: [" true", " false"], BOOLS[3]: [" true", " false"],
                BOOLS[4]: [" true", " false"], "fit": [" 0", " 1", " 2", " 3", " 4"]}
    tok_check = {f: {c: tok(c, add_special_tokens=False)["input_ids"] for c in cand_map[f]} for f in FIELDS}

    out_rows, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        tr = time.time()
        prompt = r["prompt"]
        s = prompt + '{"bucket": "'
        bi, bprobs, bfirst, _ = decide(model, tok, s, BUCKETS)
        bucket = BUCKETS[bi]
        s += bucket + '"'
        decided = {"bucket": bucket}
        bool_rec = {}
        for b in BOOLS:
            s += f', "{b}":'
            yi, yp, _, _ = decide(model, tok, s, [" true", " false"])
            val = (yi == 0)
            decided[b] = val
            bool_rec[b] = {"pred": val, "p_true": round(yp[0], 4)}
            s += " true" if val else " false"
        s += ', "fit":'
        fi, fprobs, _, _ = decide(model, tok, s, [" 0", " 1", " 2", " 3", " 4"])
        s += f' {fi}' + "}" + EOS

        jv = jev.get(str(r["id"]))
        for b in BOOLS:
            bool_rec[b]["jev"] = None if jv is None else jv["bools"][b]
            bool_rec[b]["agree"] = None if jv is None else (bool_rec[b]["pred"] == jv["bools"][b])
        body = s[len(prompt):-len(EOS)]
        try:
            format_ok = json.dumps(json.loads(body), separators=(", ", ": ")) == body
        except Exception:  # noqa: BLE001
            format_ok = False
        out_rows.append({
            "id": r["id"], "title": r.get("title"), "gold": r["gold"],
            "jev_bucket": r.get("jev_bucket"), "pred_bucket": bucket,
            "correct": bucket == r["gold"], "agree_jev": bucket == r.get("jev_bucket"),
            "bucket_conf": round(max(bprobs), 4),
            "bucket_probs": {b: round(p, 4) for b, p in zip(BUCKETS, bprobs)},
            "bucket_firsttok": BUCKETS[bfirst.index(max(bfirst))],
            "bucket_probs_firsttok": {b: round(p, 4) for b, p in zip(BUCKETS, bfirst)},
            "bools": bool_rec,
            "fit": {"pred": fi, "jev": None if jv is None else jv["fit"],
                    "delta": None if jv is None else round(fi - jv["fit"], 3)},
            "format_ok": format_ok,
            "answer_text": body,
            "row_s": round(time.time() - tr, 2),
        })
        if i % 10 == 0 or i == len(rows):
            Path(args.out + ".partial").write_text(json.dumps({"done": i, "rows": out_rows}, indent=1))
            print(f"[row {i}/{len(rows)}] {time.time()-t0:.0f}s elapsed", flush=True)

    metrics = compute_metrics(out_rows)
    miss = sum(1 for r in out_rows if r["jev_bucket"] != jev.get(str(r["id"]), {}).get("bucket"))
    result = {
        "meta": {
            "model": args.model, "adapter": args.adapter, "data": args.data, "n": len(out_rows),
            "method": "parallel constrained decoding: exact target_for() JSON shape, argmax over allowed "
                      "candidates per field, one batched forward per field (7/row), candidate scores are "
                      "full-string log-likelihood sums, confidence = softmax over those sums",
            "fields": FIELDS, "candidates": cand_map, "candidate_token_ids": tok_check,
            "prompt_source": "row['prompt'] verbatim from the eval file (compact format)",
            "jev_sources": JEV_SOURCES,
            "jev_bucket_source_mismatches_vs_card": miss,
            "threads": args.threads, "load_s": round(load_s, 1),
            "wall_s": round(time.time() - t0, 1), "row_s_mean": round((time.time() - t0) / len(out_rows), 2),
            "format_ok_all": all(r["format_ok"] for r in out_rows),
            **minfo,
        },
        "metrics": metrics,
        "rows": out_rows,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    m = metrics
    print(f"\n[{'base' if not args.adapter else Path(args.adapter).parent.name}] "
          f"bucket acc={m['bucket_accuracy']:.1%} ({sum(r['correct'] for r in out_rows)}/{m['n']}) | "
          f"jev agree={m['jev_bucket_agreement']:.1%} | mean conf={m['mean_bucket_confidence']:.3f} | "
          f"ECE={m['ece_5bin']:.3f} Brier={m['brier_4class']:.3f} | "
          f"bool agree={m['boolean_agreement_overall']} | fit mean|d|={m['fit_mean_abs_delta_vs_jev']} "
          f"| firsttok-acc={m['bucket_accuracy_firsttok_variant']:.1%}")
    print(f"wrote {args.out} ({result['meta']['wall_s']}s)", flush=True)
    wrong = [r for r in out_rows if not r["correct"]]
    for r in wrong[:12]:
        print(f"  err {r['gold']:>12} -> {r['pred_bucket']:<12} conf={r['bucket_conf']:.2f} {str(r['title'])[:55]}")


if __name__ == "__main__":
    main()
