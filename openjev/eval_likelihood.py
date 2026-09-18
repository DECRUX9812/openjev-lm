#!/usr/bin/env python3
"""openjev/eval_likelihood.py -- INDEPENDENT second-opinion evaluator (method B).

Method B = FULL-ANSWER LIKELIHOOD SCORING, deliberately *not* the per-field
argmax constrained decoder used by method A (eval_openjev.py).  This file shares
no code with method A; it is written from the token ids up.

Two decision rules are computed per row, both likelihood-based:

  (1) STAGED  -- greedy constrained scoring, field by field in answer order
      (bucket, 5 booleans, fit).  At each field every candidate is scored on the
      log-probability of exactly its own value tokens, conditioned on the prompt
      plus all previously chosen fields (teacher forcing, one KV-cached prefix
      shared by all candidates of a stage).  Highest mean-logprob candidate
      wins; the runner-up margin is recorded (the bucket margin is the headline
      number).  The chosen answer's total sequence logprob is then measured with
      one extra teacher-forced pass over the winning full answer.

  (2) ENUM    -- whole-answer enumeration: 12 complete answer strings
      (4 buckets x fit in {0,2,4}, booleans held at the Jev-free neutral
      "false") are each scored as a full answer; argmax by mean logprob gives a
      second, joint bucket decision independent of the staged path, plus the
      margin to the runner-up bucket.

Agreement between (1) and (2) is itself a diagnostic: a row where the two rules
disagree is a row where this model holds no stable opinion.

Why value-region tokens: comparing candidates on the mean logprob of their whole
divergent tail dilutes the difference behind the tens of shared scaffolding
tokens that follow the value, and comparing raw sums length-biases short values
("junk" vs "service_lead").  So each candidate is scored on just the tokens
covering its value characters.  If a tokenizer merge ever shifts that boundary
between candidates the stage falls back to the longest-common-prefix tail (sum
ranking) and flags the row; boundary_fallbacks in the output counts those.

Checkpoint sweep: --sweep runs/<run> scores every *.pt found under the run dir
(plus runs/<run>/adapter.pt); step numbers are read from progress.json and from
the checkpoint payload itself, so a learning curve can be plotted from whatever
adapters survived.

usage:
  python eval_likelihood.py --model ../models/Qwen2.5-0.5B-Instruct --limit 8
  python eval_likelihood.py --adapter runs/run1/adapter.pt --out runs/eval_likelihood_run1.json
  python eval_likelihood.py --sweep runs/run1 --out runs/eval_likelihood_sweep.json
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from train_lora import LoRALinear, apply_lora  # noqa: E402  (read-only reuse of the module layout)

DEFAULT_MODEL = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct")
DEFAULT_DATA = str(HERE / "data" / "eval_gold.jsonl")
DEFAULT_OUT = str(HERE / "runs" / "eval_likelihood_base.json")

BUCKETS = ("service_lead", "staff_role", "generic_job", "junk")
BOOLS = ("technical_need", "business_buyer", "small_firm_doable", "pay_stated", "evergreen_repost")
FITS = ("0", "1", "2", "3", "4")
ENUM_FITS = ("0", "2", "4")
ANSWER_TAIL = "<|im_end|>\n"
LORA_R, LORA_ALPHA = 16, 32


# ---------------------------------------------------------------- answer text
def build_answer(values: dict):
    """Return (answer_text, value_char_spans) for a full answer dict.

    values: keys 'bucket', the 5 bools (as 'true'/'false' strings) and 'fit'.
    value_char_spans: {field: (start, end)} char span of each value inside the
    answer text -- the region that field's candidate is scored on.
    """
    segs = []
    spans = {}

    def add(txt, field=None):
        start = sum(len(s) for s, _ in segs)
        segs.append((txt, field))
        if field is not None:
            spans[field] = (start, start + len(txt))

    add('{"bucket": "')
    add(values["bucket"], "bucket")
    add('"')
    for b in BOOLS:
        add(f', "{b}": ')
        add(values[b], b)
    add(', "fit": ')
    add(values["fit"], "fit")
    add("}")
    add(ANSWER_TAIL)
    return "".join(s for s, _ in segs), spans


def default_values():
    v = {"bucket": BUCKETS[0], "fit": "0"}
    for b in BOOLS:
        v[b] = "false"
    return v


# ------------------------------------------------------------- model handling
def load_lora_state(path: str):
    """Accept {'format':2,'lora':sd,'opt':..,'step':N} or a flat LoRA state_dict."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    step = None
    if isinstance(obj, dict) and isinstance(obj.get("lora"), dict):
        step = obj.get("step")
        obj = obj["lora"]
    if not isinstance(obj, dict):
        raise ValueError(f"unrecognised adapter payload at {path}: {type(obj)}")
    return obj, step


def apply_adapter(model, state) -> dict:
    """Copy LoRA A/B weights into an already apply_lora()-ed model, by name."""
    targets = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            targets[name + ".A.weight"] = mod.A.weight
            targets[name + ".B.weight"] = mod.B.weight
    loaded, unexpected, mismatch = 0, [], []
    with torch.no_grad():
        for k, v in state.items():
            t = targets.get(k)
            if t is None:
                unexpected.append(k)
            elif tuple(t.shape) != tuple(v.shape):
                mismatch.append((k, tuple(v.shape), tuple(t.shape)))
            else:
                t.copy_(v.to(t.dtype))
                loaded += 1
    return {"lora_params_loaded": loaded, "lora_params_total": len(targets),
            "unexpected_keys": len(unexpected), "shape_mismatch": len(mismatch),
            "unexpected_sample": unexpected[:3], "mismatch_sample": mismatch[:2]}


def load_model(model_path: str, threads: int, want_lora: bool):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    n_lora = apply_lora(model, LORA_R, LORA_ALPHA, dropout=0.0) if want_lora else 0
    model.eval()
    return model, tok, {"lora_modules": n_lora}


# ------------------------------------------------------------------ likelihood
@torch.inference_mode()
def _extend(model, cache, built: int, ids, upto: int):
    """Grow `cache` (which already covers ids[:built]) until it covers ids[:upto]."""
    if upto <= built:
        return cache, built
    chunk = torch.tensor([ids[built:upto]], dtype=torch.long)
    mask = torch.ones((1, upto), dtype=torch.long)
    pos = torch.arange(built, upto, dtype=torch.long).unsqueeze(0)
    out = model(input_ids=chunk, attention_mask=mask, position_ids=pos,
                past_key_values=cache, use_cache=True)
    return out.past_key_values, upto


@torch.inference_mode()
def _value_logps(model, cache, ids, a: int, b: int):
    """log P(ids[a:b] | ids[:a]) with a throwaway copy of `cache` (covers ids[:a-1])."""
    c = copy.deepcopy(cache)
    start = max(a - 1, 0)
    chunk = ids[start:b]
    t = torch.tensor([chunk], dtype=torch.long)
    mask = torch.ones((1, start + len(chunk)), dtype=torch.long)
    pos = torch.arange(start, start + len(chunk), dtype=torch.long).unsqueeze(0)
    out = model(input_ids=t, attention_mask=mask, position_ids=pos,
                past_key_values=c, use_cache=False)
    lp = F.log_softmax(out.logits[0, :len(chunk)].float(), dim=-1)
    # logits[j] sits at position start+j and predicts chunk[j+1]
    return [lp[j - 1, chunk[j]].item() for j in range(1, len(chunk))]


class Prefix:
    """Mutable KV-cache over a growing chosen-prefix id list."""

    def __init__(self, model):
        self.model = model
        self.cache = DynamicCache()
        self.built = 0
        self.ids: list[int] = []

    def set_ids(self, ids, rebuild_if_mismatch=True):
        """Point the prefix at `ids` (a superset/extrapolation of what we had)."""
        common = 0
        n = min(len(ids), len(self.ids))
        while common < n and ids[common] == self.ids[common]:
            common += 1
        if common < min(len(self.ids), len(ids)) or self.built > common:
            if not rebuild_if_mismatch:
                raise ValueError("prefix chain diverged")
            self.cache, self.built = DynamicCache(), 0  # cache is stale past this point
        self.ids = list(ids)
        return self

    def ensure(self, upto: int):
        self.cache, self.built = _extend(self.model, self.cache, self.built, self.ids, upto)
        return self


def tokenize_with_spans(tok, prompt: str, answer: str):
    """Tokenise prompt+answer, return (ids, answer_char_offset_at, offsets)."""
    text = prompt + answer
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"], len(prompt)


def score_stage(model, tok, prompt: str, cand_answers, value_spans, prefix: Prefix, stats):
    """Score each candidate's value region under a shared cached prefix.

    cand_answers: [(label, answer_text)]; value_spans: [(start, end)] char spans
    inside the answer text.
    Returns (list of scored dicts in input order, boundary_ok flag).
    """
    encs = [tokenize_with_spans(tok, prompt, a) for _, a in cand_answers]
    seqs = [e[0] for e in encs]
    offs = [e[1] for e in encs]
    plen = encs[0][2]

    spans, ok = [], bool(value_spans) and all(v is not None for v in value_spans)
    if ok:
        for ids, off, (a, b) in zip(seqs, offs, value_spans):
            a, b = plen + a, plen + b
            idx = [i for i, (s, e) in enumerate(off) if s < b and e > a]
            if not idx:
                ok = False
                break
            spans.append((idx[0], idx[-1] + 1, tuple(ids[:idx[0]])))
        if ok:
            ok = all(s[2] == spans[0][2] for s in spans)
    if not ok:  # fallback: score the whole divergent tail (LCP), rank on sums
        n = min(len(s) for s in seqs)
        lo = 0
        while lo < n and all(s[lo] == seqs[0][lo] for s in seqs):
            lo += 1
        spans = [(max(lo, 1), len(s), None) for s in seqs]
        stats["boundary_fallbacks"] += 1

    # the cache must cover ids[:a-1] so the first fed token's logits predict ids[a]
    prefix.set_ids(seqs[0][:max(spans[0][0] - 1, 0)])
    prefix.ensure(len(prefix.ids))

    out = []
    for (label, _), ids, (a, b, _) in zip(cand_answers, seqs, spans):
        lps = _value_logps(model, prefix.cache, ids, a, b)
        s = float(sum(lps))
        out.append({"label": label, "sum_logp": s, "mean_logp": s / max(len(lps), 1),
                    "n_tok": len(lps)})
    return out, ok


def rank(cands):
    """Rank by mean logprob, tie-break by sum (both descending)."""
    return sorted(cands, key=lambda c: (-round(c["mean_logp"], 6), -round(c["sum_logp"], 6)))


def score_row(model, tok, row, stats, use_enum=True):
    """Staged field-by-field likelihood decision + enum12 joint decision."""
    prompt = row["prompt"]
    vals = default_values()
    prefix = Prefix(model)
    prefix.set_ids(tok(prompt, add_special_tokens=False)["input_ids"])

    staged, label_map = {}, {}
    order_of_fields = (("bucket", list(BUCKETS)),
                       *[(b, ["true", "false"]) for b in BOOLS],
                       ("fit", list(FITS)))
    for field, cand_values in order_of_fields:
        cand_answers, spans = [], []
        for cv in cand_values:
            v = dict(vals)
            v[field] = cv
            txt, sp = build_answer(v)
            cand_answers.append((cv, txt))
            spans.append(sp[field])
        scored, ok = score_stage(model, tok, prompt, cand_answers, spans, prefix, stats)
        order = rank(scored)
        vals[field] = order[0]["label"]
        label_map[field] = order[0]["label"]
        staged[field] = {
            "chosen": order[0]["label"],
            "margin": order[0]["mean_logp"] - order[1]["mean_logp"] if len(order) > 1 else None,
            "value_region_only": ok,
            "candidates": [{"value": c["label"], "sum_logp": round(c["sum_logp"], 4),
                            "mean_logp": round(c["mean_logp"], 4), "n_tok": c["n_tok"]}
                           for c in order],
        }

    # exact total sequence logprob of the chosen answer: one extra teacher-forced
    # pass scored over the whole answer region
    chosen_text, _ = build_answer(vals)
    c_scored, _ = score_stage(model, tok, prompt, [("chosen", chosen_text)],
                              [(0, len(chosen_text))], prefix, stats)
    total_lp, n_answer_tok = c_scored[0]["sum_logp"], c_scored[0]["n_tok"]

    # enum12: FULL-ANSWER likelihood scoring -- 12 complete answer strings,
    # booleans at the Jev-free neutral (false), each scored over its whole answer
    enum_cands, enum_spans = [], []
    for b in BUCKETS:
        for f in ENUM_FITS:
            v = default_values()
            v["bucket"], v["fit"] = b, f
            txt, _sp = build_answer(v)
            enum_cands.append((f"{b}|fit{f}", txt))
            enum_spans.append((0, len(txt)))
    enum_scored, enum_ok = None, False
    if use_enum:
        e_prefix = Prefix(model).set_ids(tok(prompt, add_special_tokens=False)["input_ids"])
        enum_scored, enum_ok = score_stage(model, tok, prompt, enum_cands, enum_spans, e_prefix, stats)
    enum_order = rank(enum_scored) if enum_scored else []
    best_per_bucket = {}
    for c in enum_order:
        best_per_bucket.setdefault(c["label"].split("|")[0], c)
    bucket_rank = sorted(best_per_bucket.items(), key=lambda kv: -kv[1]["mean_logp"])
    enum_best = bucket_rank[0][0] if bucket_rank else None
    enum_margin = (bucket_rank[0][1]["mean_logp"] - bucket_rank[1][1]["mean_logp"]) if len(bucket_rank) > 1 else None

    return {
        "staged": {"bucket": label_map["bucket"],
                   "bools": [label_map[b] for b in BOOLS],
                   "fit": int(label_map["fit"]),
                   "bucket_margin": staged["bucket"]["margin"],
                   "bucket_candidates": staged["bucket"]["candidates"],
                   "per_field": staged,
                   "answer_total_logp": round(total_lp, 4),
                   "answer_mean_logp": round(total_lp / max(n_answer_tok, 1), 4),
                   "answer_tokens": n_answer_tok},
        "enum": {"bucket": enum_best,
                 "fit": int(enum_order[0]["label"].split("fit")[1]) if enum_order else None,
                 "bucket_margin": enum_margin,
                 "whole_answer_only": bool(enum_ok),
                 "candidates": [{"label": c["label"], "sum_logp": round(c["sum_logp"], 4),
                                 "mean_logp": round(c["mean_logp"], 4)} for c in enum_order]}
                if enum_scored else None,
    }


# --------------------------------------------------------------------- runners
def run_eval(model, tok, rows, label, log_prefix="", use_enum=True):
    stats = {"boundary_fallbacks": 0}
    records, t0 = [], time.time()
    for i, r in enumerate(rows):
        out = score_row(model, tok, r, stats, use_enum=use_enum)
        records.append({"id": r.get("id"), "title": r.get("title"), "gold": r.get("gold"),
                        "jev_bucket": r.get("jev_bucket"), **out})
        if (i + 1) % 10 == 0 or (i + 1) == len(rows):
            el = time.time() - t0
            print(f"{log_prefix}[{label}] {i+1}/{len(rows)} rows  {el:.0f}s "
                  f"({el/(i+1):.2f}s/row, eta {(len(rows)-i-1)*el/(i+1):.0f}s)", flush=True)
    elapsed = time.time() - t0
    n = len(records)
    f = lambda k: sum(1 for r in records if k(r))  # noqa: E731
    bm = [r["staged"]["bucket_margin"] for r in records if r["staged"]["bucket_margin"] is not None]
    em = [r["enum"]["bucket_margin"] for r in records if r["enum"] and r["enum"]["bucket_margin"] is not None]
    al = [r["staged"]["answer_total_logp"] for r in records]
    am = [r["staged"]["answer_mean_logp"] for r in records]
    summary = {
        "n_rows": n,
        "bucket_acc_vs_gold": round(f(lambda r: r["staged"]["bucket"] == r["gold"]) / n, 4) if n else None,
        "bucket_acc_vs_jev_bucket": round(f(lambda r: r["staged"]["bucket"] == r["jev_bucket"]) / n, 4) if n else None,
        "enum_bucket_acc_vs_gold": round(f(lambda r: r["enum"] and r["enum"]["bucket"] == r["gold"]) / n, 4) if n else None,
        "enum_bucket_acc_vs_jev_bucket": round(f(lambda r: r["enum"] and r["enum"]["bucket"] == r["jev_bucket"]) / n, 4) if n else None,
        "staged_enum_bucket_agreement": round(f(lambda r: r["enum"] and r["enum"]["bucket"] == r["staged"]["bucket"]) / n, 4) if n else None,
        "bucket_margin_mean": round(statistics.fmean(bm), 4) if bm else None,
        "bucket_margin_median": round(statistics.median(bm), 4) if bm else None,
        "enum_bucket_margin_mean": round(statistics.fmean(em), 4) if em else None,
        "answer_total_logp_mean": round(statistics.fmean(al), 4) if al else None,
        "answer_mean_logp_mean": round(statistics.fmean(am), 4) if am else None,
        "boundary_fallbacks": stats["boundary_fallbacks"],
        "elapsed_s": round(elapsed, 1),
        "s_per_row": round(elapsed / n, 2) if n else None,
    }
    return summary, records


def find_checkpoints(run_dir: Path):
    """Every *.pt under the run dir, with step numbers where determinable."""
    latest_step = None
    prog = run_dir / "progress.json"
    if prog.exists():
        try:
            latest_step = json.loads(prog.read_text()).get("step")
        except Exception:
            pass
    out = []
    for p in sorted(set(run_dir.rglob("*.pt"))):
        try:
            _sd, step = load_lora_state(str(p))
        except Exception:
            continue
        if step is None and p.name == "adapter.pt":
            step = latest_step
        out.append({"path": str(p), "step": step, "name": p.name,
                    "is_latest_adapter": p.name == "adapter.pt"})
    out.sort(key=lambda c: (c["step"] is None, c["step"] if c["step"] is not None else 0, c["path"]))
    return out, latest_step


def main():
    ap = argparse.ArgumentParser(description="openjev method-B likelihood evaluator")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="path to a LoRA .pt checkpoint")
    ap.add_argument("--sweep", default=None,
                    help="run dir (e.g. runs/run1): score every *.pt under it -> learning curve")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-enum", action="store_true",
                    help="skip the 12-candidate whole-answer stage (~85%% of per-row cost)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    model, tok, minfo = load_model(args.model, args.threads,
                                   want_lora=bool(args.adapter or args.sweep))
    meta = {"method": "B: full-answer / staged likelihood scoring",
            "model": args.model, "adapter": args.adapter, "data": args.data,
            "n_rows": len(rows), "threads": args.threads, "tag": args.tag,
            "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta.update(minfo)
    if args.adapter:
        state, step = load_lora_state(args.adapter)
        meta["adapter_step"] = step
        meta["adapter_load"] = apply_adapter(model, state)

    payload = {"meta": meta}
    if args.sweep:
        run_dir = Path(args.sweep)
        ckpts, latest_step = find_checkpoints(run_dir)
        print(f"[sweep] {run_dir}: {len(ckpts)} checkpoint(s) (progress.json step={latest_step})",
              flush=True)
        if len(ckpts) <= 1:
            print("[sweep] NOTE: only one checkpoint available -- a learning curve needs "
                  "multiple saved adapters (the trainer keeps only the latest). Scoring this "
                  "one anyway.", flush=True)
        curve = []
        for c in ckpts:
            state, step = load_lora_state(c["path"])
            ainfo = apply_adapter(model, state)
            summ, recs = run_eval(model, tok, rows, label=f"step={step}", log_prefix="[sweep] ",
                                  use_enum=not args.no_enum)
            summ.update({"checkpoint": c["path"], "step": step, "load": ainfo})
            curve.append({"summary": summ, "rows": recs})
            print(f"[sweep] step={step} acc_vs_gold={summ['bucket_acc_vs_gold']} "
                  f"enum={summ['enum_bucket_acc_vs_gold']}", flush=True)
        payload["sweep"] = {
            "run_dir": str(run_dir), "latest_step": latest_step,
            "curve": [{"step": c["summary"]["step"], "checkpoint": c["summary"]["checkpoint"],
                       "bucket_acc_vs_gold": c["summary"]["bucket_acc_vs_gold"],
                       "bucket_acc_vs_jev_bucket": c["summary"]["bucket_acc_vs_jev_bucket"],
                       "enum_bucket_acc_vs_gold": c["summary"]["enum_bucket_acc_vs_gold"],
                       "staged_enum_bucket_agreement": c["summary"]["staged_enum_bucket_agreement"]}
                      for c in curve],
            "per_checkpoint_rows": {str(c["summary"]["step"]): c["rows"] for c in curve},
            "single_checkpoint_note": (len(ckpts) <= 1)}
    else:
        summ, recs = run_eval(model, tok, rows, label="eval", use_enum=not args.no_enum)
        payload["summary"], payload["rows"] = summ, recs

    payload["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"[out] {out}", flush=True)
    if not args.sweep:
        print(json.dumps(payload["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
