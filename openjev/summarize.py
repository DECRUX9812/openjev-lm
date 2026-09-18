"""openjev/summarize.py -- final result table for the openjev experiment.

Method A = eval_openjev.py (per-field constrained argmax over an exact JSON scaffold).
Method B = eval_likelihood.py (independent sequence-likelihood scorer).
Both score the same 70 hand-labelled gold rows in openjev/data/eval_gold.jsonl.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def load(name):
    p = RUNS / name
    return json.loads(p.read_text()) if p.exists() else None


def pair_from_prompt(prompt):
    t = e = None
    for ln in prompt.splitlines():
        if ln.startswith("title: "):
            t = ln[7:].strip().lower()
        elif ln.startswith("employer: "):
            e = ln[10:].strip().lower()
    return (t, e)


def leaked_gold_ids():
    """Gold rows whose (title,employer) also appears in the other agent's train.jsonl."""
    g, tr = [], set()
    f = HERE / "data" / "eval_gold.jsonl"
    if f.exists():
        for ln in f.read_text().splitlines():
            if ln.strip():
                g.append(json.loads(ln))
    tf = HERE / "data" / "train.jsonl"
    if tf.exists():
        for ln in tf.read_text().splitlines():
            if ln.strip():
                tr.add(pair_from_prompt(json.loads(ln)["prompt"]))
    return {r["id"] for r in g if pair_from_prompt(r["prompt"]) in tr}


def method_a(name):
    d = load(name)
    if not d:
        return None
    m = d.get("metrics") or {}
    rows = d.get("rows") or []
    out = {
        "n": m.get("n", len(rows)),
        "bucket_acc": m.get("bucket_accuracy"),
        "jev_agree": m.get("jev_bucket_agreement"),
        "bool_agree": m.get("boolean_agreement_overall"),
        "fit_delta": m.get("fit_mean_abs_delta_vs_jev"),
        "conf": m.get("mean_bucket_confidence"),
    }
    return out, rows


def method_b(name):
    d = load(name)
    if not d:
        return None
    s = d.get("summary") or d.get("metrics") or d
    return {
        "n": s.get("n"),
        "bucket_acc": s.get("bucket_acc_vs_gold"),
        "jev_agree": s.get("bucket_acc_vs_jev_bucket"),
        "enum_acc": s.get("enum_bucket_acc_vs_gold"),
        "margin": s.get("bucket_margin_median"),
    }


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    leaks = leaked_gold_ids()
    print(f"gold rows whose (title,employer) also sits in data/train.jsonl: "
          f"{len(leaks)} -> {sorted(leaks)}")
    print()
    rows_out = []

    a_base = method_a("eval_baseline.json")
    if a_base:
        rows_out.append(("Qwen2.5-0.5B untrained (method A)", a_base[0], None))
    b_base = method_b("eval_likelihood_base.json")
    if b_base:
        rows_out.append(("Qwen2.5-0.5B untrained (method B)", b_base, None))
    a_n1 = method_a("eval_A_run-night1.json")
    if a_n1:
        clean = [r for r in a_n1[1] if r["id"] not in leaks]
        acc_clean = sum(r["correct"] for r in clean) / len(clean) if clean else None
        rows_out.append(("run-night1: skewed corpus (method A)", a_n1[0], len(clean), acc_clean))
    a_fin = method_a("eval_A_run-final.json")
    if a_fin:
        rows_out.append(("run-final: balanced Jev corpus (method A)", a_fin[0], None))
    b_fin = method_b("eval_B_run-final.json")
    if b_fin:
        rows_out.append(("run-final: balanced Jev corpus (method B)", b_fin, None))

    print("| arm | n | bucket acc vs gold | agree w/ Jev | bool agree | fit mean |d| | mean conf |")
    print("|---|---|---|---|---|---|---|")
    print("| **Jev itself** (reference) | 70 | **97.1%** | - | - | - | - |")
    print("| stock Qwen2.5-1.5B + PCD engine (reference) | 70 | 77.1% | - | - | - | - |")
    for label, m, n_over, acc_over in rows_out:
        n = n_over or m.get("n")
        acc = pct(acc_over if acc_over is not None else m.get("bucket_acc"))
        if acc_over is not None:
            acc += f" (clean {n}/70)"
        print(f"| {label} | {n} | {acc} | {pct(m.get('jev_agree'))} | "
              f"{pct(m.get('bool_agree'))} | {m.get('fit_delta')} | {m.get('conf')} |")

    j = [x for x in rows_out if x[0].startswith("run-final")]
    if j:
        print()
        print("run-final per-row file:", RUNS / "eval_A_run-final.json")


if __name__ == "__main__":
    main()
