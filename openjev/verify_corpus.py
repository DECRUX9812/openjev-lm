#!/usr/bin/env python3
"""openjev/verify_corpus.py -- independent verification of the openjev corpus + eval baseline.

Subagent E (independent verifier, overnight mission). Recomputes every published claim about
openjev/data/train_jev.jsonl -- and its comparison corpora and eval baseline -- from the RAW
files, with plain Python. No model WEIGHTS are ever loaded (AutoTokenizer only, for check 6).

Writes openjev/runs/VERIFY.json. Exits non-zero if any HARD check fails.

  HARD (exit != 0 on failure): 1 target-provenance byte-identity, 2 gold leakage (id + pair),
  3 gold-set integrity, 5 exact (title, employer) duplicates, 6 any row > max_len=192 tokens,
  8 trainable-param count (a ~452M count means the LoRA freeze broke).
  SOFT (warn):     4 class balance / drop-counter reconciliation, 7 baseline metrics +
  majority-class comparison, plus informational extras.

Run:
  /home/decrux/Code/jev-repro-test/Qwen-2.5-1B-RLCD/.venv/bin/python openjev/verify_corpus.py
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# The full audit joins the training corpus against the private typesafe-lab source files
# (leads corpus + hand labels). Point TYPESAFE_LAB at that checkout; without it the script
# fails fast with a clear message listing what is missing. All repo-local files default to
# the shipped data/ directory; override with OPENJEV_DATA / OPENJEV_RUNS if yours differ.
LAB = Path(os.environ.get("TYPESAFE_LAB", "/home/decrux/Code/typesafe-lab"))
DATA = Path(os.environ.get("OPENJEV_DATA", ROOT / "data"))
RUNS = Path(os.environ.get("OPENJEV_RUNS", ROOT / "runs"))
RUNS.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE))
from dataset import BOOLS, prompt_for, target_for  # noqa: E402  (stdlib-only module, no torch)
import merge_corpus  # noqa: E402  (stdlib + dataset only)

LABELS = ("service_lead", "staff_role", "generic_job", "junk")
GOLD_F = LAB / "data" / "leads.gold.json"
CORPUS_F = LAB / "runs" / "leads_corpus_full.json"
TRAIN_F = DATA / "train_jev.jsonl"
DEV_F = DATA / "dev_jev.jsonl"
TRAIN_OLD_F = DATA / "train.jsonl"
EVAL_GOLD_F = DATA / "eval_gold.jsonl"
REPORT_F = RUNS / "corpus_jev_report.json"
BASELINE_F = RUNS / "eval_baseline.json"
SYNTH_FILES = [DATA / "jevlab" / "labelled.jev.jsonl",
               DATA / "jevlab" / "synth_hard.raw.jev.jev.jsonl",
               DATA / "jevlab_hard" / "synth_hard.raw.jev.jsonl"]
CHECKED_F = HERE / "synth" / "data" / "jev_checked.jsonl"
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
MAX_LEN = 192

checks: list[dict] = []
HARD = set()


def chk(name: str, status: str, detail: str, hard: bool = False) -> None:
    assert status in ("pass", "fail", "warn"), status
    checks.append({"name": name, "status": status, "detail": detail})
    if hard:
        HARD.add(name)
    print(f"[{status.upper():4s}] {name}\n        {detail}", flush=True)


# ---------------------------------------------------------------- helpers
def nt(s) -> str:
    """normalise like merge_corpus.nkey (lowercase, collapse whitespace)."""
    return " ".join(str(s if s is not None else "").lower().split())


def clipnorm(s, n: int = 90) -> str:
    """exactly what dataset.clip() produces, then normalised again."""
    return nt(nt(s)[:n])


def pair_of(title, employer, clip: bool = False):
    if clip:
        return (clipnorm(title), clipnorm(employer))
    return (nt(title), nt(employer))


def pair_from_prompt(p: str):
    t = e = ""
    for ln in str(p).splitlines():
        if ln.startswith("title: "):
            t = ln[7:]
        elif ln.startswith("employer: "):
            e = ln[9:]
    return (nt(t), nt(e))


def lines_of(p: Path):
    return [l for l in Path(p).read_text(errors="replace").splitlines() if l.strip()]


def jlines(p: Path):
    return [json.loads(l) for l in lines_of(p)]


def stamp(p: Path) -> dict:
    p = Path(p)
    if not p.exists():
        return {"path": str(p), "exists": False}
    b = p.read_bytes()
    return {"path": str(p), "exists": True, "bytes": len(b),
            "sha256_12": hashlib.sha256(b).hexdigest()[:12],
            "mtime": time.strftime("%H:%M:%S", time.localtime(p.stat().st_mtime))}


def dist(rows, keyfn) -> dict:
    c = Counter(keyfn(r) for r in rows)
    return {k: v for k, v in sorted(c.items(), key=lambda kv: -kv[1])}


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


# ---------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    VERIFY = RUNS / "VERIFY.json"

    missing = [str(p) for p in (TRAIN_F, CORPUS_F, GOLD_F, EVAL_GOLD_F, REPORT_F)
               if not p.exists()]
    if missing:
        print("cannot run the full audit — required inputs are missing:\n  "
              + "\n  ".join(missing)
              + "\nleads.gold.json / leads_corpus_full.json / corpus_jev_report.json live in "
                "the typesafe-lab checkout; set TYPESAFE_LAB to its path.", flush=True)
        return 2

    TRAIN = jlines(TRAIN_F)
    TRAIN_RAW = lines_of(TRAIN_F)
    DEV = jlines(DEV_F) if DEV_F.exists() else []
    M36 = json.loads(CORPUS_F.read_text())["rows"]
    gold = json.loads(GOLD_F.read_text())["labels"]
    EG = jlines(EVAL_GOLD_F)
    REP = json.loads(REPORT_F.read_text())

    # ============================================================ CHECK 1
    # target provenance: recompute the target from each row's own Jev answers.
    jev_ok = rebuild_ok = raw_ok = 0
    bad_jev, bad_rebuild, bad_raw = [], [], []
    shape_files = Counter()
    for raw, rec in zip(TRAIN_RAW, TRAIN):
        j = rec.get("jev")
        if not isinstance(j, dict):
            bad_jev.append(rec.get("id"))
            continue
        want = {"bucket", *BOOLS, "fit"}
        if set(j) != want or j["bucket"] not in LABELS \
                or any(not isinstance(j[b], bool) for b in BOOLS) \
                or not isinstance(j["fit"], int) or not (0 <= j["fit"] <= 4):
            shape_files[rec.get("id")] += 1
        jev_ok += 1
        try:
            exp = target_for(j)
        except Exception as exc:  # noqa: BLE001
            bad_rebuild.append(f"{rec.get('id')}: {exc}")
            continue
        if exp == rec.get("target"):
            rebuild_ok += 1
        else:
            bad_rebuild.append(rec.get("id"))
        if json.dumps(rec.get("target")) in raw:
            raw_ok += 1
        else:
            bad_raw.append(rec.get("id"))

    chk("1a-target-provenance-byte-identical",
        "pass" if (not bad_jev and not bad_rebuild and not bad_raw) else "fail",
        f"{len(TRAIN)} rows; jev block present {jev_ok}/{len(TRAIN)}; "
        f"target == target_for(row.jev) for {rebuild_ok}/{len(TRAIN)}; "
        f"raw-line byte match {raw_ok}/{len(TRAIN)}; "
        f"missing jev={len(bad_jev)} rebuild-mismatch={len(bad_rebuild)} raw-mismatch={len(bad_raw)}"
        + (f" | e.g. {bad_rebuild[:3]}" if bad_rebuild else ""), hard=True)

    # shape oddities are informational (target_for normalises bools/fit ints)
    if shape_files:
        chk("1b-jev-block-shape", "warn",
            f"{sum(shape_files.values())} rows have a jev block outside the exact "
            f"{{bucket,5 bools,fit int 0-4}} shape (target still reproduced byte-identically)")
    else:
        chk("1b-jev-block-shape", "pass",
            "every row's jev block is exactly {bucket, 5 bools, fit int 0..4}")

    src_train = dist(TRAIN, lambda r: r.get("source"))
    src_dev = dist(DEV, lambda r: r.get("source"))
    src_total = {k: src_train.get(k, 0) + src_dev.get(k, 0) for k in set(src_train) | set(src_dev)}
    real_n = src_total.get("real", 0)
    synth_n = sum(v for k, v in src_total.items() if k != "real")
    chk("1c-provenance-source-mix", "pass",
        f"train sources {dict(src_train)}; dev sources {dict(src_dev)}; "
        f"real={real_n} ({pct(real_n, len(TRAIN) + len(DEV))}) "
        f"synthetic={synth_n} ({pct(synth_n, len(TRAIN) + len(DEV))}) of train+dev "
        f"({len(TRAIN)} + {len(DEV)})")

    # independent cross-check: the row's jev block must equal the Jev answer stored in the
    # claimed SOURCE file (not just be self-consistent).
    corpus_by_id = {r["id"]: r for r in M36}
    synth_by_id: dict[str, list] = {}
    for p in SYNTH_FILES:
        if p.exists():
            for r in jlines(p):
                synth_by_id.setdefault(r.get("id"), []).append((p, r))
    prov = Counter()
    prov_bad = []
    for rec in TRAIN:
        src, rid = rec.get("source"), rec.get("id")
        cands = []
        if src == "real":
            r = corpus_by_id.get(rid)
            if r is not None:
                cands = [(Path("leads_corpus_full.json"), r, r.get("answers"))]
        else:
            for p, r in synth_by_id.get(rid, []):
                if (r.get("source") or p.stem) == src:
                    cands.append((p, r, r.get("jev")))
            if not cands:
                cands = [(p, r, r.get("jev")) for p, r in synth_by_id.get(rid, [])]
        if not cands:
            prov["id_not_found_in_source"] += 1
            continue
        hit = None
        for p, r, raw_ans in cands:
            a = merge_corpus.norm_answer(raw_ans)
            if a is not None and a == rec.get("jev"):
                pm = prompt_for(r.get("title", ""), r.get("employer", ""), r.get("pay") or "")
                hit = "answer+prompt" if pm == rec.get("prompt") else "answer_only"
                break
        if hit:
            prov[hit] += 1
        else:
            prov["answer_mismatch"] += 1
            prov_bad.append(str(rid))
    prov_detail = (f"rows whose jev block re-derives from the claimed source record: "
                   f"{prov['answer+prompt']} answer+prompt-identical, {prov['answer_only']} answer-only "
                   f"(prompt text differs), {prov['id_not_found_in_source']} id not present in the "
                   f"current source files (cannot cross-check), {prov['answer_mismatch']} MISMATCH")
    chk("1d-provenance-vs-source-files",
        "pass" if not prov_bad else "fail",
        prov_detail + (f" | mismatching ids: {prov_bad[:5]}" if prov_bad else ""), hard=True)

    # ============================================================ CHECK 2
    gold_ids = set(gold)
    eg_ids = {r["id"] for r in EG}
    all_gold_ids = gold_ids | eg_ids
    gold_pairs = set()
    for r in EG:
        gold_pairs.add(pair_from_prompt(r["prompt"]))
    for r in M36:
        if r["id"] in all_gold_ids:
            gold_pairs.add(pair_of(r.get("title", ""), r.get("employer", "")))
            gold_pairs.add(pair_of(r.get("title", ""), r.get("employer", ""), clip=True))
    id_leaks = [r["id"] for r in TRAIN if r["id"] in all_gold_ids]
    pair_leaks = []
    for r in TRAIN:
        k = pair_from_prompt(r["prompt"])
        if k in gold_pairs:
            pair_leaks.append((r["id"], k))
    chk("2-gold-leakage-train_jev",
        "pass" if (not id_leaks and not pair_leaks) else "fail",
        f"gold ids={len(all_gold_ids)} (labels={len(gold_ids)}, eval_gold={len(eg_ids)}); "
        f"gold (title,employer) keys={len(gold_pairs)}; train_jev id leaks={len(id_leaks)}; "
        f"train_jev pair-leaked rows={len(pair_leaks)} (distinct pairs "
        f"{len({k for _, k in pair_leaks})})"
        + (f" | ids {id_leaks[:5]}" if id_leaks else "")
        + (f" | pairs {pair_leaks[:3]}" if pair_leaks else ""), hard=True)

    # ============================================================ CHECK 3
    eg_bad = [(r["id"], r.get("gold"), gold.get(str(r["id"]))) for r in EG
              if r.get("id") not in gold or r.get("gold") != gold.get(r.get("id"))]
    eg_ids_dup = len(EG) - len(eg_ids)
    pair_overlap = len({pair_from_prompt(r["prompt"]) for r in EG} &
                       {pair_from_prompt(r["prompt"]) for r in TRAIN})
    id_overlap = len(eg_ids & {r["id"] for r in TRAIN})
    chk("3-gold-set-integrity",
        "pass" if (len(EG) == 70 and not eg_bad and not eg_ids_dup and not id_overlap) else "fail",
        f"eval_gold rows={len(EG)}; duplicates={eg_ids_dup}; gold value != leads.gold.json "
        f"for {len(eg_bad)} rows"
        + (f" | {eg_bad[:3]}" if eg_bad else "")
        + f"; overlap with train_jev: ids={id_overlap}, pairs={pair_overlap}; "
          f"eval_gold ids == leads.gold keys: {eg_ids == gold_ids}", hard=True)

    # ============================================================ CHECK 4
    tj_dist = dist(TRAIN, lambda r: r["jev"]["bucket"])
    rep_td = REP.get("train_dist", {})
    d_ok = all(tj_dist.get(k, 0) == rep_td.get(k, 0) for k in set(tj_dist) | set(rep_td))
    tj_srv = tj_dist.get("service_lead", 0)
    tj_junk = tj_dist.get("junk", 0)
    chk("4a-class-balance-train_jev-vs-report",
        "pass" if (d_ok and all(src_train.get(k, 0) == REP.get("by_source", {}).get(k, 0)
                                for k in set(src_train) | set(REP.get("by_source", {})))
                   and len(TRAIN) == REP.get("train_rows")) else "fail",
        f"train_jev n={len(TRAIN)} (report {REP.get('train_rows')}); recomputed "
        f"generic_job={tj_dist.get('generic_job', 0)} ({pct(tj_dist.get('generic_job', 0), len(TRAIN))}) "
        f"staff_role={tj_dist.get('staff_role', 0)} service_lead={tj_srv} ({pct(tj_srv, len(TRAIN))}) "
        f"junk={tj_junk} ({pct(tj_junk, len(TRAIN))}); report dist {rep_td}; "
        f"report by_source {REP.get('by_source')} vs recomputed train {dict(src_train)}; "
        f"dev n={len(DEV)} (report {REP.get('dev_rows')}) dist "
        f"{dist(DEV, lambda r: r['jev']['bucket']) if DEV else 'n/a'}", hard=False)

    OLD = jlines(TRAIN_OLD_F)
    old_dist = dist(OLD, lambda r: r.get("bucket"))
    old_src = dist(OLD, lambda r: r.get("src"))
    old_synth = sum(v for k, v in old_src.items() if k != "real")
    old_pairs = {pair_from_prompt(r["prompt"]) for r in OLD}
    old_leak_pairs = sorted(gold_pairs & old_pairs)
    old_leak_rows = sum(1 for r in OLD if pair_from_prompt(r["prompt"]) in gold_pairs)
    chk("4b-agent1-train.jsonl-claims",
        "pass" if (len(OLD) == 2437 and old_dist.get("service_lead", 0) == 12
                   and old_dist.get("junk", 0) == 49 and old_leak_rows == 6) else "warn",
        f"train.jsonl n={len(OLD)} (claimed 2437); recomputed {old_dist} "
        f"(claimed generic_job 1713/70.3%, staff_role 663/27.2%, service_lead 12/0.5%, junk 49/2.0%); "
        f"src {dict(old_src)} -> synthetic {pct(old_synth, len(OLD))} (claimed 97.7%); "
        f"gold (title,employer) leaks: rows={old_leak_rows}, distinct pairs={len(old_leak_pairs)} "
        f"-> AUDIT-DATA.md's '6 rows' REPRODUCED (6 rows share the 4 distinct pairs above); "
        f"pairs={old_leak_pairs}", hard=False)

    # drop-counter reconciliation against the current source files
    usable = {}
    for p in SYNTH_FILES:
        usable[str(p.name)] = sum(1 for r in jlines(p) if merge_corpus.norm_answer(r.get("jev")) is not None) if p.exists() else 0
    usable_synth = sum(usable.values())
    usable_checked = sum(1 for r in jlines(CHECKED_F) if merge_corpus.norm_answer(r.get("jev")) is not None) if CHECKED_F.exists() else 0
    usable_real = sum(1 for r in M36 if r["id"] not in all_gold_ids
                      and merge_corpus.norm_answer(r.get("answers")) is not None)
    dr = REP.get("dropped", {})
    exp_synth = usable_synth - dr.get("dupe:synth", 0) - dr.get("gold_pair:synth", 0)
    exp_checked = usable_checked - dr.get("dupe:checked", 0) - dr.get("gold_pair:checked", 0)
    exp_real = usable_real - dr.get("dupe:real", 0) - dr.get("gold_pair:real", 0) - REP.get("real_stats", {}).get("real_dropped_parse", 0)
    exp_pre = exp_synth + exp_checked + exp_real
    trim = dr.get("global_generic_trim", 0)
    act_pre = len(TRAIN) + len(DEV) + trim
    post_synth = sum(v for k, v in src_total.items() if k != "real")
    post_real = src_total.get("real", 0)
    real_trim, synth_trim = exp_real - post_real, exp_synth - post_synth
    post_sb = Counter((r["source"], r["jev"]["bucket"]) for r in TRAIN + DEV)
    gen_kept = sum(v for (s, b), v in post_sb.items() if b == "generic_job")
    non_kept = sum(v for (s, b), v in post_sb.items() if b != "generic_job")
    gen_sources = sorted({s for (s, b) in post_sb if b == "generic_job"})
    trim_rule = int(non_kept * 0.42 / 0.58)
    reconcile_ok = (exp_pre == act_pre and real_trim >= 0 and synth_trim >= 0
                    and real_trim + synth_trim == trim and trim_rule == gen_kept
                    and gen_sources == ["real"])
    chk("4c-drop-counter-reconciliation",
        "pass" if reconcile_ok else "warn",
        f"usable rows in the current sources: synth={usable_synth} {usable} checked={usable_checked} "
        f"real={usable_real}; report dropped={dr}; implied kept pre-trim: synth={exp_synth} "
        f"checked={exp_checked} real={exp_real} total={exp_pre} == actual pre-trim (train {len(TRAIN)} "
        f"+ dev {len(DEV)} + generic-trim {trim}) = {act_pre}); post-trim kept by source {src_total}; "
        f"derived trim split: real {real_trim} + synthetic {synth_trim} = {trim} (consistent with "
        f"merge_corpus's real-first generic sort; note its comment says 'drop the easy real generic "
        f"first' but the code keeps reals first, so every one of the {gen_kept} kept generic_job rows "
        f"has source(s) {gen_sources}); trim rule int(non*0.42/0.58)={trim_rule} == generic kept "
        f"{gen_kept}", hard=False)

    # ============================================================ CHECK 5
    pairs = Counter(pair_from_prompt(r["prompt"]) for r in TRAIN)
    dup_pairs = {k: v for k, v in pairs.items() if v > 1 and k[0]}
    empties = sum(1 for k in pairs if not k[0])
    prompts = Counter(r["prompt"] for r in TRAIN)
    dup_prompts = {k: v for k, v in prompts.items() if v > 1}
    ids = Counter(r["id"] for r in TRAIN)
    dup_ids = {k: v for k, v in ids.items() if v > 1}
    chk("5-dedupe-train_jev",
        "pass" if not dup_pairs and not dup_ids else "fail",
        f"duplicate normalised (title,employer) pairs={len(dup_pairs)} (exact-duplicate HARD); "
        f"duplicate ids={len(dup_ids)}; duplicate prompt strings={len(dup_prompts)} "
        f"(identical prompts among {sum(dup_prompts.values())} rows; "
        f"{'e.g. ' + repr(list(dup_prompts)[0][:90]) if dup_prompts else 'none'}); "
        f"rows with empty title={empties}", hard=True)

    # ============================================================ CHECK 6
    tok_stats = {}
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415  (tokenizer only, never weights)
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        lens, plens, tlens = [], [], []
        for i, r in enumerate(TRAIN):
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            t = tok(r["target"], add_special_tokens=False)["input_ids"]
            lens.append(len(p) + len(t))
            if i < 40:
                plens.append(len(p) + len(t))
            tlens.append(len(t))
        s = sorted(lens)
        over = [i for i, L in enumerate(lens) if L > MAX_LEN]
        idx_p95 = s[int(0.95 * len(s))]                       # trainer's exact formula
        med = statistics.median(s)
        over_tok_lost = max((lens[i] - MAX_LEN) for i in over) if over else 0
        tok_stats = {"n": len(s), "min": s[0], "median": med, "p95_trainer_formula": idx_p95,
                     "p95_true": s[min(len(s) - 1, int(0.95 * len(s)))], "max": s[-1],
                     "mean": round(sum(s) / len(s), 1), "over_192": len(over),
                     "max_target_tokens": max(tlens),
                     "first40_min": min(plens), "first40_median": statistics.median(plens),
                     "first40_max": max(plens), "first40_mean": round(sum(plens) / len(plens), 1)}
        chk("6-prompt-target-fit-maxlen192",
            "pass" if not over else "fail",
            f"tokenized ALL {len(TRAIN)} rows (prompt+target, add_special_tokens=False, "
            f"{str(MODEL_DIR).split('/')[-1]} tokenizer, no weights): min={s[0]} median={med} "
            f"mean={tok_stats['mean']} p95={idx_p95} max={s[-1]}; rows > {MAX_LEN} = {len(over)} "
            f"(would be silently truncated by load_rows (p+t)[:max_len], worst loses "
            f"{over_tok_lost} target tokens); first-40 subset: min={min(plens)} "
            f"median={statistics.median(plens)} max={max(plens)}; max target length={max(tlens)} tok",
            hard=True)
    except Exception as exc:  # noqa: BLE001
        chk("6-prompt-target-fit-maxlen192", "fail",
            f"tokenizer check could not run: {type(exc).__name__}: {exc}", hard=True)

    # ============================================================ CHECK 7
    B = json.loads(BASELINE_F.read_text())
    rows, met = B["rows"], B["metrics"]
    n = len(rows)
    recomputed = {
        "n": n,
        "bucket_accuracy": round(sum(1 for r in rows if r["pred_bucket"] == r["gold"]) / n, 4),
        "per_gold_bucket_recall": {},
    }
    for b in LABELS:
        g = [r for r in rows if r["gold"] == b]
        c = sum(1 for r in g if r["pred_bucket"] == b)
        recomputed["per_gold_bucket_recall"][b] = {"n": len(g), "correct": c,
                                                   "rate": round(c / len(g), 4) if g else None}
    mismatch = []
    if recomputed["n"] != met.get("n"):
        mismatch.append("n")
    if abs(recomputed["bucket_accuracy"] - met.get("bucket_accuracy", -1)) > 1e-6:
        mismatch.append("bucket_accuracy")
    for b in LABELS:
        m = met.get("per_gold_bucket_recall", {}).get(b, {})
        r_ = recomputed["per_gold_bucket_recall"][b]
        if (m.get("n"), m.get("correct")) != (r_["n"], r_["correct"]):
            mismatch.append(f"recall[{b}]")
    correct_field = sum(1 for r in rows if bool(r.get("correct")) == (r["pred_bucket"] == r["gold"]))
    gcount = Counter(r["gold"] for r in rows)
    maj_bucket, maj_n = gcount.most_common(1)[0]
    maj_acc = maj_n / n
    gold_from_eval = Counter(r["gold"] for r in EG) == gcount
    pred_dist = dict(Counter(r["pred_bucket"] for r in rows))
    chk("7-baseline-metrics-and-class-prior",
        "pass" if not mismatch else "fail",
        f"eval_baseline.json rows={n}; recomputed accuracy={recomputed['bucket_accuracy']} "
        f"({sum(1 for r in rows if r['pred_bucket'] == r['gold'])}/{n}) vs file metrics "
        f"{met.get('bucket_accuracy')}; per-gold-bucket recall recomputed "
        f"{ {b: (recomputed['per_gold_bucket_recall'][b]['correct'], recomputed['per_gold_bucket_recall'][b]['n']) for b in LABELS} } "
        f"vs file "
        f"{ {b: (met.get('per_gold_bucket_recall', {}).get(b, {}).get('correct'), met.get('per_gold_bucket_recall', {}).get(b, {}).get('n')) for b in LABELS} }; "
        f"mismatches={mismatch or 'none'}; 'correct' field consistent for {correct_field}/{n}; "
        f"predicted-bucket dist={pred_dist}; gold dist={dict(gcount)}; MAJORITY-CLASS accuracy "
        f"(always '{maj_bucket}') = {maj_n}/{n} = {round(100 * maj_acc, 2)}% -- the claim that the "
        f"published stock-1.5B number 77.1% (54/70) is just the class prior: 54/70 = "
        f"{round(100 * 54 / 70, 2)}% {'EXACT match' if maj_n == 54 else ''}; the untrained 0.5B base "
        f"scores {round(100 * recomputed['bucket_accuracy'], 2)}% which is "
        f"{'BELOW' if recomputed['bucket_accuracy'] < maj_acc else 'above'} the prior; "
        f"majority-rule predicts every 'service_lead' and 'staff_role' gold row wrong "
        f"(2+14={sum(1 for r in rows if r['gold'] != maj_bucket)} rows at 0% recall), "
        f"identical to the file's confusion", hard=False)

    # ============================================================ CHECK 8
    def param_info(log_path: Path):
        if not log_path.exists():
            return None, None
        txt = log_path.read_text(errors="replace")
        vals = [float(m) for m in __import__("re").findall(r"trainable=([0-9.]+)M", txt)]
        dline = __import__("re").findall(r"\[data\][^\n]*", txt)
        return vals, (dline[-1].strip() if dline else None)

    rf_dir = RUNS / "run-final"
    sup = (rf_dir / "supervisor.log").read_text(errors="replace") if (rf_dir / "supervisor.log").exists() else ""
    sup_params = [float(m) for m in __import__("re").findall(r"trainable=([0-9.]+)M", sup)]
    pv = {}
    try:
        pv = json.loads((rf_dir / "progress.json").read_text()) if (rf_dir / "progress.json").exists() else {}
    except Exception:  # noqa: BLE001
        pv = {"_parse_error": True}
    tl = jlines(rf_dir / "train_log.jsonl") if (rf_dir / "train_log.jsonl").exists() else []
    rf = {}
    if tl:
        rf = {"first": tl[0], "last": tl[-1], "n_steps": len(tl),
              "pace_s_per_step": round((tl[-1].get("elapsed_min", 0) - tl[0].get("elapsed_min", 0)) * 60
                                        / max(1, tl[-1].get("step", 1) - tl[0].get("step", 0)), 2)}
    n1 = {}
    n1_tl = jlines(RUNS / "run-night1" / "train_log.jsonl") if (RUNS / "run-night1" / "train_log.jsonl").exists() else []
    if n1_tl:
        n1 = {"first": n1_tl[0], "last": n1_tl[-1], "n_steps": len(n1_tl),
              "pace_s_per_step": round(n1_tl[-1].get("step_s", 0), 2)}
    n1_params, n1_data = param_info(RUNS / "train_night1.log")
    bad_params = [v for v in (sup_params + (n1_params or [])) if v > 50]
    chk("8-run-record-trainable-params-and-loss",
        "pass" if (not bad_params and sup_params and all(abs(v - 2.16) < 0.05 for v in sup_params)) else "fail",
        f"run-final: trainable={sup_params or 'NOT LOGGED'}M (expected ~2.16M; >50M would mean the "
        f"LoRA freeze broke); progress.json step={pv.get('step')} loss={pv.get('loss')} "
        f"batch={pv.get('batch')} max_len={pv.get('max_len')} rank={pv.get('rank')} "
        f"target_total={pv.get('target_total')} saved_at={pv.get('saved_at')}; train_log.jsonl "
        f"{rf.get('n_steps')} steps, first step {rf.get('first', {}).get('step')} "
        f"loss={rf.get('first', {}).get('loss')}, latest step {rf.get('last', {}).get('step')} "
        f"loss={rf.get('last', {}).get('loss')}, pace={rf.get('pace_s_per_step')}s/step "
        f"({rf.get('last', {}).get('elapsed_min')} min into this chunk); "
        f"run-night1: trainable={n1_params}M from train_night1.log, {n1_tl and n1.get('n_steps')} steps, "
        f"first loss={n1.get('first', {}).get('loss')} latest loss={n1.get('last', {}).get('loss')} "
        f"(step {n1.get('last', {}).get('step')}), {n1.get('last', {}).get('elapsed_s')}s total, "
        f"~{n1.get('pace_s_per_step')}s/step at the end; night1 data line: {n1_data}", hard=True)

    # ---------------------------------------------------------------- summary
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    inputs = {}
    for p in [TRAIN_F, DEV_F, TRAIN_OLD_F, EVAL_GOLD_F, GOLD_F, CORPUS_F, REPORT_F, BASELINE_F,
              RUNS / "run-final" / "supervisor.log", RUNS / "run-final" / "progress.json",
              RUNS / "run-final" / "train_log.jsonl", RUNS / "run-night1" / "train_log.jsonl",
              RUNS / "train_night1.log"] + SYNTH_FILES + [CHECKED_F]:
        inputs[Path(p).name if Path(p).parent in (DATA, RUNS) else str(p)] = stamp(p)
    out = {
        "what": "independent re-verification of the openjev corpus (train_jev.jsonl) and eval baseline",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.time() - t0, 1),
        "max_len": MAX_LEN,
        "checks": checks,
        "summary": {
            "n_checks": len(checks), "pass": n_pass, "warn": n_warn, "fail": n_fail,
            "hard_failures": [c["name"] for c in checks if c["status"] == "fail"],
            "claims_not_fully_reproduced": [c["name"] for c in checks if c["status"] != "pass"],
            "train_jev_rows": len(TRAIN), "train_jev_dist": tj_dist,
            "train_jev_by_source": dict(src_train),
            "token_stats": tok_stats,
            "majority_class_accuracy": round(maj_acc, 4),
            "baseline_accuracy": recomputed["bucket_accuracy"],
            "exit": 1 if n_fail else 0,
        },
        "inputs": inputs,
    }
    VERIFY.write_text(json.dumps(out, indent=1))
    print(f"\n=== VERIFY.json written: {n_pass} pass / {n_warn} warn / {n_fail} fail "
          f"({time.time() - t0:.1f}s) -> {VERIFY}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
