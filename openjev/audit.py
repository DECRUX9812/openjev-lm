#!/usr/bin/env python3
"""openjev/audit.py -- independent audit of the openjev training corpus (Subagent C).

READ-ONLY: this script never writes, renames or rewrites anything. Every number is
recomputed from the raw files; no manifest/summary file is trusted.

Checks
  1. leakage        -- gold ids / (title, employer) pairs must not appear in train or dev
  2. format         -- JSON validity, target byte-identical to dataset.target_for(jev),
                       prompt byte-identical to dataset.prompt_for(parsed title/employer/pay),
                       legal bucket, fit 0..4, all 5 booleans present, key order
  3. provenance     -- jev block present, plausible ranges, target agrees with jev probs
  4. distribution   -- buckets / sources / real generic_job cap, trainability verdict
  5. duplicates     -- exact + near-duplicate groupings, contradicting bucket labels
  6. realism        -- seeded sample of synthetic rows printed for the human read
  7. teacher vs Jev -- bucket agreement rate on rows carrying both

usage:
  python openjev/audit.py [--seed 11] [--sample-n 25] [--cap 250000] [--quiet-samples]

exit code: 0 = no hard failure, 1 = hard failure, 2 = nothing auditable found.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import BOOLS, prompt_for, target_for  # noqa: E402  single source of truth

LAB = Path("/home/decrux/Code/typesafe-lab")
DATA = HERE / "data"
SYNTH = HERE / "synth" / "data"
RUNS = HERE / "runs"

LEGAL = ("service_lead", "staff_role", "generic_job", "junk")
KEY_ORDER = ["bucket", *BOOLS, "fit"]
END = "<|im_end|>\n"

HARD: list[str] = []
SOFT: list[str] = []
MISSING: list[str] = []


def hard(msg: str) -> None:
    HARD.append(msg)


def soft(msg: str) -> None:
    SOFT.append(msg)


def nkey(s) -> str:
    return " ".join(str(s if s is not None else "").lower().split())


def nopunct(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", nkey(s))).strip()


def load_jsonl(path: Path):
    """-> (rows, parse_errors). Missing file -> (None, []) and recorded in MISSING."""
    if not path.exists():
        MISSING.append(str(path))
        return None, []
    rows, errs = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errs.append((i, str(exc), line[:120]))
    return rows, errs


def fstat(path: Path) -> str:
    if not path.exists():
        return f"MISSING  {path}"
    st = path.stat()
    try:
        n = sum(1 for _ in path.open("rb"))
    except OSError:
        n = -1
    return (f"{path}  size={st.st_size}  mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}"
            f"  lines={n}")


# ---------------------------------------------------------------- prompt/target


def parse_prompt(prompt: str):
    """Parse title/employer/pay out of a compact prompt, in order. -> (t, e, pay, ok_struct)"""
    lines = str(prompt or "").split("\n")
    t = e = pay = None
    ok = True
    for i, ln in enumerate(lines):
        if ln.startswith("title: "):
            if t is not None:
                ok = False
            t = ln[7:]
            if i + 1 < len(lines) and lines[i + 1].startswith("employer: "):
                e = lines[i + 1][10:]
                if i + 2 < len(lines) and lines[i + 2].startswith("pay: "):
                    pay = lines[i + 2][5:]
                else:
                    ok = False
            else:
                ok = False
            break
    if t is None or e is None or pay is None:
        ok = False
    return t, e, pay, ok


def row_te(row: dict):
    t = row.get("title")
    e = row.get("employer")
    if t is None and row.get("prompt"):
        t, e, _, _ = parse_prompt(row["prompt"])
    return t, e


def bucket_of(row: dict):
    j = row.get("jev")
    if isinstance(j, dict):
        b = j.get("bucket")
        if isinstance(b, dict):
            b = b.get("choice")
        if b:
            return b
    if row.get("bucket"):
        return row["bucket"]
    tgt = row.get("target")
    if isinstance(tgt, str) and tgt.strip():
        try:
            return json.loads(tgt[: tgt.rfind(END)] if tgt.endswith(END) else tgt).get("bucket")
        except Exception:
            return None
    return None


def num_of(v):
    """Raw Jev answer dialect -> (float, kind) or (None, None)."""
    if isinstance(v, bool):
        return (1.0 if v else 0.0, "bool")
    if isinstance(v, (int, float)):
        return float(v), "num"
    if isinstance(v, dict):
        for k in ("noul", "probability", "prob", "score"):
            if isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool):
                return float(v[k]), "num"
        if "choice" in v:
            return None, "choice"
    return None, None


# ---------------------------------------------------------------- row checks


def check_rows(name: str, rows: list, hard_cap: int = 12, expect_target: bool = True):
    """Format + provenance for one JSONL file. Returns counters."""
    c = Counter()
    bad_examples: dict[str, list] = defaultdict(list)

    def note(kind, i, row, detail=""):
        c[kind] += 1
        if len(bad_examples[kind]) < hard_cap:
            bad_examples[kind].append((i, row.get("id"), (row_te(row)[0] or "")[:50], detail))

    for i, row in enumerate(rows, 1):
        c["rows"] += 1
        has_prompt = isinstance(row.get("prompt"), str) and row["prompt"]
        has_target = isinstance(row.get("target"), str) and row["target"] != ""
        jev = row.get("jev")

        if not isinstance(row, dict):
            note("not_object", i, {}, "")
            continue
        if "target" in row and not has_target and expect_target:
            note("empty_target", i, row, repr(row.get("target"))[:40])

        # ---- target structure
        if has_target:
            tgt = row["target"]
            if not tgt.endswith(END):
                note("target_no_im_end", i, row, repr(tgt[-20:]))
            body = tgt[: -len(END)] if tgt.endswith(END) else tgt
            try:
                obj = json.loads(body)
            except Exception as exc:  # noqa: BLE001
                note("target_not_json", i, row, str(exc)[:60])
                obj = None
            if isinstance(obj, dict):
                if list(obj.keys()) != KEY_ORDER:
                    note("target_key_order", i, row, str(list(obj.keys()))[:120])
                if obj.get("bucket") not in LEGAL:
                    note("target_bucket_illegal", i, row, repr(obj.get("bucket")))
                for b in BOOLS:
                    if b not in obj or not isinstance(obj[b], bool):
                        note("target_bool_missing_or_nonbool", i, row, b)
                        c["bools_total"] += 1
                fit = obj.get("fit")
                if not isinstance(fit, int) or isinstance(fit, bool) or not (0 <= fit <= 4):
                    note("target_fit_out_of_range", i, row, repr(fit))
            else:
                c["bools_total"] += 1

        # ---- prompt round-trip
        if has_prompt:
            t, e, pay, ok = parse_prompt(row["prompt"])
            if not ok:
                note("prompt_structure", i, row, "title/employer/pay lines not in order")
            elif prompt_for(t, e, pay) != row["prompt"]:
                note("prompt_roundtrip", i, row, "prompt_for(parsed) != stored prompt")
            for b in BOOLS:
                pass  # deliberately nothing: bools are not in the prompt
            if "bucket(service_lead|staff_role|generic_job|junk)" not in row["prompt"]:
                note("prompt_keys_missing", i, row, "")

        # ---- provenance: the jev block
        if jev is None:
            if has_target:
                note("jev_block_absent", i, row, "no embedded jev block (provenance not auditable per-row)")
            continue
        if not isinstance(jev, dict):
            note("jev_not_object", i, row, type(jev).__name__)
            continue

        b = jev.get("bucket")
        if isinstance(b, dict):
            if b.get("choice") not in LEGAL:
                note("jev_bucket_illegal", i, row, repr(b.get("choice")))
            conf = b.get("confidence")
            if conf is not None and not (isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0):
                note("jev_bucket_confidence_range", i, row, repr(conf))
            probs = b.get("probabilities")
            if isinstance(probs, dict):
                vals = [float(v) for v in probs.values() if isinstance(v, (int, float))]
                if any(not (0.0 <= v <= 1.0) for v in vals):
                    note("jev_bucket_prob_range", i, row, "")
                if vals and abs(sum(vals) - 1.0) > 0.02:
                    note("jev_bucket_prob_sum", i, row, f"sum={sum(vals):.3f}")
            # plain-dict bucket (merged rows)
        elif b not in LEGAL:
            note("jev_bucket_illegal", i, row, repr(b))

        target_obj = None
        if has_target:
            body = row["target"][: -len(END)] if row["target"].endswith(END) else row["target"]
            try:
                target_obj = json.loads(body)
            except Exception:  # noqa: BLE001
                target_obj = None

        all_bool_shape = True
        for k in BOOLS:
            if k not in jev or jev[k] is None:
                note("jev_bool_missing", i, row, k)
                all_bool_shape = False
                continue
            v = jev[k]
            if isinstance(v, bool):
                continue
            all_bool_shape = False
            val, kind = num_of(v)
            if val is None:
                note("jev_bool_unparseable", i, row, f"{k}={v!r}"[:60])
                continue
            if not (0.0 <= val <= 1.0):
                note("jev_bool_range", i, row, f"{k}={val}")
            if isinstance(target_obj, dict) and target_obj.get(k) is not None:
                if bool(target_obj[k]) != (val >= 0.5):
                    note("target_disagrees_with_jev_prob", i, row, f"{k}: target={target_obj[k]} p={val}")

        # probability mirrors written by label_jev.py
        for k in BOOLS:
            pk = k + "_p"
            if pk in jev and isinstance(jev[pk], (int, float)):
                p = float(jev[pk])
                if not (0.0 <= p <= 1.0):
                    note("jev_prob_range", i, row, f"{pk}={p}")
                if isinstance(jev.get(k), bool) and jev[k] != (p >= 0.5):
                    note("jev_bool_vs_prob", i, row, f"{k}={jev[k]} p={p}")

        fitv = jev.get("fit")
        if fitv is None:
            note("jev_fit_missing", i, row, "")
        else:
            fv, kind = num_of(fitv)
            if fv is None:
                note("jev_fit_unparseable", i, row, repr(fitv)[:40])
            elif not (0.0 <= fv <= 4.0):
                note("jev_fit_range", i, row, f"fit={fv}")

        # byte-identical target == target_for(jev) when jev is the normalized bool shape
        if has_target and all_bool_shape and isinstance(jev.get("fit"), (int, float)) and target_obj is not None:
            try:
                if target_for(jev) != row["target"]:
                    note("target_not_byte_identical", i, row, "")
            except Exception as exc:  # noqa: BLE001
                note("target_recompute_error", i, row, str(exc)[:60])

    return c, bad_examples


def print_checks(name: str, c: Counter, bad: dict, path: Path, legacy: bool = False):
    print(f"--- {name}: {c['rows']} rows")
    SOFT_ONLY = {"jev_block_absent", "prompt_keys_missing", "prompt_roundtrip"}
    keys = [k for k in c if k not in ("rows", "bools_total")]
    if not keys:
        print("    clean")
    for k in sorted(keys):
        print(f"    {k}: {c[k]}")
        for ex in bad.get(k, [])[:6]:
            print(f"       e.g. line {ex[0]} id={ex[1]} title={ex[2]!r} {ex[3]}")
    for k in keys:
        (soft if (legacy or k in SOFT_ONLY) else hard)(f"{name}: {k} x{c[k]}")


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sample-n", type=int, default=25)
    ap.add_argument("--cap", type=int, default=250_000, help="max near-dup pair comparisons")
    args = ap.parse_args()

    print("=" * 78)
    print("openjev corpus audit (independent, read-only) ", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 78)

    train_path = DATA / "train_jev.jsonl"
    dev_path = DATA / "dev_jev.jsonl"
    v1_train, v1_dev = DATA / "train.jsonl", DATA / "dev.jsonl"
    jevlab_files = sorted((DATA / "jevlab").glob("*.jsonl")) if (DATA / "jevlab").exists() else []

    files = [train_path, dev_path, v1_train, v1_dev, DATA / "eval_gold.jsonl",
             SYNTH / "labelled.jsonl", SYNTH / "jev_checked.jsonl", SYNTH / "raw_postings.jsonl"] + jevlab_files
    files += [LAB / "data" / "leads.gold.json", LAB / "runs" / "leads_corpus_full.json"]
    print("\n[0] files")
    for p in files:
        print("   ", fstat(p))
    if not train_path.exists():
        soft(f"PRIMARY CORPUS MISSING: {train_path} (audit fell back to what exists)")
        print("    !! train_jev.jsonl absent -- auditing what exists, per brief")

    # ---------------- formats ----------------
    print("\n[1/2] format + provenance per file")
    per_file = {}
    for p in [train_path, dev_path, v1_train, v1_dev] + jevlab_files + [DATA / "eval_gold.jsonl", SYNTH / "labelled.jsonl", SYNTH / "jev_checked.jsonl"]:
        rows, errs = load_jsonl(p)
        if rows is None:
            print(f"--- {p.name}: MISSING")
            continue
        if errs:
            print(f"--- {p.name}: {len(errs)} JSON PARSE ERRORS")
            for ln, msg, frag in errs[:5]:
                print(f"       line {ln}: {msg} :: {frag!r}")
            hard(f"{p.name}: {len(errs)} unparseable lines")
        c, bad = check_rows(p.name, rows, expect_target=(p != DATA / "eval_gold.jsonl"))
        per_file[p.name] = c
        print_checks(p.name, c, bad, p, legacy=(p in (v1_train, v1_dev)))

    # ---------------- leakage ----------------
    print("\n[3] leakage")
    gold, _ = None, None
    if (LAB / "data" / "leads.gold.json").exists():
        gold = json.loads((LAB / "data" / "leads.gold.json").read_text())
    gold_ids = set(gold["labels"]) if gold else set()
    eval_rows, _ = load_jsonl(DATA / "eval_gold.jsonl")
    eval_rows = eval_rows or []
    eval_ids = {r.get("id") for r in eval_rows}
    print(f"    held-out ids: gold {len(gold_ids)} | eval_gold {len(eval_ids)} | union {len(gold_ids | eval_ids)}")

    corpus = []
    if (LAB / "runs" / "leads_corpus_full.json").exists():
        corpus = json.loads((LAB / "runs" / "leads_corpus_full.json").read_text())["rows"]
    by_id = {r["id"]: r for r in corpus}
    gold_pairs = {}
    for gid in gold_ids:
        r = by_id.get(gid)
        if r is not None:
            gold_pairs[(nkey(r.get("title")), nkey(r.get("employer")))] = gid
    for r in eval_rows:
        t, e = row_te(r)
        gold_pairs[(nkey(t), nkey(e))] = r.get("id")
    print(f"    gold postings resolvable in corpus: {sum(1 for g in gold_ids if g in by_id)}/{len(gold_ids)}"
          f" | gold (title,employer) keys: {len(gold_pairs)}")

    train_rows, _ = load_jsonl(train_path)
    dev_rows, _ = load_jsonl(dev_path)
    train_rows, dev_rows = train_rows or [], dev_rows or []
    cur_train, _ = load_jsonl(v1_train)      # data/train.jsonl -- train_v2.py's DEFAULT --train-file
    cur_dev, _ = load_jsonl(v1_dev)
    cur_train, cur_dev = cur_train or [], cur_dev or []
    primaries = {"train_jev.jsonl": train_rows, "dev_jev.jsonl": dev_rows,
                 "train.jsonl": cur_train, "dev.jsonl": cur_dev}

    def clip90(s):
        return " ".join(str(s if s is not None else "").split())[:90]

    def ckey(t, e):
        return (nkey(clip90(t)), nkey(clip90(e)))

    def answers_from(d):
        """Any Jev answer dialect -> {bucket, 5 bools, fit} exactly as dataset.py normalises it."""
        if not isinstance(d, dict):
            return None
        b = d.get("bucket")
        if isinstance(b, dict):
            b = b.get("choice")
        if b not in LEGAL:
            return None
        out = {"bucket": b}
        for k in BOOLS:
            fv, _k = num_of(d.get(k))
            if fv is None:
                return None
            out[k] = bool(fv >= 0.5)
        fit = d.get("fit")
        if isinstance(fit, dict):
            fit = fit.get("score")
        fv, _k = num_of(fit)
        if fv is None:
            return None
        out["fit"] = float(min(4.0, max(0.0, fv)))
        return out

    def leak_check(name, rows):
        if not rows:
            print(f"    {name}: no rows to check")
            return 0
        id_hits, pair_hits = [], []
        for i, r in enumerate(rows, 1):
            if r.get("id") in gold_ids or r.get("id") in eval_ids:
                id_hits.append((i, r.get("id")))
            t, e = row_te(r)
            k = (nkey(t), nkey(e))
            if k in gold_pairs:
                pair_hits.append((i, r.get("id"), t, e, gold_pairs[k]))
        print(f"    {name}: id leaks {len(id_hits)} | (title,employer) leaks {len(pair_hits)}")
        for h in id_hits[:8]:
            print(f"       id leak line {h[0]} id={h[1]}")
        for h in pair_hits[:8]:
            print(f"       pair leak line {h[0]} id={h[1]} ({h[2]!r},{h[3]!r}) ~ gold {h[4]}")
        return len(id_hits) + len(pair_hits)

    leaked = 0
    for nm in ("train_jev.jsonl", "dev_jev.jsonl", "train.jsonl", "dev.jsonl"):
        leaked += leak_check(nm, primaries[nm])
    # cross-split leakage
    if train_rows and dev_rows:
        ti, di = {r.get("id") for r in train_rows}, {r.get("id") for r in dev_rows}
        tp = {(nkey(*row_te(r)),) for r in train_rows}
        dp = {(nkey(*row_te(r)),) for r in dev_rows}
        print(f"    train/dev id overlap: {len(ti & di)} | (title,employer) overlap: {len(tp & dp)}")
        if tp & dp:
            soft(f"train/dev (title,employer) overlap: {len(tp & dp)} keys")
        if ti & di:
            hard(f"train/dev id overlap: {len(ti & di)}")
    if leaked:
        hard(f"GOLD LEAKAGE: {leaked} offending rows")
    else:
        print("    no gold leakage detected in train_jev/dev_jev")
    if not train_rows or not dev_rows:
        soft("leakage check on train_jev/dev_jev not possible (file missing)")

    # ---------------- distribution ----------------
    print("\n[4] distribution")
    meta = {r["id"]: r for r in corpus}
    real_buckets = Counter()
    for r in corpus:
        a = r.get("answers") or {}
        b = a.get("bucket")
        b = b.get("choice") if isinstance(b, dict) else b
        real_buckets[b] += 1

    for name, rows in primaries.items():
        if not rows:
            continue
        d = Counter(bucket_of(r) for r in rows)
        n = sum(d.values())
        print(f"    {name}: n={n} " + "  ".join(f"{k}={d.get(k,0)} ({d.get(k,0)/n:.1%})" for k in LEGAL))
        srcs = Counter(r.get("source") or r.get("src") or ("real" if r.get("id") in meta else "?") for r in rows)
        print(f"       sources: {dict(srcs)}")
        if name.startswith("train"):
            worst = min(LEGAL, key=lambda k: d.get(k, 0))
            if d.get(worst, 0) < 50 or d.get(worst, 0) / n < 0.05:
                soft(f"{name}: class '{worst}' has {d.get(worst,0)} rows ({d.get(worst,0)/n:.1%}) -- under the 5% / 50-example bar")
            else:
                print(f"       balance OK: smallest class '{worst}' = {d[worst]} ({d[worst]/n:.1%})")
    ng_real = real_buckets.get("generic_job", 0)
    print(f"    real corpus buckets (all {len(corpus)} rows): {dict(real_buckets)}")
    if train_rows:
        real_train = [r for r in train_rows if r.get("source") == "real"]
        rg = sum(1 for r in real_train if bucket_of(r) == "generic_job")
        print(f"    real rows in train_jev: {len(real_train)} | of them generic_job: {rg} "
              f"| corpus generic_job total: {ng_real} -> cap dropping {ng_real - rg}")
    if (RUNS / "corpus_jev_report.json").exists():
        rep = json.loads((RUNS / "corpus_jev_report.json").read_text())
        print(f"    (builder's own report, NOT trusted: train={rep.get('train_rows')} dev={rep.get('dev_rows')} "
              f"drop={rep.get('dropped')} gold_leaks={rep.get('gold_leaks')})")

    # ---------------- duplicates ----------------
    print("\n[5] duplicates / contradictions")
    for name, rows in primaries.items():
        if not rows:
            continue
        ex = defaultdict(list)
        tight = defaultdict(list)
        loo = defaultdict(list)
        for idx, r in enumerate(rows, 1):
            t, e = row_te(r)
            ex[(nkey(t), nkey(e))].append((idx, r))
            tight[(nopunct(t), nopunct(e))].append((idx, r))
            loo[nopunct(t)].append((idx, r))
        ex_dup = {k: v for k, v in ex.items() if len(v) > 1}
        tight_dup = {k: v for k, v in tight.items() if len(v) > 1}
        contra = {k: v for k, v in ex_dup.items() if len({bucket_of(x[1]) for x in v}) > 1}
        contra_t = {k: v for k, v in tight_dup.items() if len({bucket_of(x[1]) for x in v}) > 1}
        loo_contra = {k: v for k, v in loo.items() if len({bucket_of(x[1]) for x in v}) > 1}
        print(f"    {name}: exact (title,employer) dupes {sum(len(v)-1 for v in ex_dup.values())} in {len(ex_dup)} groups"
              f" | punctuation-insensitive dupes {sum(len(v)-1 for v in tight_dup.values())} in {len(tight_dup)} groups"
              f" | same-title (any employer) groups with >1 bucket: {len(loo_contra)}/{len(loo)}")
        print(f"       contradictory buckets in exact-dup groups: {len(contra)}"
              f" | punctuation-insensitive: {len(contra_t)}")
        for k, v in list(contra_t.items())[:6]:
            print(f"       CONTRADICTION {k}: " + " | ".join(f"{bucket_of(x[1])} (line {x[0]} id={x[1].get('id')})" for x in v[:4]))
        if contra_t:
            hard(f"{name}: {len(contra_t)} near-duplicate groups carry contradicting bucket labels")
        elif contra:
            hard(f"{name}: {len(contra)} exact-duplicate groups carry contradicting bucket labels")
        if loo_contra:
            soft(f"{name}: {len(loo_contra)} same-title groups (different employers) carry >1 bucket")

        # near-duplicates by difflib, blocked by longest title token, capped
        blocks = defaultdict(list)
        for idx, r in enumerate(rows, 1):
            t, e = row_te(r)
            toks = nopunct(t).split()
            key = max(toks, key=len) if toks else ""
            blocks[key].append((idx, nopunct(t) + " || " + nopunct(e), bucket_of(r), r.get("id")))
        pairs = 0
        capped = False
        near = []
        for key, blk in blocks.items():
            if len(blk) < 2:
                continue
            for i in range(len(blk)):
                for j in range(i + 1, len(blk)):
                    if pairs >= args.cap:
                        capped = True
                        break
                    pairs += 1
                    a, b = blk[i][1], blk[j][1]
                    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
                    if sm.quick_ratio() < 0.9:
                        continue
                    if sm.ratio() >= 0.9:
                        near.append((blk[i], blk[j], sm.ratio()))
                if capped:
                    break
            if capped:
                break
        nc = [p for p in near if p[0][2] != p[1][2]]
        print(f"       near-dup pairs (>=0.90 similarity on title||employer): {len(near)} (compared {pairs} pairs"
              f"{', CAPPED' if capped else ''}) | of those with contradicting buckets: {len(nc)}")
        for p in nc[:6]:
            print(f"       NEAR-CONTRADICTION {p[2]:.2f} [{p[0][2]}] {p[0][1][:70]!r} (id={p[0][3]}) vs "
                  f"[{p[1][2]}] {p[1][1][:70]!r} (id={p[1][3]})")
        if nc:
            soft(f"{name}: {len(nc)} near-duplicate pairs with contradicting buckets")

    # ---------------- realism ----------------
    print(f"\n[6] realism sample: {args.sample_n} seeded synthetic rows")
    pool = [r for r in train_rows if r.get("source") != "real"] or \
           [r for r in cur_train if (r.get("src") or r.get("source")) != "real"] or \
           [r for r in ((load_jsonl(DATA / "jevlab" / "labelled.jev.jsonl")[0]) or [])] or \
           [r for r in ((load_jsonl(SYNTH / "jev_checked.jsonl")[0]) or [])]
    if pool:
        rng = random.Random(args.seed)
        pick = rng.sample(pool, min(args.sample_n, len(pool)))
        print(f"    pool={len(pool)} (sources: {dict(Counter(r.get('source') for r in pool))})")
        for i, r in enumerate(pick, 1):
            t, e = row_te(r)
            pay = ""
            if r.get("pay") is not None:
                pay = str(r.get("pay"))
            elif r.get("prompt"):
                _, _, pay, _ = parse_prompt(r["prompt"])
            print(f"    {i:2d}. {t!r} | {e!r} | pay={pay!r} | bucket={bucket_of(r)}"
                  f" | family={r.get('family') or r.get('stratum') or ''}")
    else:
        print("    no synthetic rows available to sample")

    # ---------------- teacher vs jev ----------------
    print("\n[7] teacher vs Jev agreement")
    for name, p in (("jev_checked.jsonl", SYNTH / "jev_checked.jsonl"), ("labelled.jsonl", SYNTH / "labelled.jsonl")) + \
            tuple((f.name, f) for f in jevlab_files):
        rows, _ = load_jsonl(p)
        if not rows:
            continue
        agree = dis = both = 0
        noj = 0
        for r in rows:
            tb = (r.get("label") or r.get("teacher") or {})
            tb = tb.get("bucket") if isinstance(tb, dict) else None
            jb = None
            j = r.get("jev")
            if isinstance(j, dict):
                b = j.get("bucket")
                jb = b.get("choice") if isinstance(b, dict) else b
            if tb and jb:
                both += 1
                agree += tb == jb
                dis += tb != jb
            elif tb and not jb:
                noj += 1
        if both or noj:
            print(f"    {name}: rows={len(rows)} both-labelled={both} agree={agree} "
                  f"disagree={dis}" + (f" ({agree/both:.1%} agreement)" if both else "")
                  + (f" | teacher-only (no Jev answer)={noj}" if noj else ""))

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print(f"HARD FAILURES: {len(HARD)}")
    for m in HARD:
        print("   X", m)
    print(f"WARNINGS: {len(SOFT)}")
    for m in SOFT:
        print("   !", m)
    print(f"MISSING FILES: {len(MISSING)}")
    for m in MISSING:
        print("   ?", m)
    if HARD:
        print("\nEXIT 1 -- hard check failures above")
        return 1
    if not train_rows and not cur_train:
        print("\nEXIT 2 -- no auditable corpus present")
        return 2
    print("\nEXIT 0 -- no hard failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
