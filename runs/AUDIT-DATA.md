# AUDIT-DATA — independent audit of the openjev training corpus

Auditor: Subagent C (independent data auditor; no model loading, no GPU).
Script: `openjev/audit.py` (read-only; recomputes every number from the raw files, trusts no manifest).
Reproduce: `cd ~/Code/jev-repro-test && python3 openjev/audit.py --sample-n 25` (exit 1 on any hard failure).
Audit run stamped **2026-09-18 03:13:15 CST**; a validation re-run of the patched script at 03:24 returned **EXIT 1** (the one hard failure below).

## 0. Files as they existed when the audit ran

| file | bytes | mtime | lines |
|---|---|---|---|
| `openjev/data/train.jsonl` | 1,516,699 | 2026-09-18 03:04:19 | 2,437 |
| `openjev/data/dev.jsonl` | 137,218 | 2026-09-18 03:04:19 | 221 |
| `openjev/data/eval_gold.jsonl` | 44,061 | 2026-09-18 03:04:19 | 70 |
| `openjev/data/manifest.json` | 376 | 2026-09-18 03:04:19 | (report) |
| `openjev/synth/data/labelled.jsonl` | 1,580,760 | 2026-09-18 03:00:26 | 3,039 |
| `openjev/synth/data/jev_checked.jsonl` | 4,567,451 | 2026-09-18 03:04:11 | 3,039 |
| `openjev/synth/data/raw_postings.jsonl` | 698,861 | 2026-09-18 02:16:58 | 3,040 |
| `typesafe-lab/data/leads.gold.json` | 6,957 | 2026-09-16 02:24:08 | 70 labels |
| `typesafe-lab/runs/leads_corpus_full.json` | 5,805,583 | 2026-09-16 02:34:42 | 2,631 rows |

**`data/train_jev.jsonl`, `data/dev_jev.jsonl` and `data/jevlab/*.jsonl` were never created.** I polled for them
02:58→03:16 (450 s of watching, plus spot checks after) and they were still absent. So this audit covers what
actually exists — and what is actually trained on: `train_v2.py`'s default `--train-file` is `openjev/data/train.jsonl`,
`run_training.sh` passes no override, and the last completed run (`runs/smoke-v2/adapter_config.json`) records
`train_file = .../openjev/data/train.jsonl`. Every finding below is about that file and its dev sibling.

## 1. Leakage — 1 hard failure, 6 offending rows (0.25 % of train)

* gold ids: 70 in `leads.gold.json` ∪ 70 eval_gold ids (same set); all 70 resolve to corpus rows.
* **id leakage: 0** in train.jsonl, dev.jsonl, train/dev cross-split (no shared ids).
* **(title, employer) leakage: 6 rows in train.jsonl, 0 in dev.jsonl.** These train rows are *different posting ids*
  carrying the same case/whitespace-normalised (title, employer) as a held-out gold posting:

| train.jsonl line | train id | (title, employer) | gold id |
|---|---|---|---|
| 40 | 1539368 | Licensed Practical Nurse / Saskatchewan Health Authority | 1536255 |
| 1552 | 1537547 | Licensed Practical Nurse / Saskatchewan Health Authority | 1536255 |
| 226 | 1538999 | Information Technology Senior Analyst / Saskatchewan Health Authority | 1539681 |
| 1437 | 1543882 | Information Technology Senior Analyst / Saskatchewan Health Authority | 1539681 |
| 1664 | 1506692 | Medical Radiation Technologist - Specialty / Saskatchewan Health Authority | 1533170 |
| 1711 | 1537765 | Web Designer / PrintWest | 1544518 |

All six are real-corpus rows; `dataset.py` excludes gold **ids** but never gold **(title, employer) pairs**, so the
model is trained on the same role at the same employer it is later tested on, with Jev's label attached to a
different posting (which may carry a different bucket). With a 70-row eval, 6 train-side twins is the largest
single threat I found to the headline score.

## 2. Format integrity — clean except one prompt

* 100 % of lines parse as JSON in every file (0 parse errors).
* `train.jsonl` 2,437 rows / `dev.jsonl` 221 / `eval_gold.jsonl` 70: **0** target violations — every target is valid
  JSON, key order exactly `bucket, technical_need, business_buyer, small_firm_doable, pay_stated, evergreen_repost,
  fit`, bucket ∈ 4 legal values, `fit` an int in 0..4, all 5 booleans present as booleans.
* Prompt round-trip (`prompt == dataset.prompt_for(title, employer, pay)` parsed back out of the prompt):
  **1 violation in 2,437** — `train.jsonl` line 920, `'Retail Sales Associate - Contract'`. dev/eval_gold/labelled/jev_checked: 0.
* `labelled.jsonl` and `jev_checked.jsonl` (3,039 rows each): structurally clean.

## 3. Label provenance

* `train.jsonl` / `dev.jsonl` / `eval_gold.jsonl` carry **no `jev` block at all** (2,728 rows): per-row provenance is
  not auditable from the corpus file itself — it exists only in `synth/data/jev_checked.jsonl` and
  `leads_corpus_full.json`. Flagged as warning, not failure. (`merge_corpus.py` would have embedded it; it never ran.)
* I therefore cross-checked independently: re-normalised each source row's raw Jev answers
  (`noul`/`probability` → ≥0.5, `fit.score`) exactly as `dataset.py` does, and compared `dataset.target_for(...)`
  byte-for-byte with the stored target:
  * `train.jsonl`: **2,433 / 2,437 reproduce Jev's source answers exactly**; 4 flagged; 0 unresolved.
  * `dev.jsonl`: 221 / 221. `eval_gold.jsonl`: 70 / 70.
  * The 4 flags are duplicate-key collisions in my source index (e.g. `Sales Representative / Stonegate Housing
    Society` exists twice with different Jev labels), i.e. evidence of source rows that differ only in their answers,
    not of target corruption. No row anywhere had an out-of-range probability, an illegal bucket, a missing boolean
    or a target/jev disagreement.
* Headline provenance story: 2,595 of the 3,039 synthetic rows (85.4 %) were labelled by Jev; the other 444 were
  dropped for teacher/Jev bucket disagreement before training. Nothing in the corpus is teacher-labelled-only.

## 4. Distribution and trainability — **still skewed; two classes are untrainable**

| file | n | service_lead | staff_role | generic_job | junk |
|---|---|---|---|---|---|
| `train.jsonl` | 2,437 | **12 (0.5 %)** | 663 (27.2 %) | 1,713 (70.3 %) | **49 (2.0 %)** |
| `dev.jsonl` | 221 | 4 (1.8 %) | 67 (30.3 %) | 145 (65.6 %) | 5 (2.3 %) |
| real corpus (all 2,631) | 2,631 | 1 | 51 | 2,579 | 0 |

* Sources in `train.jsonl`: **synth 2,380 (97.7 %) / real 57 (2.3 %)** — and 26 of those 57 are the capped
  `generic_job` real rows. The real generic_job cap outcome: `real_generic_total = 2,524`, `real_generic_kept = 26`,
  `real_nongeneric = 37`, so **2,498 real generic postings (99 %) were left on the floor** by
  `--max-generic-share 0.42`; the corpus contains ~43× more synthetic Regina-shaped rows than real ones.
* Verdict on mix: **not trainable as a 4-class problem.** `service_lead` (12) and `junk` (49) are each below both
  bars (≥5 %, ≥50 examples); `generic_job` at 70.3 % dominates. In practice this corpus teaches a binary
  generic vs staff_role decision plus a handful of memorised outliers, and no metric computed on the 70-row
  eval can distinguish a real service_lead/junk ability from 12/49 training instances.

## 5. Duplicates and label contradictions — small but real

* `train.jsonl`: 8 exact-normalised `(title, employer)` duplicate groups (8 redundant rows); 0 of them carry
  contradicting buckets. `dev.jsonl`: 0.
* Near-duplicates (difflib ≥0.90 on normalised `title||employer`, 57,778 blocked comparisons, not capped):
  **43 pairs, of which 1 contradicts itself** —
  `'project coordinator || legacy contracting' → generic_job` vs `'it coordinator || legacy contracting' → staff_role`.
  Same employer, near-identical title, opposite label: that is a noise example the model has no way to resolve.
* Same-title/different-employer groups carrying more than one bucket: 11 in train, 2 in dev — expected
  (different employers legitimately differ), reported for completeness, not a defect.

## 6. Realism read — 25 seeded synthetic rows (seed 11), judged as a SaskJobs reader

Verdict: the rows read as **competent template postings, not real ones**. Job titles, pay formats and tactic
markers (Talent Pool, URGENT, Part-Time, bilingual `Caissier(ère)`) are convincing; employer names come from a
small invented pool (`Legacy`, `Aspen`, `Cedar Ridge`, `Stonegate`, `Pinewood`, `Northgate`, `Sask …` + one of
`Ltd./Services/Group/Corporation/Housing Society/Staffing Solutions`) and the role↔employer pairing is random,
which is where the realism leaks.

**5 best (plausible as-is on SaskJobs):**

1. `Information Security Officer - Talent Pool | Prairie Sky Health Region | $41.25 hourly | staffing bucket=staff_role`
2. `Programmer Analyst (Part-Time) | City of Estevan | (no pay) | staff_role`
3. `Data Analyst | Prairie Qu'Appelle Group | $33.00 hourly | staff_role`
4. `Cashier / Caissier(ère) | Moose Jaw Housing Society | $15.00 hourly | generic_job`
5. `Part-Time Systems Administrator | Qu'Appelle Sales Partners | $28.00 - $36.00/hr | staff_role`

**5 worst (template mush):**

1. `Business Intelligence Analyst | Team Cedar Ridge Marketing | Competitive earnings potential | staff_role` — invented "Team X Marketing" employer, and a pay line no Sask employer posts for a BI analyst.
2. `Full Stack Developer | Pinewood School Division | (no pay) | staff_role` — role/sector mismatch.
3. `Graphic Designer (Overnight Shift) | Lakeshore Community Association | Commensurate with experience | generic_job` — overnight graphic design does not exist.
4. `107410 - Territory Manager | Aspen Staffing Solutions | (no pay) | generic_job` — a numeric job-order prefix (real-corpus scrape artifact) grafted onto a synthetic employer.
5. `Bylaw Officer | Deer Valley Recruitment Group | $64,000 annually | generic_job` — a staffing agency as bylaw-officer employer; also the only sampled row where a stated salary coexists with a `generic_job` label, which invites the model to learn "pay stated ⇒ generic".

(Runner-up worth fixing: `Change Management Analyst | Regina Beach Public Library | $65,000 - $78,000 annually | generic_job` — a village library paying 78 k.)

## 7. Teacher vs Jev agreement

`jev_checked.jsonl`: 3,039 rows labelled by both → **2,595 agree / 444 disagree = 85.4 % bucket agreement**
(exactly matching `dataset.py`'s own stats, independently recomputed). Meaning: **444 rows (14.6 %) were dropped by
the merge for disagreement**, and a teacher-labelled corpus would have differed from Jev on ~1 in 7 bucket labels.
Bucket-level agreement is the only thing the merge enforced; the five booleans and `fit` were never cross-checked
between teacher and Jev.

## Concrete fixes, ranked by impact

1. **Close the 6 gold-pair leaks** (0.25 % of train): teach `dataset.py` to exclude by normalised `(title, employer)`
   as well as by id (the same key `merge_corpus.py` already dedupes on). Re-run before any reported eval number.
2. **Fix the class mix before spending more GPU**: `service_lead` 12 and `junk` 49 cannot be learned or measured.
   Generate/label ≥150 targeted rows of each (the generator families exist) or report the experiment as a
   2-class result and drop the other two columns from the claims.
3. **Raise the real-data share above ~2 %**: 2,498 real generic postings were capped away. Either raise
   `--max-generic-share`, or train a real-only arm; otherwise "reproduce Jev on Regina postings" is measured
   against a corpus that is 97.7 % generator output.
4. **Dedupe in `dataset.py`**: port `merge_corpus.py`'s `(title, employer)` dedupe; it removes 8 exact duplicates,
   and manually resolve the one contradicting near-duplicate pair
   (`project coordinator`/`it coordinator` @ `Legacy Contracting`, generic_job vs staff_role).
5. **Generator realism pass**: widen the employer-name pool (≈16 stems visible in 25 rows), constrain role↔sector
   pairing, strip numeric job-order prefixes, and vet pay strings ("Competitive earnings potential" reads like a
   commission ad, not a posting).
6. **Fix the one prompt round-trip violation** (`train.jsonl` line 920, `'Retail Sales Associate - Contract'`);
   it is a whitespace/clip edge case in `clip()` that also affects `prompt == prompt_for(...)` for that row.

**Not blocking, but worth knowing:** `train_jev.jsonl`/`dev_jev.jsonl` do not exist, so the corpus audited here is
the one `dataset.py` wrote at 03:04 — the trainer default. If a `merge_corpus.py` corpus is produced later it must be
re-audited (`python3 openjev/audit.py`; it checks both filename families automatically).
