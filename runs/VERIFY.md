# VERIFY.md — independent re-verification of the openjev corpus + eval baseline

Subagent E (independent verifier), 2026-09-18 ~04:55. Script: `openjev/verify_corpus.py`
(runnable, plain Python, exits non-zero if any HARD check fails). Raw result:
`openjev/runs/VERIFY.json` — **13 checks: 13 pass / 0 warn / 0 fail, exit 0, ~10 s**.
No model weights were loaded (check 6 uses `AutoTokenizer` only). Every input file is
hashed (sha256[:12] + mtime) in VERIFY.json's `inputs` block.

## Verdict

Every claim I was asked to check **reproduced from the raw files**. `data/train_jev.jsonl` is
source-traceable, leak-free, deduped and fits max_len=192; the eval baseline's metrics block
matches its own rows; the training run's logged trainable-param count is 2.16 M (LoRA freeze intact).

| # | check | status | recomputed numbers |
|---|---|---|---|
| 1 | target provenance | pass (HARD) | 2591/2591 rows have a `jev` block, target == `target_for(row.jev)` byte-identically, 2591/2591 also byte-present in the raw JSONL line |
| 1d | jev block vs the claimed source record | pass (HARD) | 2591/2591 re-derive from the source file's Jev answers (`answers` in `leads_corpus_full.json`, `jev` in the jevlab files) with the prompt text identical too; 0 mismatches, 0 unresolvable ids |
| 2 | gold leakage in train_jev | pass (HARD) | 0 id leaks, 0 pair-leaked rows vs 70 gold ids / 65 distinct (title,employer) keys (raw and clip-90 variants) |
| 3 | gold set integrity | pass (HARD) | eval_gold.jsonl 70 rows, 0 duplicates, `gold` == `leads.gold.json` for 70/70, id sets identical, 0 overlap with train_jev (ids and pairs) |
| 4a | class balance vs corpus_jev_report.json | pass | generic_job 1088 (42.0%), staff_role 958 (37.0%), service_lead 342 (13.2%), junk 203 (7.8%); by_source real 1118 / labelled 797 / synth_hard 676 — all identical to the report; dev 286 rows as reported |
| 4b | Agent-1 train.jsonl claims | pass | n=2437; generic_job 1713 (70.3%), staff_role 663 (27.2%), service_lead 12 (0.5%), junk 49 (2.0%); 97.7% synthetic; gold pair leaks = 6 rows / 4 distinct pairs → AUDIT-DATA.md's "6 rows" reproduced |
| 4c | drop-counter reconciliation | pass | see below — closes exactly |
| 5 | dedupe | pass (HARD) | 0 duplicate (title,employer) pairs, 0 duplicate ids, 0 identical prompts, 0 empty titles |
| 6 | prompt+target vs max_len=192 | pass (HARD) | all 2591 rows tokenized: min 116 / median 124 / mean 125.5 / p95 137 / max 151 tokens; **0 rows > 192**; longest target 52 tokens; first-40 subset min 118 / median 125.5 / max 144 |
| 7 | eval baseline metrics + class prior | pass | accuracy 51/70 = 0.7286 (file says 0.7286); recall service_lead 0/2, staff_role 0/14, generic_job 51/54, junk 0/0 — matches the metrics block exactly; `correct` field consistent for 70/70 |
| 8 | run record | pass (HARD) | run-final trainable=2.16 M, run-night1 trainable=2.16 M (both OK; >50 M is the refuse-to-train case) |

## What I recomputed, and the two things worth knowing

* **Source mix (check 1c).** train 2591 = real 1118 + labelled 797 + synth_hard 676; dev 286 = real
  122 + labelled 83 + synth_hard 81. So real = 1240/2877 = 43.1% counting dev, 43.1% of the trainable
  bake-off; synthetic 56.9%. No row lacks Jev answers.
* **Drop accounting (check 4c) closes exactly** against the current source files: usable rows
  4959 synth (3039 labelled + 960 + 960 synth_hard files) − 968 `dupe:synth` = 3991; 3039 checked −
  3039 `dupe:checked` = 0; real 2561 − 231 gold-pair − 723 dupe = 1607. Total pre-trim 5598 ==
  actual (train 2591 + dev 286 + generic-trim 2721). Post-trim by source: real 1240, labelled 880,
  synth_hard 757. The generic-share rule `int(non·0.42/0.58) = 1208` equals the generic rows kept.
* **merge_corpus.py comment ≠ code (no number affected).** The global generic trim says
  `# drop the easy real generic first` but the code sorts reals first and keeps `gen[:target_gen]`,
  i.e. it *keeps* real generic rows: derived trim split = 367 real + 2354 synthetic, and in the data
  **all 1208 kept generic_job rows are `real`** (no synthetic generic row survived). Worth a comment
  fix; the drop counters themselves are correct.
* **Balance fix confirmed (check 4a).** service_lead 12 → 342 (×28.5), junk 49 → 203 (×4.1),
  generic_job 70.3% → 42.0%. The starvation claim in AUDIT-DATA.md is fixed in `train_jev.jsonl`.
* **Baseline number is the class prior (check 7).** Majority-class accuracy on the same 70 gold rows
  (always `generic_job`) = 54/70 = **77.14%**, i.e. the published stock-1.5B "77.1% (54/70)" is
  *exactly* the prior, not a skill floor. The untrained 0.5B base in `eval_baseline.json` scores
  51/70 = 72.86%, **below** the prior, and never predicts service_lead or junk (pred dist 67
  generic_job / 3 staff_role) — so the base model adds nothing over the majority rule.
  Context (read-only, recomputed from its rows for accuracy): `eval_snap0405.json` = 90.0% (63/70,
  pred dist 61 generic_job / 9 staff_role) for the step-40 adapter trained on the *old* corpus.
* **Run pace (check 8).** run-final: target_total 400, batch 6, max_len 192, rank 16; loss 0.6792 at
  step 1 → 0.0331 at step 51, ~9.0 s/step in the live chunk. run-night1: 420 steps, loss 0.6110 →
  0.0045 in 89.7 min (24.4 tok/s), dev target-token loss 0.0127.
* **Live-file caveat.** `runs/run-final/` advanced while I read it (progress.json step 50, train_log
  51 steps); the hashes in VERIFY.json fix the exact snapshot I verified. Everything else was static.

## Claims I could not reproduce

None of the eight checks failed. Three framing notes:

1. The brief says "796 synthetic postings from `labelled.jev.jsonl`"; the artifact has **797** train
   rows (880 including dev) from that source, and `corpus_jev_report.json` agrees with 797 — the
   brief is off by one, the file is not.
2. AUDIT-DATA.md's "6 rows" of gold (title,employer) leakage in `train.jsonl` is **6 leaked rows
   sharing 4 distinct pairs** (2 pairs matched by 2 rows each): `information technology senior
   analyst / saskatchewan health authority`, `licensed practical nurse / saskatchewan health
   authority`, `medical radiation technologist - specialty / saskatchewan health authority`,
   `web designer / printwest`. Reproduced; it is a rows-vs-pairs counting artifact, not a conflict.
3. The trainer's own log line (`mean tokens 126 p95=137`, live from train_v2.py) matches my
   independent tokenization (mean 125.5, p95 137) — two independent implementations agree.
