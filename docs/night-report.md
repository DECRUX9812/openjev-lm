# openjev — can a 0.5B model reproduce Jev's judgment?

An experiment run overnight 2026-09-18 on the VM, by two Hermes sessions working the same box
under one model lock. The parent teardown of the viral `Qwen-2.5-1B-RLCD` artifact is in
`../REPORT.md`; this file is about the reproduction attempt that followed it.

## The question

Jev (TypeSafe's hosted decision model) takes a Regina job posting and returns a typed decision:
`bucket` (service_lead / staff_role / generic_job / junk), five booleans and a 0–4 `fit` score.
On 70 hand-labelled postings Jev agrees with the human labels 97.1% of the time. A stock
Qwen2.5-1.5B driven by the published parallel-constrained-decoding engine scored 77.1%.

**Does training a small open model on Jev's own answers transfer Jev's judgment?**

## Headline

| arm | what it is | bucket accuracy vs 70 hand labels | per-class recall (service_lead / staff_role / generic_job) |
|---|---|---|---|
| Jev itself | the hosted model, same 70 rows | **97.1%** | — |
| stock Qwen2.5-1.5B + PCD engine | published baseline | 77.1% | note: this equals the majority-class rate |
| Qwen2.5-0.5B, untrained | base model, same harness | 72.9% | 0/2 / 0/14 / 51/54 |
| run-night1 | 0.5B LoRA, 420 steps, **skewed** synthetic corpus | 90.0% (snapshot at ~step 260) | 0/2 / 9/14 / 54/54 |
| run-final | 0.5B LoRA, 400 steps, **balanced Jev-labelled** corpus | **92.9%** (65/70) | **1/2 / 12/14 / 52/54** |
| run-final (method B) | independent scorer, same adapter | **92.9%** (staged likelihood; agrees row-for-row with method A) | — |

The 77.1% reference needs a caveat that only appears when you look per class: **54 of the 70 gold
rows are `generic_job`**, so "always answer generic_job" scores 54/70 = 77.1% exactly — the
published 1.5B number is, to one decimal, the class prior it was measured against. The untrained
0.5B scores *below* the prior (72.9%) because its few non-generic answers are wrong, and its
recall on the classes that matter commercially (`staff_role` 0/14, `service_lead` 0/2) is zero.
That is the number the trained runs move.

## Corpus

Two corpora were built in parallel. Both are in `data/`; only one of each agent's corpora was
used per run, which makes the two runs a clean data-quality ablation.

| | `data/train.jsonl` (run-night1) | `data/train_jev.jsonl` (run-final) |
|---|---|---|
| rows | 2,437 | 2,591 (plus 286 dev) |
| synthetic share | 97.7% | 57% (797 generated + 676 hard-case + 1,118 real) |
| generic_job | 70.3% | 42.0% |
| service_lead | **12 rows (0.5%)** | **342 rows (13.2%)** |
| junk | 49 (2.0%) | 203 (7.8%) |
| label source | teacher model, Jev-verified on a subset, disagreements dropped (444) | **real Jev on every row** |
| gold (title, employer) pairs inside training data | 6 rows / 4 pairs | 0 |
| longest row (tokens) | — | 151 (`max_len` 192, nothing truncated) |

The hard-case set (`synth_hard.py` → `data/jevlab_hard/`) was written specifically against the
failure modes in the teardown: technical employee seats that read as plain jobs, genuine small
service needs, junk/commission postings, and confusable negatives. 960 postings, every one
labelled by real Jev: +364 `service_lead`, +221 `staff_role`, +172 `junk`, +203 `generic_job`;
intent-vs-Jev agreement 842/960 (88%).

Independent verification (`verify_corpus.py` → `runs/VERIFY.md`, 13/13 checks pass): every one of
the 2,591 training targets is byte-reproducibly Jev's own answer, 0 id leaks, 0 pair leaks, 0
duplicate pairs, gold set disjoint and internally consistent with `leads.gold.json`.

## Runs

Both runs: Qwen2.5-0.5B-Instruct, LoRA r=16 α=32 on q/k/v/o (2.16M trainable params),
batch 6, cross-entropy on answer tokens only, CPU only (6 vCPU, no CUDA on this box).

| | run-night1 | run-final |
|---|---|---|
| trainer | `train_lora.py` | `train_v2.py` + `run_training.sh` |
| steps | 420 | 400 |
| max_len / lr | 160 / 3e-4 | 192 / 1.5e-4 (OneCycle) |
| corpus | `data/train.jsonl` | `data/train_jev.jsonl` |
| final train loss | 0.0045 | 0.0468 (mean of last 10: 0.0253) |
| wall clock | 90 min | 89 min (3 supervised chunks, 04:18→05:48, CPU only) |

Two engineering notes worth keeping:

* **The 02:35 crash.** The host OOM'd and killed every Hermes child process. Root cause: `apply_lora()`
  had not frozen the base model, so two trainers were each optimising 452M parameters in fp32 plus
  AdamW state. Fixed by freezing before wrapping (`train_lora.py`), and both trainers now log and
  assert the trainable-param count (2.16M) — `train_v2.py` refuses to start above 50M.
* **Durability.** `run_training.sh` is a supervisor launched detached (`setsid --fork`) that runs the
  trainer in 25-minute chunks, checkpoints LoRA weights *and* AdamW state every 5 steps, snapshots
  every 100, and resumes until the step target is met. It takes `/tmp/jev-model.lock` so only one
  model-heavy job is ever in RAM — the fix for the OOM failure mode. This run survived a backend
  restart and 3× CPU oversubscription without losing a step.

## Evaluation

Two independent harnesses score the same 70 gold rows (`data/eval_gold.jsonl`), and never trained
on them:

* **Method A** (`eval_openjev.py`) — parallel constrained decoding: one batched forward pass per
  field, argmax over the allowed candidates, answer forced into the exact JSON shape the trainer
  writes. Candidates compared by full-string log-likelihood (3 of 4 bucket labels are multi-token);
  confidence = softmax over those sums.
* **Method B** (`eval_likelihood.py`) — independent scorer that ranks whole candidate answers by
  summed log-probability, with a staged field-by-field variant, so a result cannot come from a bug
  in one harness's candidate handling.

Both methods land on **65/70 = 92.9%** and agree on every row's bucket, so the headline does not
depend on one harness's candidate handling. Method B's `enum12` joint-answer variant is reported as
**not run** (its rows are `null` in `runs/eval_B_run-final.json`; the smoke phase never cleared the
model lock before the deadline) — do not cite an enum number. Mean bucket confidence on run-final is
0.945, close to Jev's 0.956 and far above the stock model's 0.610. `run-final`'s corpus was audited
clean of gold (title, employer) pairs (VERIFY.md 13/13), unlike run-night1's.

## Caveats, honestly

1. **Single seed, one run per corpus.** The comparison is between two runs, not two distributions.
2. **The gold set is small and unbalanced** — 70 rows, of which 54 are `generic_job` and only 2 are
   `service_lead`. Overall accuracy is dominated by the majority class; per-class recall is the
   number to watch, and `service_lead` recall rests on 2 rows.
3. **run-night1's snapshot number (90.0%) is contaminated** by 6 training rows / 4 pairs that share a
   (title, employer) with a gold row, worth up to +8.6pp; the clean-subset figure is in
   `runs/eval_A_run-night1.json` and recomputed by `summarize.py`. run-final has no such rows.
4. **0.5B is the compute-driven choice.** This box has no CUDA; a 1.5B LoRA would have taken ~9 h.
   The point here is transfer of judgment at fixed compute, not a claim about what a 7B would do.
5. Training loss ≠ agreement with Jev: the target is a 45-token JSON string, and the metric that
   matters is the constrained-decoding decision on held-out rows.

## Files

Scripts: `label_jev.py` (Jev's answers on any posting JSONL), `merge_corpus.py` (balanced,
gold-clean corpus), `train_v2.py` + `run_training.sh` (durable training), `eval_openjev.py`,
`eval_likelihood.py`, `verify_corpus.py`, `summarize.py` (regenerates the headline table).
Data: `data/train_jev.jsonl`, `data/dev_jev.jsonl`, `data/eval_gold.jsonl`,
`data/jevlab_hard/synth_hard.raw.jev.jsonl`. Runs: `runs/run-final/`, `runs/eval_*.json`.
Audits and status: `runs/VERIFY.md`, `runs/AUDIT-DATA.md`, `runs/EVAL.md`,
`runs/HARDCASE.md`, `runs/STATUS-*.md`. Coordination between the two sessions:
`COORDINATION.md`.

Reproduce: `openjev/merge_corpus.py` (corpus) → `TRAIN_FILE=openjev/data/train_jev.jsonl
openjev/run_training.sh run-final 400 25 6` → `openjev/eval_openjev.py --adapter
runs/run-final/adapter.pt` → `openjev/summarize.py`.
