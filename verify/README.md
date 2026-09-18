# Verify every number yourself

No installs, no torch, no trust required. One command re-derives every headline number in the
README and the paper:

```bash
python3 verify/score.py --self-test
```

It prints a PASS/FAIL line per claim. Current output (and what CI runs on every push):

```
LM arm, harness B (step-400 adapter)             65/70 = 92.9%  (service_lead 1/2 · staff_role 12/14 · generic_job 52/54)   PASS
LM arm, harness A (step-386 snapshot)            65/70 = 92.9%  (service_lead 1/2 · staff_role 12/14 · generic_job 52/54)   PASS
night-1 final adapter (420 steps)                65/70 = 92.9%  (...)                                                     PASS
night-1 snapshot (step 405)                      63/70 = 90.0%  (...)                                                     PASS
untrained 0.5B base                              51/70 = 72.9%  (...)                                                     PASS
viral Jev repro (as-is decoding)                 54/70 = 77.1%  (service_lead 0/2 · staff_role 3/14 · generic_job 51/54)    PASS
classifier arm (3 MB frozen encoder)             66/70 = 94.3%  (service_lead 0/2 · staff_role 12/14 · generic_job 54/54)   PASS
Jev (hosted) reference answers                   68/70 = 97.1%  (service_lead 1/2 · staff_role 13/14 · generic_job 54/54)   PASS
harness A vs harness B: identical buckets        70/70 identical                                                          PASS
live: classifier vs hosted Jev, 106 fresh        106/106 agree                                                            PASS
live: classifier vs hosted Jev, 107 boundary     104/107 agree                                                            PASS
live: LM vs hosted Jev, 106 fresh                104/106 agree                                                            PASS
```

## The receipts, one file per claim

| claim | receipt | notes |
|---|---|---|
| LM arm: 65/70 = 92.9% | `predictions/lm_B_step400.jsonl`, `lm_A_step386.jsonl` | per-row predictions, both harnesses, same ids |
| the two harnesses agree on every row | both files above | `score.py` compares them bucket by bucket |
| untrained base 72.9%, viral repro 77.1% | `predictions/untrained_0.5b.jsonl`, `predictions/viral_asis.jsonl` | viral file is the artifact's own per-row outputs, filtered to the 70 gold ids |
| classifier 66/70 = 94.3% | `predictions/classifier.jsonl` | regenerated from the classifier repo (`DECRUX9812/openjev`) |
| Jev itself 68/70 = 97.1% | `predictions/jev_reference.jsonl` | Jev's own answers for the same rows, as stored |
| 99.39% agreement on 2,631 production postings | `DECRUX9812/openjev` → `results/openjev_results.json` | the classifier repo's own receipt |
| live: classifier 106/106, 104/107; LM 104/106 vs hosted Jev | `live/fresh_106.jsonl`, `live/boundary_107.jsonl`, `live/lm_fresh_106.json` | unseen postings, side-by-side hosted vs local decisions; reports in `live/*.md` |
| the archive sweep (2,844 postings, 18 s) | `live/dark_sweep.jsonl` + `live/dark_sweep_report.md` | every row's decision + confidence |

Everything joins on the posting `id` (saskjobs job ids), so you can pull any row and read the
actual text for yourself.

## Why you can trust these receipts more than a README table

1. **Per-row, not aggregate.** Every number above is recomputable from row-level files; nothing is
   a summary that only we can produce.
2. **The gold set is published and audited.** `data/eval_gold.jsonl` (70 hand-labelled rows) with
   the audit in `runs/VERIFY.md` (13/13 checks: targets byte-identical to the stored API answers,
   no id leaks, gold disjoint from training).
3. **Two independent harnesses.** `eval_openjev.py` and `eval_likelihood.py` share no scoring code
   and return identical buckets on all 70 rows. Disagreement between them would be visible.
4. **Unseen-traffic receipts.** The live files are postings that arrived *after* the training
   corpus was built; the LM never saw them, and the classifier was compared against hosted Jev the
   same morning.
5. **CI re-runs it.** `.github/workflows/verify.yml` runs `score.py --self-test` on every push, so
   the badge in the README is a live check, not a screenshot.

## What we do *not* claim

- **We did not beat Jev.** Jev scores 97.1% on this set and stays there. The open arms reach 94.3%
  (classifier) and 92.9% (LM).
- The viral artifact's 77.1% is its *measured accuracy* on the same rows — and it happens to equal
  the majority-class rate of the set (54/70). That is the point, not a gotcha about its code.
- 70 rows is small; `service_lead` has **2 rows**, so its per-class recall (1/2) carries almost no
  statistical weight. We publish it anyway, because hiding it would be worse.
- Labels come from a hosted API that is not perfectly deterministic across draws
  (`runs/FACTS.md`), and this is a single-seed result on a single host.

## Regenerating the receipts from scratch

```bash
# harness outputs -> prediction files (needs the run artifacts in runs/)
~/.venvs/embed/bin/python verify/build_receipts.py

# the two harnesses themselves (needs torch + transformers; CPU is fine)
python openjev/eval_openjev.py --model <base model> --adapter runs/run-final/adapter.pt \
    --data data/eval_gold.jsonl --out runs/eval_A.json
python openjev/eval_likelihood.py --adapter runs/run-final/adapter.pt \
    --data data/eval_gold.jsonl --out runs/eval_B.json
```
