# EVAL.md - openjev eval harness: the untrained-base "before" number

Generated from `/home/decrux/Code/jev-repro-test/openjev/runs/eval_baseline.json` (run finished 2026-09-18 04:46; 70 rows). Harness: `openjev/eval_openjev.py`.

- **Scored thing:** `models/Qwen2.5-0.5B-Instruct` - untrained base (no adapter)
- **Data:** `openjev/data/eval_gold.jsonl` - 70 hand-labelled Regina postings, never trained on; prompts verbatim from the file (compact format)
- **Runtime:** 260.3s wall, CPU 4 threads, 3.72s/row; model load 1.2s; schema-valid answers on every row: True
- **Jev answer vectors:** `/home/decrux/Code/typesafe-lab/runs/leads_corpus_full.json`, `/home/decrux/Code/jev-repro-test/runs_local/leads_jev_live_nocache.json`

## Method (parallel constrained decoding - same shape as the published baseline)

One batched forward pass per field (7 per row); the answer is forced into the exact JSON shape
`dataset.py:target_for()` writes, and at each field ONLY the allowed candidates are scored and the
argmax appended. Decision positions match the published RLCD engine (`compile_parallel_metadata`):
the value is scored right after the field's scaffold - bucket at `{"bucket": "`, booleans at
`", "<field>":`, fit at `, "fit":`.

Candidates (tokenization verified at runtime; ids in the JSON's `meta.candidate_token_ids`):

- bucket -> `service_lead | staff_role | generic_job | junk` - every label is 2 tokens (`generic_job` -> `generic`+`_job`, `junk` -> `j`+`unk`)
- booleans -> `' true' | ' false'` - single tokens, exactly as the trainer's targets spell them
- fit -> `' 0' ... ' 4'` - 2 tokens (`' '` + digit)

Since 3 of 4 bucket labels (and fit) are multi-token, candidates are compared by **full-string
log-likelihood** (sum of per-token logprobs) - the fallback the brief prescribes - not a single-token
logit. Confidence = softmax over those summed logprobs (identical to the published single-token
softmax when the candidates are single tokens). `bucket_probs_firsttok*` keeps the published
first-token variant for a direct cross-check.

## Numbers (untrained base)

| metric | value |
|---|---|
| bucket accuracy vs hand labels | **72.9%** (51/70) |
| per-bucket recall vs gold | service_lead: 0/2 (0.0%), staff_role: 0/14 (0.0%), generic_job: 51/54 (94.4%), junk: 0/0 (n/a) |
| per-bucket precision (predicted) | service_lead: 0/0 (n/a), staff_role: 0/3 (0.0%), generic_job: 51/67 (76.1%), junk: 0/0 (n/a) |
| agreement with Jev's bucket (`jev_bucket`) | 74.3% |
| boolean agreement vs Jev (all 5 fields) | 63.7% |
|   - technical_need | 55/70 = 78.6% |
|   - business_buyer | 26/70 = 37.1% |
|   - small_firm_doable | 55/70 = 78.6% |
|   - pay_stated | 44/70 = 62.9% |
|   - evergreen_repost | 43/70 = 61.4% |
| fit mean abs delta vs Jev | 0.3633 levels |
| fit exact-level rate vs Jev | 2.9% |
| mean bucket confidence | 0.7698 |
| ECE (5-bin) / Brier (4-class) | 0.1172 / 0.463 |
| cross-check: published first-token variant accuracy | 72.9% (0 labels differ from full-string) |

Context: real Jev scores **97.1%** (68/70) on these rows; the published stock-1.5B parallel-decoding
baseline scored **77.1%** (54/70) with first-token scoring (`REPORT.md` sections 3-4).

## Confusion (gold -> predicted, all 70 gold rows)

| gold \ pred | service_lead | staff_role | generic_job | junk |
|---|---|---|---|---|
| service_lead | 0 | 0 | 2 | 0 |
| staff_role | 0 | 0 | 14 | 0 |
| generic_job | 0 | 3 | 51 | 0 |
| junk | 0 | 0 | 0 | 0 |

## Caveats

- Jev's booleans/fit used here are one live draw (`noul >= 0.5` binarisation); Jev is not deterministic, so
  boolean agreement carries Jev-side noise (cached-vs-live draws differ; see REPORT.md section 4).
- The brief listed space-prefixed bucket candidates (' service_lead'); at the exact decoding position
  (inside the opening quote, matching the published engine's suffix) the labels carry no leading space.
  Deviation documented; the first-token cross-check above quantifies how much the scoring variant matters.
- Full-string likelihood makes confidence sensitive to how a label tokenizes (cheap vs expensive tokens).
  Treat confidence/ECE/Brier as ranking diagnostics, not calibrated probabilities.
- n=70: about +/-5-6 points at 95% CI on the headline; tiny buckets (e.g. service_lead n=2)
  have near-meaningless per-bucket rates.
- `fit` is a 5-way choice at one position, so delta-vs-Jev is discrete and bounded (0-4 levels).

## Exact commands

Re-run on the trained adapter (the 'after' number):

```bash
cd /home/decrux/Code/jev-repro-test && flock -w 3600 /tmp/jev-model.lock -c 'nice -n 5 \
  /home/decrux/Code/jev-repro-test/Qwen-2.5-1B-RLCD/.venv/bin/python openjev/eval_openjev.py \
    --model models/Qwen2.5-0.5B-Instruct --adapter openjev/runs/<run-name>/adapter.pt \
    --data openjev/data/eval_gold.jsonl --out openjev/runs/eval_<run-name>.json'
```

Smoke (8 rows) / recompute metrics from a finished JSON without loading a model:

```bash
flock -w 3600 /tmp/jev-model.lock -c 'nice -n 5 /home/decrux/Code/jev-repro-test/Qwen-2.5-1B-RLCD/.venv/bin/python \
  openjev/eval_openjev.py --model models/Qwen2.5-0.5B-Instruct --data openjev/data/eval_gold.jsonl --limit 8 --out /tmp/eval_smoke8.json'

python3 openjev/eval_openjev.py --recompute openjev/runs/eval_baseline.json   # rows -> metrics, reproducible
```
