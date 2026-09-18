# QUICKSTART — reproduce open-Jev 0.5B end to end

Everything below is CPU-only; no GPU is needed. The recorded run (2026-09-18) was: 6 vCPU, no
CUDA, Python 3.11.15 · torch 2.14.0+cpu · transformers 5.17.0 · 400 LoRA steps in 89.3 minutes.

## What's here

| path | what |
|---|---|
| `openjev/` | training + eval code (`train_v2.py`, `train_lora.py`, `run_training.sh`, `eval_openjev.py`, `eval_likelihood.py`, corpus tooling) |
| `data/` | `train_jev.jsonl` (2,591) · `dev_jev.jsonl` (286) · `eval_gold.jsonl` (70) |
| `runs/` | receipts: `FACTS.json` (every number used in the cards), `VERIFY.md` (13/13 corpus audit), `eval_A_run-final.json`, `eval_B_run-final.json` |
| `hf/` | the Hugging Face packaging: `upload_model.py`, `upload_dataset.py` (both dry-run by default), `README-model.md`, `README-dataset.md`, `MODEL_INFO.json`, and `staging/` holding the exact upload payload |

## 1. Environment

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
pip install "transformers==5.17.0" huggingface_hub
```

Those are the recorded versions. The harness is written to tolerate both the transformers v4 and
v5 spellings of the logits kwarg (`num_logits_to_keep` / `logits_to_keep`), so a nearby version
is fine — the numbers in `runs/` come from the stack above.

## 2. Base model

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
# older CLI:  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

## 3. Train (400 steps ≈ 89 minutes at 6 threads)

```bash
python openjev/train_v2.py \
    --model models/Qwen2.5-0.5B-Instruct \
    --run-name run-final \
    --train-file data/train_jev.jsonl \
    --steps 400 --target-total 400 --batch 6 --lr 1.5e-4 \
    --rank 16 --alpha 32 --max-len 192 --seed 7 --threads 6
```

Expect the first line to print `LoRA layers=96 r=16 | trainable=2.16M` — the trainer refuses to
start above 50M trainable params (that guard is what keeps the base weights frozen). The recorded
loss curve: **0.6792 at step 1 → 0.0468 at step 400** (mean of the last 10: 0.0253). The adapter is
written to `openjev/runs/run-final/adapter.pt` (~25 MiB; the checkpoint also carries AdamW state
and the step count).

The recorded run was actually driven through the chunked supervisor instead of one long call —
checkpoint every 5 steps, 25-minute chunks, one global model lock, survives restarts:

```bash
TRAIN_FILE=data/train_jev.jsonl openjev/run_training.sh run-final 400 25 6
```

Two things to fix before using the supervisor on another machine: it resolves its interpreter as
your interpreter via `OPENJEV_PY=/path/to/venv/bin/python` (defaults to `python3`), and it `cd`s to the
repo root before invoking the trainer, so `TRAIN_FILE` must be repo-root-relative
(`data/train_jev.jsonl`).

## 4. Evaluate (the 70 hand-labelled rows, same harness as the headline)

```bash
python openjev/eval_openjev.py \
    --model models/Qwen2.5-0.5B-Instruct \
    --adapter openjev/runs/run-final/adapter.pt \
    --data data/eval_gold.jsonl \
    --out runs/eval_openjev-final.json --threads 6
```

Expected summary line: **`bucket acc=92.9% (65/70)`** (per-class recall 1/2 `service_lead`,
12/14 `staff_role`, 52/54 `generic_job`; mean bucket confidence 0.945). Takes a few minutes on
CPU (~4.4 min for 70 rows at 4 threads in the recorded run).

Controls on the same rows:

```bash
# untrained base, no adapter → 72.9% (51/70)
python openjev/eval_openjev.py --model models/Qwen2.5-0.5B-Instruct \
    --data data/eval_gold.jsonl --out runs/eval_baseline-repro.json --threads 6

# score the released adapter without retraining
hf download decrux9812/openjev-0.5b adapter.pt --local-dir openjev/runs/run-final
sha256sum openjev/runs/run-final/adapter.pt
# expected: e49b717438fa54ea2bf03dd102ea7229045abf9e514db17b0ef5a93aefa027fe
```

The corpus and its cards are on the dataset repo `decrux9812/openjev-jev-labelled` if you'd
rather not re-derive the corpus from the sources.

## 5. Expected numbers (all from `runs/FACTS.json`)

| what | number | source |
|---|---|---|
| Jev (hosted), same 70 rows | 97.1% (68/70) | `runs/FACTS.json` |
| open-Jev 0.5B LoRA, 400 steps | 92.9% (65/70) — both eval harnesses, row-for-row agreement | `runs/eval_A_run-final.json`, `runs/eval_B_run-final.json` |
| per-class recall | service_lead 1/2 · staff_role 12/14 · generic_job 52/54 | same |
| untrained Qwen2.5-0.5B-Instruct | 72.9% (51/70) | `runs/eval_baseline.json` |
| stock Qwen2.5-1.5B + published constrained-decoding engine | 77.1% (54/70) = the majority-class rate | `runs/FACTS.json` |
| train loss | 0.6792 → 0.0468 (mean last 10: 0.0253) | `runs/FACTS.json` (`training.loss`) |
| trainable params / wall | 2,162,688 / 89.3 min | `runs/FACTS.json` (`training`) |

## 6. Path notes and gotchas

- The scripts' built-in defaults assume `data/` and `models/` sit inside `openjev/`
  (`eval_openjev.py` defaults `--data` to `openjev/data/eval_gold.jsonl`, `train_v2.py` defaults
  `--train-file` to `openjev/data/train.jsonl`). In this repo both live at the repo root, so pass
  `--data`, `--train-file` and `--model` explicitly as shown above.
- The eval harness also tries to read Jev's raw answers from a sibling `typesafe-lab` checkout for
  its Jev-agreement columns; where that is absent it prints `[jev] cannot read …` and continues —
  bucket accuracy against the `gold` column embedded in `eval_gold.jsonl` is unaffected.
- Training writes into `openjev/runs/<run-name>/`; run the commands from the repo root.
- Memory: batch 6 / `max_len` 192 logged a peak of ~6.3 GB RSS; `train_v2.py` checkpoints and
  stops cleanly above `--max-rss-gb` (default 7) if the box is smaller.
- To re-verify the corpus before trusting it: `python openjev/verify_corpus.py` reproduces the
  13/13 audit in `runs/VERIFY.md` (it only needs the tokenizer, not the model).
