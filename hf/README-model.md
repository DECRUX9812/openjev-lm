---
license: mit
base_model: Qwen/Qwen2.5-0.5B-Instruct
pipeline_tag: text-classification
tags:
- jev
- structured-output
- lora
- cpu
- classification
---

# open-Jev 0.5B — a local workalike of Jev's job-posting decision

A LoRA adapter (rank 16, alpha 32, on the q/k/v/o projections of every attention block) for
`Qwen/Qwen2.5-0.5B-Instruct`. Given a job posting it answers the same typed decision Jev returns:
a `bucket` (`service_lead` | `staff_role` | `generic_job` | `junk`), five booleans
(`technical_need`, `business_buyer`, `small_firm_doable`, `pay_stated`, `evergreen_repost`) and a
0–4 `fit` score, emitted as a fixed JSON object. The adapter was trained for 400 steps on 2,591
postings whose targets are the hosted model's own recorded answers — 2,162,688 trainable
parameters, base weights frozen, 89.3 minutes on a 6-vCPU CPU-only host. It is an independent
workalike: no Jev weights or internals were used, and there is no affiliation with TypeSafe.

**Files in this repo:** `adapter.pt` (the LoRA weights, 25.0 MiB / 26,169,663 bytes,
sha256 `e49b717438fa54ea2bf03dd102ea7229045abf9e514db17b0ef5a93aefa027fe`), `MODEL_INFO.json`
(machine-readable config + train stats), `README.md`.

## Results

Bucket accuracy on 70 hand-labelled postings; every system scored with the same harness and the
same rows:

| system | bucket accuracy | per-class recall (service_lead / staff_role / generic_job) |
|---|---|---|
| Jev (the hosted model being imitated) | 97.1% (68/70) | — |
| **open-Jev 0.5B (this adapter)** | **92.9% (65/70)** | 1/2 / 12/14 / 52/54 |
| stock Qwen2.5-1.5B + the published parallel-constrained-decoding engine | 77.1% (54/70) | 0/2 / 3/14 / 51/54 |
| base Qwen2.5-0.5B-Instruct, no adapter | 72.9% (51/70) | 0/2 / 0/14 / 51/54 |

Two independent harnesses scored the same adapter on the same 70 rows and land on the same 65/70 —
`eval_openjev.py` (parallel constrained decoding: one batched forward per field, argmax over the
allowed candidates) and `eval_likelihood.py` (staged whole-answer likelihood) — and they agree on
every row's bucket, so the headline does not depend on one harness's candidate handling. Mean
bucket confidence is 0.945. Two notes that keep the table honest:

- 54 of the 70 gold rows are `generic_job`, so "always answer `generic_job`" scores 54/70 =
  77.1% — the published 1.5B baseline's number is, to one decimal, the majority-class rate on
  this set, not a skill floor above it. The untrained 0.5B base scores *below* that rate (72.9%)
  and has zero recall on `service_lead` and `staff_role`.
- The method-A receipt was written while the final training chunk was still in flight and read
  the step-386 checkpoint; method B scored the released step-400 adapter. Both report 65/70.

## How to use

The adapter is a plain `torch.save` checkpoint (dict keys: `format` = 2, `lora`, `opt`,
`step` = 400), not a PEFT directory, so it loads with the small manual-LoRA implementation in the
source repo ([github.com/DECRUX9812/openjev](https://github.com/DECRUX9812/openjev), MIT).

```bash
git clone https://github.com/DECRUX9812/openjev && cd openjev
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers huggingface_hub
hf download decrux9812/openjev-0.5b adapter.pt --local-dir runs/run-final

# score it on the 70 hand-labelled rows — expect the final line "bucket acc=92.9% (65/70)"
python openjev/eval_openjev.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter runs/run-final/adapter.pt --data data/eval_gold.jsonl \
    --out runs/eval_openjev-0.5b.json
```

Deciding a single posting, using the same decode interface the numbers above were measured with:

```python
import sys; sys.path.insert(0, "openjev")   # eval_openjev.py imports train_lora.py from here
from eval_openjev import load_model, decide, BUCKETS, BOOLS, EOS

model, tok, info = load_model("Qwen/Qwen2.5-0.5B-Instruct", "runs/run-final/adapter.pt")
print(info)   # {'adapter_rank': 16, 'adapter_step': 400, 'adapter_params': 192, ...}

posting = "title: IT Support Specialist\nemployer: Acme Industries\npay: $28/hr"
prompt = ("<|im_start|>system\nDecide.<|im_end|>\n<|im_start|>user\n" + posting +
          "\nbucket(service_lead|staff_role|generic_job|junk) technical_need business_buyer "
          "small_firm_doable pay_stated evergreen_repost fit(0-4)<|im_end|>\n<|im_start|>assistant\n")

s = prompt + '{"bucket": "'
bi, _, _, _ = decide(model, tok, s, BUCKETS)          # argmax over the 4 bucket labels
s += BUCKETS[bi] + '"'
for b in BOOLS:
    s += f', "{b}":'
    yi, _, _, _ = decide(model, tok, s, [" true", " false"])
    s += " true" if yi == 0 else " false"
s += ', "fit":'
fi, _, _, _ = decide(model, tok, s, [" 0", " 1", " 2", " 3", " 4"])
s += f" {fi}" + "}" + EOS
print(s[len(prompt):-len(EOS)])                       # the answer JSON, schema-valid by construction
```

`eval_openjev.py` reads `adapter_config.json` next to the checkpoint if present for rank/alpha,
and defaults to exactly this adapter's values (rank 16, alpha 32) otherwise — `adapter.pt` alone
is sufficient. Inference is CPU float32; no GPU or CUDA build is required. A GPU is not needed but
a stock PEFT/safetensors export would load faster if you have one; this file is a PyTorch pickle.

## Training data

The training corpus is the companion dataset [`decrux9812/openjev-jev-labelled`](https://huggingface.co/datasets/decrux9812/openjev-jev-labelled):
2,591 postings, every target byte-verified against the hosted API's response (13/13 audit checks,
0 id leaks, 0 duplicate `(title, employer)` pairs — `runs/VERIFY.md` in the source repo). Class
balance: `generic_job` 1,088 (42.0%), `staff_role` 958 (37.0%), `service_lead` 342 (13.2%),
`junk` 203 (7.8%). Source mix: 1,118 real postings (`source=real`), 797 generated postings
(`source=labelled`), 676 hard-case rows (`source=synth_hard`, retained from a 960-row set written
against the classic failure modes and labelled by the API for $0.0425 of spend). 286 dev rows and
the 70 hand-labelled gold rows ship with the dataset; the gold rows were never trained on.

Training run: 400 steps, batch 6, `max_len` 192, lr 1.5e-4 (OneCycleLR, `pct_start` 0.05), seed 7,
loss 0.6792 at step 1 → 0.0468 at step 400 (mean of the last 10: 0.0253), 89.3 minutes across
three supervised chunks on a 6-vCPU CPU-only host.

## Limitations

- **Single seed, single run.** This is one 400-step adapter, not a distribution over runs.
- **The eval set is small and unbalanced.** 70 rows, 54 of them `generic_job`, and only 2
  `service_lead` — the `service_lead` recall above rests on those 2 rows (1/2). Overall accuracy
  is dominated by the majority class; per-class recall is the number to watch.
- **It imitates outputs, not internals.** The targets are a hosted model's answers at one point
  in time. A hosted model's answers are not perfectly deterministic, so a small fraction of
  targets would differ if collected again, and the modelled behaviour can drift with the service.
- **One task, one prompt format.** Scope is the posting→typed-decision task with the prompt shown
  above; it is not a general-purpose instruct model — use `Qwen/Qwen2.5-0.5B-Instruct` for
  anything else.
- **No separate accuracy claim for the five booleans or `fit`.** They are trained and emitted, and
  the harness records agreement/delta columns for them, but the measured headline is the bucket.
- Trained on a 6-vCPU CPU-only box by choice of compute, not a claim that 0.5B is the ceiling.

## Provenance and ethics

- Independent workalike by Ritesh Patel (GitHub [DECRUX9812](https://github.com/DECRUX9812)).
  Not affiliated with, endorsed by, or connected to TypeSafe or Jev.
- **No Jev weights, internals, or training code were used.** The adapter was trained only on
  input→output pairs obtained by calling the hosted API with ordinary requests; the dataset is a
  recording of API responses.
- Please respect the upstream service's terms of use. This release exists for research and
  evaluation of local inference on a structured decision task; do not present it as Jev, and do
  not treat its outputs as carrying Jev's authority.
- The adapter file, this card and the accompanying code are released under the MIT license.
