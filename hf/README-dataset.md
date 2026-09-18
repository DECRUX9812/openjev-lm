---
license: mit
task_categories:
- text-classification
tags:
- jev
- structured-output
- lora
- sft
- json
- job-postings
---

# open-Jev labelled corpus — job postings with Jev's own decisions as targets

2,591 job postings whose supervision targets are the hosted Jev decision model's own answers,
plus a 286-row dev split and the 70-row hand-labelled evaluation set. This corpus is what the
companion model [`decrux9812/openjev-0.5b`](https://huggingface.co/decrux9812/openjev-0.5b) was
trained on.

**What this is, precisely:** a recording of a hosted API's responses, obtained through ordinary
requests. It is *not* Jev's weights, not a distillation of Jev's internals, and not affiliated
with, endorsed by, or connected to TypeSafe.

## Files

| file | rows | contents |
|---|---|---|
| `train_jev.jsonl` | 2,591 | training rows: `id`, `source`, `prompt`, `target`, `jev`, `family` |
| `dev_jev.jsonl` | 286 | held-out dev rows, same schema |
| `eval_gold.jsonl` | 70 | evaluation rows: `id`, `title`, `gold`, `jev_bucket`, `prompt`, `target` — `gold` is a human label; these rows were never trained on |

## Format

One JSON object per line.

- `prompt` — the exact training prompt, chat-template form: a `Decide.` system message, the
  posting as `title:` / `employer:` / `pay:` lines, then the requested field list
  (`bucket(service_lead|staff_role|generic_job|junk) technical_need business_buyer
  small_firm_doable pay_stated evergreen_repost fit(0-4)`), ending at `<|im_start|>assistant\n`.
- `target` — the hosted model's answer, exactly as the trainer consumes it: the JSON object in
  the inference schema, followed by the `<|im_end|>\n` terminator. Example ending:
  `{"bucket": "staff_role", "technical_need": true, "business_buyer": true, "small_firm_doable": true, "pay_stated": false, "evergreen_repost": true, "fit": 1}<|im_end|>\n`
- `jev` — the same answer as typed fields (`bucket`, the five booleans, `fit`), taken verbatim
  from the API response. `target` is generated from `jev`.
- `source` — where the posting came from: `real` (a posting from the live lead corpus) ·
  `labelled` (a generated posting) · `synth_hard` (from a hard-case set written against the
  classic failure modes: technical employee seats that read as plain jobs, genuine small service
  needs, commission/MLM junk, confusable negatives).
- `family` — a topic tag carried by the synthetic rows (present on 1,473/2,591 rows; absent on
  `real` rows).

`eval_gold.jsonl` additionally carries the human hand label (`gold`) and Jev's answer on that row
(`jev_bucket`); its `target` field is Jev's answer in the same format as training.

## Class balance (`train_jev.jsonl`, by `jev.bucket`)

| bucket | rows | share |
|---|---|---|
| `generic_job` | 1,088 | 42.0% |
| `staff_role` | 958 | 37.0% |
| `service_lead` | 342 | 13.2% |
| `junk` | 203 | 7.8% |

Source mix: `real` 1,118 · `labelled` 797 · `synth_hard` 676. (`dev_jev.jsonl`, same schema:
`generic_job` 120 / `staff_role` 106 / `service_lead` 38 / `junk` 22. The gold set is
`generic_job` 54 / `staff_role` 14 / `service_lead` 2 — small and unbalanced by design; it is an
evaluation set, not a sample of the class distribution.)

## Provenance and verification

- **Target provenance.** Every one of the 2,591 targets is byte-identical to the target
  generated from that row's `jev` block, and every `jev` block re-derives from the recorded API
  answers in the source corpus files (2,591/2,591 rows, 0 mismatches). Independent audit:
  `runs/VERIFY.md` in the source repo — 13/13 checks pass, exit 0.
- **No leakage.** 0 posting-id leaks and 0 `(title, employer)` pair leaks between train/dev and
  the 70 gold rows; the gold set is disjoint from both and internally consistent with its human
  labels (70/70).
- **Clean.** 0 duplicate ids, 0 duplicate `(title, employer)` pairs, 0 empty titles. All rows fit
  the training context (`max_len` 192; longest prompt+target 151 tokens — nothing truncated).
- **Cost.** Labels were collected by calling the hosted API: the 960-row hard-case labelling pass
  cost $0.0425 in API spend.
- **Determinism caveat.** A hosted model's answers are not perfectly deterministic. The corpus is
  a snapshot; re-collecting the same postings may return a small fraction of different targets.

## Intended use and limits

- Intended for research and reproduction: training and evaluating small local models on the
  posting → typed-decision task (bucket, five booleans, 0–4 fit).
- The labels are a hosted model's outputs, including whatever errors and biases it has. Prompts
  include generated postings; labels were not human-verified beyond the provenance checks above
  (the 70-row `gold` column is the human-labelled part).
- Do not present these labels as TypeSafe's canonical data, do not attempt to reconstruct or
  redistribute any Jev model from them, and respect the upstream service's terms of use.
- Corpus files and cards are released under the MIT license. Author: Ritesh Patel
  (GitHub [DECRUX9812](https://github.com/DECRUX9812)).
