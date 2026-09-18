# Model card — open-Jev LM arm (`openjev-lm`)

**What it is.** A LoRA adapter for `Qwen/Qwen2.5-0.5B-Instruct` that answers the same seven typed
questions as Jev (TypeSafe): a four-class `bucket`, five boolean fields (`technical_need`,
`business_buyer`, `small_firm_doable`, `pay_stated`, `evergreen_repost`), and a five-point `fit`
score — each with a probability — over a job posting.

**Results** (70 hand-labelled postings, held out of all training):

| arm | bucket accuracy | staff_role | service_lead | mean conf |
|---|---|---|---|---|
| Jev (hosted, reference) | 68/70 = 97.1% | 13/14 | 1/2 | 0.956 |
| open-Jev LM (this adapter) | **65/70 = 92.9%** | 12/14 | 1/2 | 0.945 |
| viral "reproduction" (stock 1.5B + constrained decoding) | 54/70 = 77.1% | 3/14 | 0/2 | 0.610 |
| untrained 0.5B base | 51/70 = 72.9% | 0/14 | 0/2 | — |

Two independently written harnesses (`eval_openjev.py` on the step-386 snapshot,
`eval_likelihood.py` on the released step-400 adapter) both score 92.9% with identical buckets on all
70 rows.

**Training.** 2,591 postings whose targets are Jev's own API answers, stored byte-for-byte and
re-verified (`verify_corpus.py`: 13/13 checks — no id leaks, no (title, employer) pair leaks,
gold disjoint). LoRA r=16, α=32, on q/k/v/o; 2,162,688 trainable params. 400 steps, batch 6,
max_len 192, lr 1.5e-4 OneCycle, ~7.5 s/step. **89 minutes wall, six vCPUs, no GPU.** Loss
0.679 → 0.047.

**Limitations.** Single seed. 70-row evaluation set. `service_lead` recall is 1/2 — two rows
exist; the rare class is where every open reproduction we know of fails. Trained on one stream
(Regina job postings) — domain and vocabulary specific. Staged likelihood scoring is ~3.6 s/row
on CPU; the classifier arm is the deployment path for volume.

**Provenance and ethics.** No Jev weights were used or redistributed. Every training target is an
answer obtained from Jev's public API; the corpus is published as a record. The adapter is a
workalike, not a copy, and is unaffiliated with TypeSafe. MIT. If you build on this, respect
upstream terms — details in `docs/provenance-and-ethics.md`.

**Citation.** Patel, R. *open-Jev: reproducing a hosted decision model's judgment locally, for
nothing.* September 2026. Paper: `paper/openjev-paper.pdf`.
