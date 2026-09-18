# r/LocalLLaMA post — open-Jev

_Author: Ritesh Patel — independent researcher, Regina, Canada. Not affiliated with TypeSafe. Placeholders: {REPO} · {PAPER} · {HF}._

**Title:** 0.5B + LoRA, trained on CPU in 89.3 minutes: 92.9% bucket accuracy on held-out postings (Jev itself: 97.1%) — and the viral 77.1% RLCD result was just the majority-class prior

**Body:**
**TL;DR** — Qwen2.5-0.5B + LoRA (2.16M trainable params), trained on 2,591 rows labelled by Jev's public API. CPU-only, 6 vCPU, 89.3 minutes, $0. On 70 hand-labelled postings (never trained on): 65/70 = 92.9% bucket accuracy, vs 68/70 = 97.1% for Jev itself. Confirmed by two independent eval harnesses. Code, data and receipts: {REPO} (MIT).

**Why this exists: the viral 77.1% was a class prior.** The artifact that made the rounds — stock Qwen2.5-1.5B-Instruct plus parallel constrained decoding, with the HF repo carrying zero weight bytes — scores 54/70 = 77.1% on the same 70 hand labels. But 54 of those 70 rows are generic_job, so "always answer generic_job" scores 54/70 = 77.1% exactly — the headline is the majority-class prior, to one decimal. Its service_lead recall: 0/2. staff_role: 3/14. Mean bucket confidence: 0.61. The interface reproduces; the judgment does not. That gap is this project.

**What I trained.** Qwen2.5-0.5B-Instruct + LoRA r=16, alpha=32 on q/k/v/o — 2,162,688 trainable params, frozen base enforced. Corpus: 2,591 postings, every target label Jev's own answer (958 staff_role / 1,088 generic_job / 342 service_lead / 203 junk). 400 steps, batch 6, max_len 192, lr 1.5e-4 OneCycle. 89.3 minutes wall clock on 6 vCPU, CPU-only — PyTorch, this box has no CUDA.

**Results vs the 70 hand labels:**
- Jev (hosted): 68/70 = 97.1%
- open-Jev LM: 65/70 = 92.9% — service_lead 1/2, staff_role 12/14, generic_job 52/54
- open-Jev LM, second harness: 92.9%, agrees row-for-row with the first
- Untrained 0.5B: 51/70 = 72.9% — below the prior, 0/2 and 0/14 on the rare classes
- Encoder arm (frozen bge-small-en-v1.5 + heads): 66/70 = 94.3% on gold, 99.39% bucket agreement with Jev over a 2,631-posting production stream, 3 MB weights, 6.0 ms per posting

**Eval, briefly.** Method A is parallel constrained decoding (one batched forward per field, argmax over allowed candidates). Method B is an independent staged whole-candidate likelihood scorer. Both score the same adapter on the same 70 rows, land on 92.9%, and agree per row — so the headline does not rest on one harness's candidate handling. Corpus audit: 13/13 checks — every training target byte-reproducibly Jev's answer, no id leaks, no pair leaks, gold disjoint.

**Honest limits.** 70 gold rows, 54 of them generic_job, 2 service_lead; overall accuracy is majority-dominated and 1/2 service_lead recall rests on two rows. Single seed, one run per corpus. 0.5B because there is no GPU here — a compute-bound choice, not a claim about bigger models. Not affiliated with TypeSafe; labels come from Jev's public API; respect their terms.

**Reproduce it.** The dev set is 286 rows, gold is 70, and both eval harnesses plus the corpus verifier are in the repo. Run it and check the numbers yourself.

{REPO} · {PAPER} · {HF}
