# Hugging Face community post — open-Jev

_Author: Ritesh Patel — independent researcher, Regina, Canada. Not affiliated with TypeSafe. Placeholders: {REPO} · {PAPER} · {HF}._

**Title:** open-Jev: a 0.5B local workalike of Jev's judgment (92.9% on 70 held-out rows, MIT)

**Post:**
open-Jev is an open, local reproduction of Jev's typed judgments on job postings: a four-way bucket, five booleans, and a 0-4 fit score. It was trained only on labels from Jev's public API.

Recipe: Qwen2.5-0.5B-Instruct + LoRA r=16, alpha=32 on q/k/v/o — 2,162,688 trainable params, frozen base — on 2,591 Jev-labelled rows. 400 steps, batch 6, max_len 192, 89.3 minutes on 6 vCPU, CPU-only, no GPU. $0 per call at inference.

On 70 hand-labelled postings: open-Jev 65/70 = 92.9%; Jev itself 68/70 = 97.1%; untrained 0.5B 51/70 = 72.9%. Two independent harnesses (parallel constrained decoding, staged likelihood scoring) both land on 92.9% and agree row-for-row. Corpus audit 13/13: every training target is byte-reproducibly Jev's own answer, no gold leakage.

Also in this release: a frozen bge-small-en-v1.5 encoder arm with small heads — 3 MB, 6.0 ms per posting, 66/70 = 94.3% on gold, 99.39% bucket agreement with Jev over a 2,631-posting production stream.

Limits, plainly: 70 gold rows (54 generic_job, 2 service_lead — recall 1/2), single seed, and 0.5B was the CPU-bound choice, not a claim about bigger models. Labels are Jev's answers from its public API; this is independent work, not affiliated with TypeSafe. Please respect upstream terms.

Model: {HF} · Data and code: {REPO} · Paper: {PAPER}
