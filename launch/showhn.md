# Show HN post — open-Jev

_Author: Ritesh Patel — independent researcher, Regina, Canada. Not affiliated with TypeSafe. Placeholders: {REPO} · {PAPER} · {HF}._

**Title:** Show HN: Open-Jev – a 0.5B model at 92.9% on Jev's judgment task (Jev: 97.1%)

**Body:**
Jev (TypeSafe's hosted decision model) takes a short piece of text — here, a job posting used as a sales lead — and answers seven typed questions about it: a four-way bucket (service_lead / staff_role / generic_job / junk), five booleans, and a 0-4 fit score. There are no open weights and no published training data, so I trained a workalike from its answers alone. Open-Jev is Qwen2.5-0.5B-Instruct plus a LoRA adapter, trained on 2,591 postings labelled by Jev's own public API. All MIT.

The question was whether a half-billion-parameter model can pick up a bigger hosted model's judgment, not just its output format. On 70 hand-labelled postings it gets 65/70 = 92.9% bucket accuracy, against 68/70 = 97.1% for Jev itself. Trained overnight, CPU-only, no GPU, $0 at inference.

The numbers:
- Jev: 68/70 = 97.1% bucket accuracy on the hand labels
- open-Jev LM (0.5B + 2.16M-param LoRA): 65/70 = 92.9% — recall service_lead 1/2, staff_role 12/14, generic_job 52/54
- Two independent eval harnesses both land on 92.9% and agree row-for-row
- Untrained 0.5B baseline: 51/70 = 72.9%, below the majority-class prior, zero recall on the two rare classes
- Training: 400 steps, batch 6, max_len 192, 89.3 minutes on 6 vCPU, CPU-only
- Second arm, frozen encoder: bge-small-en-v1.5 + heads — 66/70 = 94.3%, bucket agreement with Jev on 2,615/2,631 postings = 99.39%, 3 MB, 6.0 ms per posting
- Context: the viral RLCD artifact (stock Qwen2.5-1.5B-Instruct + constrained decoding) measured 54/70 = 77.1% — exactly the majority-class prior, since 54 of the 70 gold rows are generic_job

What's in the repo: the Jev labeler (label_jev.py), the corpus builder/merger plus verifier (13/13 checks: every training target is byte-reproducibly Jev's own answer, no id leaks, no pair leaks, gold disjoint), the durable chunked trainer (train_v2.py + run_training.sh), two eval harnesses (eval_openjev.py, eval_likelihood.py), the data (2,591 train / 286 dev / 70 gold), and the run outputs behind every number above (runs/FACTS.json). All MIT.

Honest limits: the gold set is 70 rows and unbalanced — 54 are generic_job and only 2 are service_lead, so overall accuracy is majority-dominated and 1/2 service_lead recall means little on its own. Single seed, one run per corpus; no error bars worth quoting. 0.5B was the compute-driven choice (no CUDA on this box), not a claim about larger models. The encoder arm above never catches service_lead (0/2) — use the LM arm for that class. An earlier run on a skewed corpus also scored 65/70, but 4 gold rows shared (title, employer) with its training data, so only its clean subset (62/66) is reported. Labels come from Jev's public API; this is independent work, not affiliated with TypeSafe; check upstream terms before commercial use.

Why: at $0 per call and ~6 ms per posting, classification stops being a metered operation and becomes something you can run over everything you ever collected. Labelling 960 extra hard-case postings through Jev cost $0.0425; classifying them locally now costs nothing. That is the dark-data case.

Questions about the eval protocol or the corpus welcome in the comments.

{REPO} · {PAPER} · {HF}
