# open-Jev — LM arm

**A 0.5B model that reproduces a hosted decision model's judgment, trained overnight on a CPU, for $0.**

Jev (TypeSafe) answers seven typed questions about a text: a category, five booleans, and a 0–4 fit score.
This repo contains the **LM arm** of the open-Jev project: `Qwen2.5-0.5B-Instruct` + a LoRA adapter
(**2,162,688 trainable parameters**) trained on 2,591 rows whose targets are **Jev's own API answers** —
400 steps, 6 vCPU, **no GPU**, 89 minutes — plus the corpora, both evaluation harnesses, the corpus
auditor, and every receipt.

The **classifier arm** (frozen `bge-small` + small heads, 3 MB, 6 ms/posting, 99.39% agreement with Jev
on a 2,631-posting production stream) lives at **[DECRUX9812/openjev](https://github.com/DECRUX9812/openjev)**.
Both are MIT. The paper covering both: [`paper/openjev-paper.pdf`](paper/openjev-paper.pdf) —
launch page with figures and the full thread: [decrux9812.github.io/openjev-lm](https://decrux9812.github.io/openjev-lm/).

## The numbers (all on the same 70 hand-labelled postings)

| arm | bucket accuracy | service_lead / staff_role / generic_job | notes |
|---|---|---|---|
| Jev (hosted) | **68/70 = 97.1%** | 1/2 · 13/14 · 54/54 | the thing being reproduced |
| open-Jev classifier | 66/70 = 94.3% | 0/2 · 12/14 · 54/54 | 3 MB, 6 ms, 99.39% agreement on 2,631 |
| **open-Jev LM (this repo)** | **65/70 = 92.9%** | 1/2 · 12/14 · 52/54 | 2.16M LoRA params, 89 min CPU |
| viral "Jev repro" | 54/70 = 77.1% | 0/2 · 3/14 · 51/54 | = the majority-class prior (54/70) |
| untrained 0.5B base | 51/70 = 72.9% | 0/2 · 0/14 · 51/54 | below the prior |

Two independently written harnesses score this run — the first against the step-386 snapshot, the
second against the released step-400 adapter — and both land on **65/70 = 92.9%** with identical
buckets on all 70 rows. Mean bucket confidence: **0.945** (Jev 0.956, the viral artifact 0.610).

## Quickstart

```bash
# env: python 3.12, torch (CPU), transformers
git clone https://github.com/DECRUX9812/openjev-lm && cd openjev-lm
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# train (89 min on 6 vCPU; 3 supervised chunks if you use the wrapper)
bash openjev/run_training.sh run-final 400 25 6

# evaluate — expect 65/70, twice, from two harnesses
python openjev/eval_openjev.py    --run runs/run-final --data data/eval_gold.jsonl
python openjev/eval_likelihood.py --run runs/run-final --data data/eval_gold.jsonl

# audit the corpus — expect 13/13
python openjev/verify_corpus.py --data data/train_jev.jsonl --gold data/eval_gold.jsonl
```

Adapter: ships in this repo at `runs/run-final/adapter.pt` (25 MB, sha256 `e49b717438fa54ea…`);
Hugging Face mirror `DECRUX9812/openjev-0.5b` (upload pending).
(mirrored here as a release). Corpus: [`data/train_jev.jsonl`](data/train_jev.jsonl) (2,591 rows).

## What's in the box

```
openjev/        code — dataset builder, trainer, two eval harnesses, corpus auditor, labelling tools
data/           train_jev.jsonl (2,591) · dev_jev.jsonl (286) · eval_gold.jsonl (70)
runs/           every eval output, the corpus audit, the leak audit, FACTS.json + FACTS.md
docs/           night-report.md — the full engineering log of the run
paper/          the paper (md · html · pdf) and figures
weights/        pointer + receipts for the adapter
```

`runs/FACTS.md` is generated from the raw run files and is the single source every number above is
checked against — if a claim isn't in there, treat it as wrong.

## Honest limits

- **70-row evaluation set, 2 `service_lead` rows.** Every rare-class recall figure is one row wide.
- **One seed, one domain.** Regina job postings, one annotator, single training run.
- **Corpus #1 leaked** (6 rows / 4 gold pairs) — its numbers are reported raw *and* clean-subset
  (`runs/AUDIT-DATA.md`); the shipped corpus passes a 13/13 audit.
- **Labels come from a hosted API** whose answers are not perfectly deterministic. This is a research
  workalike, unaffiliated with TypeSafe; no Jev weights were used or redistributed. Respect upstream
  terms if you build on it.

## Negative results (also published)

Synthetic corpora fail (class balance is behaviour: 0/2 service_lead on a 70%-majority corpus);
adding synthetic data to real data *hurts* (66 → 61–63/70); kNN and kNN-blend lose to plain heads
(60/70, 62/70); a 12× larger encoder buys nothing after dev tuning; `enum12` joint-answer scoring
never produced usable answers and is reported as *not run*, not as a score. Full ledger:
classifier repo `docs/approaches-tried.md`.

MIT · Ritesh Patel (DECRUX9812) · September 2026
