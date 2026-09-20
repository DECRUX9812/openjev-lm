# GRPO post-training — the DeepSeek recipe on a $0 budget

`openjev/grpo_train.py` is a CPU-feasible implementation of **GRPO (Group Relative Policy
Optimization)** — the RL algorithm behind DeepSeek-R1 / R1-Zero — applied to the Jev
decision task. It post-trains the LoRA adapter that `train_v2.py` produces.

## The recipe, mapped onto this task

| DeepSeek-R1 | this repo |
|---|---|
| group of K sampled reasoning traces per prompt | K sampled typed-JSON decisions per posting |
| verifiable reward (math answer == gold) | per-field exact match vs the stored Jev teacher answer |
| no critic: advantage = (r − group mean) / group std | identical |
| KL penalty vs frozen SFT reference (k3 estimator) | identical |
| policy = full model | policy = the 2.16M-param LoRA adapter (base stays frozen) |

Reward (deterministic, no model in the loop):

| term | weight |
|---|---|
| `bucket` exact match | +1.00 |
| each of 5 booleans | +0.10 |
| `fit` closeness | +0.25 · (1 − \|Δ\|/4) |
| parses to JSON with the exact schema keys | +0.15 |
| unparseable | −0.20 flat |

Two practical details inherited from the field, not optional:

- **Degenerate-group skipping.** When all K samples score identically the advantage is
  zero and the step carries no gradient — skipped rather than burned. A warm SFT policy
  makes most easy prompts degenerate, so this is what keeps the loop informative.
- **Reference by weight-swap, not a second model.** The ref adapter is ~9 MB; it is
  swapped into the LoRA slots for each no-grad reference pass. Two 0.5B copies never
  coexist, so the whole loop stays inside the box's memory budget.

## Honest limits

- This is **RL-shaped calibration training on a distilled policy** — the audit's
  "RLCD-style post-training" item — not a replica of Jev's RLCD at scale. It sharpens
  what the teacher already taught; it cannot exceed the teacher's label quality.
- Reward signal is the corpus's hard Jev answers, not Jev's per-bucket probability
  distribution (the shipped corpus stores hard labels only).
- From the raw base model, K samples are all-unparseable → all rewards equal →
  degenerate groups → no learning. **GRPO needs the SFT warm start** (the same reason
  R1 ran SFT before RL). Run `train_v2.py` first.

## Usage

```bash
# 1. SFT warm start (as before)
python openjev/train_v2.py --run-name sft --train-file data/train_jev.jsonl \
    --steps 400 --target-total 400 --batch 6 --threads 6

# 2. GRPO on top (steps = RL update rounds, each = prompts × K rollouts)
python openjev/grpo_train.py --run-name rl1 --init openjev/runs/sft/adapter.pt \
    --steps 60 --prompts 3 --k 8 --temperature 1.0 --lr 1e-5 --beta 0.04

# 3. evaluate — the checkpoint format is train_v2-compatible, both harnesses read it
python openjev/eval_likelihood.py --adapter openjev/runs/rl1/adapter.pt \
    --data data/eval_gold.jsonl
```

`runs/<name>/grpo_log.jsonl` is the receipt ledger: per-step mean reward, bucket-rate
under sampling, parse rate, KL, wall-clock. `--resume` continues a killed run (the ref
adapter is stored inside the checkpoint so resume is exact).

## Proof run (this repo, 2026-09-20, Apple Silicon CPU)

150 SFT steps (4.9 min, final loss ≈ 0.026 — matches the recorded curve) then 60 GRPO
steps (13.2 min, ~15 s/step, 3 prompts × 8 samples). Method-B likelihood eval on the
full 70-row gold set:

| | SFT only | SFT + GRPO |
|---|---|---|
| bucket vs gold | 63/70 = 90.0% | 62/70 = 88.6% |
| bucket vs Jev's buckets | 63/70 = 90.0% | **64/70 = 91.4%** |
| bucket margin (mean, nats) | 1.55 | **2.00** (+29%) |
| answer mean logp | −0.0065 | **−0.0032** |

What the RL pass actually did: sampled bucket-rate went 0.75 → ~1.0 and mean reward
1.49 → 1.90 over the run with KL ≤ 0.04 — the policy sharpened onto the rewarded mode
without leaving the teacher's distribution. Row-level: 3 flips — one fix (`1533170`),
one regression (`1543088`), one tracking a Jev teacher error (`1541075`). At proof scale
the effect is **calibration and teacher-alignment, not an accuracy jump** — exactly what
"RL on top of distillation" predicts. The 400-step shipped recipe remains the accuracy
reference; this harness is the post-training stage on top of it.

## Knobs that matter

- `--temperature` / `--k`: exploration width. Too cold → degenerate groups; too hot →
  reward noise. 1.0 / 8 works on this corpus.
- `--beta`: KL leash to the SFT policy. 0.04 keeps decisions inside the teacher's
  distribution while still letting borderline rows move.
- `--lr`: 1e-5 (an order below SFT — the policy is already close).
