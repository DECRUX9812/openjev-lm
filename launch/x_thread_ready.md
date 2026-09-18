# open-Jev — X thread, READY TO POST (real URLs filled in)

Post in order, images mapped per tweet. The HF link works only after the HF upload
(one command once the token file exists) — if you post before that, drop the HF bullet
from tweet 14 and add it as a reply later.

---

**1/**
A viral artifact claimed a stock 1.5B + a decoding trick "reproduces Jev".

We measured it on 70 hand-labelled rows: 77.1%.

Which is exactly the majority-class prior of that set.

So we trained the real thing overnight. Everything is open. 🧵

*[card1.png]*

**2/**
The ladder, all on the same 70 rows:

Jev (hosted): 97.1%
open-Jev classifier: 94.3%
open-Jev LM (0.5B + 2.16M LoRA params): 92.9%
the viral repro: 77.1%
untrained 0.5B: 72.9%
always-answer-`generic_job`: 77.1%

*[fig1_ladder.png]*

**3/**
What the viral artifact is: constrained decoding on stock Qwen2.5-1.5B — its repo ships zero weight bytes.

It reproduces the interface (typed answers, probabilities, one pass), almost none of the judgment: service_lead 0/2, staff_role 3/14, confidence 0.61 vs 0.96.

**4/**
Its errors aren't random. 10 of 16 are technical employee seats read as plain jobs — Data Architect, Network & Server Analyst, IT Senior Analyst.

It has no working notion of "a business buying technical work" — the one distinction the task exists to make.

**5/**
So: 2,591 real postings, every target = Jev's own API answer, stored byte-for-byte, audited 13/13 (labels re-checked, gold held out, no id or pair leaks).

Total labelling spend: under 5 cents.

**6/**
Training: Qwen2.5-0.5B-Instruct + LoRA r=16.

2,162,688 trainable params. 400 steps. Batch 6. 89 minutes. 6 vCPU. No GPU.

Loss 0.679 → 0.047.

*[card2.png]*

**7/**
Evaluation, because one harness is a rumour: two independently written scorers (step-386 snapshot, then the released adapter).

Both: 65/70 = 92.9%. Identical buckets on all 70 rows.

Mean confidence 0.945 (Jev: 0.956).

**8/**
Per-class where it matters: staff_role 12/14, generic_job 52/54.

And the honest one: service_lead 1/2. The rarest class is where every open reproduction still fails. Our archive contains exactly one true "business buying technical work" lead — both our arms miss it.

**9/**
The second arm is the one you'd actually deploy: a 3 MB frozen-encoder classifier.

94.3% on the same 70 rows. 99.39% agreement with Jev across 2,631 production postings. 6 ms/posting. $0. Offline. Deterministic — byte-identical reruns.

**10/**
Tested on live traffic, not just a frozen benchmark:

- 106 fresh, unseen postings: 106/106 bucket agreement with hosted Jev
- 107-postion cross-region boundary slice: 104/107
- whole 2,844-posting archive re-classified: 18 seconds, $0

**11/**
Negative results are published too — they're the interesting part:

- synthetic corpora fail (class balance is behaviour: 0/2 on the rare class)
- adding synthetic data to real data hurts (66 → 61/70)
- kNN blend loses to plain heads
- a 12× bigger encoder buys nothing

**12/**
The actual unlock isn't the score. It's the price.

At $0/call you stop budgeting per question and just… classify everything. Every posting, every night, forever, diffed.

Dark data: everything you collected and could never afford to read.

**13/**
What this is NOT: not Jev's weights, not a leak, not affiliated with TypeSafe. A workalike trained on answers obtained from their public API. No weights redistributed. MIT.

Respect upstream terms if you build on it.

**14/**
Everything, open:

- paper: https://decrux9812.github.io/openjev-lm/paper/openjev-paper.pdf
- LM arm + corpora: https://github.com/DECRUX9812/openjev-lm
- weights: https://huggingface.co/DECRUX9812/openjev-0.5b

~2 hours of CPU + 5 cents of labelling to reproduce.

Judgment-as-a-service has a $0 local alternative. The last 5% is the interesting part.

---

## Standalone bangers (post on other days)

**A/** "77.1%" is what you get for free by always answering the majority class. The viral Jev repro's headline number is the floor of the benchmark it was measured on. The judgment is the part you have to train.

**B/** A 0.5B model trained overnight on six CPU cores reproduces 92.9% of a hosted decision model's judgment. The interesting number isn't the accuracy — it's the marginal price: $0. Dark data is the dataset you could never afford to read.

**C/** Calibration is where the decoding-trick artifacts die: Brier 0.359 vs 0.058, mean confidence 0.61 vs 0.96. "Calibrated confidences" is softmax over candidate tokens, not calibration. Ask for the reliability diagram.

## Reply-guys, pre-empted

- "did you just distill someone's API?" → yes, that's stated on page 1, the model card, and tweet 13; no weights are redistributed; MIT for our artifacts; adjust on upstream request.
- "70 rows lmao" → correct, and the paper leads with it: ±5 points of noise, 2 service_lead rows, single seed, one domain. The 99.39% agreement number on 2,631 rows exists precisely because we know which of the two numbers is load-bearing.
- "a 3B would beat this" → probably; the 0.5B on a CPU is a deliberate floor, not a ceiling. The point is the floor is enough for most pipelines.
