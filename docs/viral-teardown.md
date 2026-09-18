# What the viral "Qwen-2.5-1B-RLCD" artifact is — and what it proves about Jev

**Verdict up front:** the artifact that went viral is **not a trained model and not a Jev reproduction in the ways that matter.** It is a *decoding trick* (parallel constrained decoding) applied to **stock Qwen2.5-1.5B-Instruct** — the HF repo contains **zero weight bytes** — and when we ran the exact same 70 hand-labelled postings through it and through real Jev, the stock model reproduced roughly 80% of Jev's *interfaces* (typed answers, probabilities, one forward pass) and **none of Jev's judgment** (77.1% vs 97.1% bucket accuracy; near-zero correlation on the money question).

Everything below is from runs on this machine on 2026-09-17/18, with artifacts on disk.

---

## 1. What the artifact actually is (verified)

| Claim | Verified fact |
|---|---|
| "open sourced Qwen-2.5-1B-RLCD" | HF repo `harshatheg/Qwen-2.5-1B-RLCD`: **`usedStorage: 0`**, file list is code only (`core/`, `presets/`, `server/`, `web/`, `app.py`) — **no `.safetensors`, no training, no RLCD phase** |
| "1B param model" | base is `Qwen/Qwen2.5-1.5B-Instruct` (1.5B), stock weights, loaded read-only |
| "5.6–7× faster … 100% schema validity … calibrated confidences" | we reproduced the mechanics (§4); "calibrated" is marketing — the numbers are raw softmax over candidate tokens (§3) |
| "built in stealth for 2 hours" | plausible; the entire repo is ~5 files of decoding logic + a Gradio demo |

The tracker Space (multimodalart/jev-reproductions-tracker) lists this artifact under **Decoding** ("inference technique, no new weights") — its own tracker agrees with our finding.

**Evidence:** `hf api` dump (usedStorage 0), `git ls-files` of the clone, model sha256 `dd924a11…c6d3ee` for the Qwen base.

## 2. Test setup (all local, reproducible)

- Engine: `harshatheg/Qwen-2.5-1B-RLCD` @ main, PyTorch path (`core/engine_torch.py`), CPU only (Intel Xeon W-2135, 6 vCPU, fp32).
- Model: Qwen2.5-1.5B-Instruct, sha256 verified.
- Task: the same 70-postings hand-labelled set the typesafe-lab uses for Jev (`data/leads.sample.jsonl` + `data/leads.gold.json`), mirrored into the same 7 questions (4-way bucket + 5 booleans + 0–4 fit score) — **identical inputs for both systems**.
- Jev side: live `jev-latest` run, 70 calls, **$0.0031**, 8.4 s wall; cached and live draws agree.
- Two variants: engine as-shipped, and with a boolean-token defect fixed (§4). Bucket labels identical on all 70 rows; per-field boolean probabilities do move (max |Δp| = 0.12 on one field of one row) → the defect is real but does not change the headline result.

**Files:** `runs_local/leads_local_asis.jsonl`, `runs_local/leads_local_spacefix.jsonl`, `runs_local/leads_jev_live_nocache.json`, `runs_local/analysis_leads_local_asis.json`.

## 3. Results — same 70 postings, same questions

| metric | stock Qwen2.5-1.5B + PCD | real Jev (`jev-latest`) |
|---|---|---|
| bucket accuracy (hand labels) | **77.1%** (54/70) | **97.1%** (68/70) |
| generic_job recall | 51/54 (94%) | 54/54 (100%) |
| staff_role recall (technical seats) | **3/14 (21%)** | 13/14 (93%) |
| service_lead recall (the money bucket) | **0/2** | 1/2 |
| bucket agreement with Jev | 78.6% | — |
| mean bucket confidence | 0.610 | 0.956 |
| ECE (5-bin) / Brier (4-class) | 0.157 / **0.359** | 0.097 / **0.058** |
| technical_need vs Jev | r=0.74, sign-agree 74% | — |
| small_firm_doable vs Jev | r=0.74, sign-agree 87% | — |
| pay_stated vs Jev | r=0.35, sign-agree 93% | — |
| business_buyer vs Jev | r=0.23, sign-agree 84% | — |
| evergreen_repost vs Jev | r=0.30, sign-agree 54% | — |
| **fit (0–4 opportunity score)** | mean \|Δ\| **3.22 levels**, r=**0.09**, exact-level 9% | — |
| latency / call | 8.5 s (6-core CPU, fp32; M4-Max 4-bit claims 68–270 ms) | 0.71 s mean / 0.38 s p50 (API RTT) |

**Failure shape:** 10 of the 16 local errors are *technical employee seats read as plain jobs* (Data Architect, Network & Server Analyst, Application Analyst, Data Modeler, IT Senior Analyst… → `generic_job`), plus both true `service_lead`s missed, plus 3 false positives working the other way (Welder and Carpenters read as "staff_role", a Director of Engineering as "service_lead"). The model has **no working notion of "a business buying technical work"** — precisely the distinction the whole exercise exists to make. Its `fit` field is noise (r=0.09), and it is systematically overconfident where it errs (Brier 6× Jev's).

**The confidences are not calibrated confidence.** The README says "calibrated field-level confidence scores"; mechanically they are softmax over the candidate first-token logits at one position, with no calibration training. On our set the local numbers are much less informative than Jev's (Brier 0.359 vs 0.058; mean confidence 0.61 vs 0.96) and they concentrate in 0.4–0.8 — i.e. the "probability" mostly encodes tokenizer uncertainty, not the chance of being right.

## 4. Mechanism checks (their claims, our hardware)

- **Schema validity:** 100% across all 70+70 runs and both variants — true by construction (values are picked from the candidate set and assembled programmatically), so "0% hallucination" here is definitional, exactly as TypeSafe says about its own 0%.
- **Parallel decoding:** verified — 1 prefill + 1 batched forward over M field suffixes per request; `sequential_forward_passes: 1`.
- **Speedup vs autoregressive (their own benchmark, our CPU — Xeon W-2135, 6 vCPU, fp32, no GPU):**

| preset | autoregressive | parallel | speedup | schema match |
|---|---|---|---|---|
| FinTech/AML, 28 fields | 160.4 s, 284 tokens, 284 passes | 14.1 s, 1 pass | **11.4×** | naive **False**, parallel True |
| Code security, 28 fields | 188.8 s, 291 tokens, 291 passes | 13.9 s, 1 pass | **13.6×** | naive **False**, parallel True |
| 255-choice tariff, 4 fields | 26.7 s, 48 tokens, 48 passes | 3.8 s, 1 pass | **6.9×** | naive **False**, parallel True |

  Their 5.6–7× README figures are M4-Max numbers with the 4-bit MLX engine; on our CPU the same code gives 6.9–13.6× and — a detail their README buries — the autoregressive baseline **fails to produce schema-valid JSON on all three presets** while the parallel path is always valid. The mechanism claims hold; the harness is honest about them.
- **Determinism:** 3 repeat runs of one posting → bit-identical probabilities (0.8289/0.1279/0.0247/0.0185 all three passes). So the local artifact is deterministic, while Jev itself is NOT (our cached vs live draws differ; third-party latency probing found the same).
- **Defect found:** boolean fields compare the logits of `true`/`false` (no leading space) while the prompt makes the model emit ` true`/` false`; we measured the impact by re-running all 70 rows with space-prefixed candidates → **identical bucket labels on all 70 rows**, but per-field boolean probabilities do move (max |Δp| = 0.12 on one field of one row; small_firm_doable sign-agreement with Jev shifts 87% → 79%). Real defect — fix it before reusing the engine — but it does not change the headline result.

## 5. So — is Jev a really small model? (the actual question)

The honest answer has three layers:

**a) The *interface* is small-model-cheap.** Verified twice over: one 1.5B stock model, prefill-only, all fields in one pass — that gets you Jev's *shape* (typed answers + per-field probabilities + ~100 ms-class latency on a laptop). This is why "Jev is just a small model" feels right to people watching the demos.

**b) The *judgment* is not.** On a real 70-row labelled task the stock small model is 20 points behind Jev on the headline bucket, ~4–70× behind on calibration, and 3.2 levels off on the score question. Jev's 97.1% here is *not* explained by "small model + decoding trick". Independent evidence agrees: the community's honest laptop benchmark (jev-on-a-laptop, Richard Becker) hit 73.8% agreement to Jev with Qwen 7B/8B; the LFM2.5-RLCD decoding-only artifacts sit around ~60% field accuracy; and the tracker's own summary says "what is still NOT in the open: TypeSafe's weights, the RLCD algorithm, and any open model matching Jev's calibration claims."

**c) The public evidence points to "small compute per decision, not small weights".** The deepest available teardown (Archer Hume, "Jev's Architecture Unmasked", 10k probing calls): shared-prefix prefill ~30k tokens in ~160 ms, per-question branches, direct probability readout ("no text generation anywhere in the pipeline"), **MMLU-Pro ≈ 84.6% re-measured on 1,200 items**, non-public tokenizer (415 probes), and scaling consistent with **a sparse MoE of roughly ~10B active parameters** on datacenter GPUs. A 1.5B dense model sits at ~20% MMLU-Pro (BenchGecko measurement of Qwen2.5-1.5B-Instruct) — that knowledge gap cannot be decoded into existence.

**Odds:** P(Jev is genuinely 1–4B-class) ≈ **10–15%**; P(Jev runs a sparse MoE / multi-branch transformer with ~10B-class active compute, i.e. "small per-decision compute" but not a small model) ≈ **70–80%**; P(it's a dense frontier-scale 200B+ monolith like a Llama/GPT-class thing) ≈ small — the latency and price don't fit, and nothing public suggests it.
What would settle it, in order of strength: (i) you can't — no weights, no paper; (ii) MMLU-Pro-style breadth probing of Jev vs each size class (Archer Hume's direction, extend to 3–5k items with better statistics); (iii) price-floor analysis: $0.042/MTok, output-free, 70–500 ms sets a hard ceiling on per-decision FLOPs — currently consistent with ~10B-active cost structure, not with a 1.5B dense (which would be ~10× cheaper to run and could be priced lower) and not with a 200B+ dense.

## 6. Bottom line

- The viral tweet proves something true and interesting: **the Jev *interface* is reproducible in a day on a stock 1.5B model.** It does not reproduce Jev.
- Anyone quoting "no hallucination" from that artifact should read it the way rahlquist did in the channel: it won't fabricate — it also won't be *right*; on our tests it is 77% right where Jev is 97%, and confidently wrong in the specific places that matter (it can't tell "a business buying work" from "a job posting").
- "Jev is a really small model" is likely **false as stated** (weights scale) and **true in the sense that matters for the bubble economics** (no generation, tiny active compute per decision, $0.042/MTok) — i.e. the cost structure of decisions is collapsing toward small-model economics while the *quality* still comes from serious training and calibration that no one has open-sourced.

## Receipts (on disk)

- `models/Qwen2.5-1.5B-Instruct/model.safetensors` — sha256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`
- `runs_local/leads_local_asis.jsonl` (70), `leads_local_spacefix.jsonl` (70), `leads_jev_live_nocache.json` (70), `analysis_leads_local_*.json`
- `runs_local/run_asis.log`, `runs_local/run_spacefix.log`, `runs_local/bench_presets.log`
- Engine clone + venv: `Qwen-2.5-1B-RLCD/` (`.venv`: torch 2.14.0+cpu, transformers 5.17.0)
- HF API dump: usedStorage 0 (code-only repo)
