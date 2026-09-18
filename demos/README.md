# openjev-lm demo videos

All demos show **real local inference on CPU** — no API calls, no GPU. Every number
on screen is live output from the models in this repo.

## The master cut

| File | Length | Contents |
|---|---|---|
| `videos/openjev-demo-master-216s.mp4` | 3:36 | Full demo: hook → offline proof → classifier sweep → receipts → LM arm money shot. Clip freely. |

Pre-cut clips (from the master, in `videos/clip*.mp4`):
`clip5_hook` (cold open) · `clip3_offline` (airplane-mode proof) ·
`clip4_determinism_sweep` (same input → same JSON, 3×) · `clip2_receipts`
(row-level prediction receipts) · `clip1_lm_money_shot` (adapter answers
`service_lead` on a posting it never saw).

## Standalone demos

| File | What happens on screen |
|---|---|
| `videos/openjev-browser-use.mp4` | Browser use — pulls the live HN "Who is hiring?" thread (260 real postings via Algolia), the 3MB classifier decides each at ~6ms, then Chrome navigates to the model's top pick. |
| `videos/openjev-computer-use.mp4` | Computer use — 20 mixed files in a watched folder; the LM arm sorts them into OUTREACH / FORWARD / JOB_ADS / TRASH and writes reply drafts. Honest misses shown. |
| `videos/openjev-drone-sim.mp4` | Simulated UAV SAR mission — the same adapter re-pointed at `action(PROCEED\|HOLD\|RETURN_HOME\|ABORT)` with zero retraining. The model's margin is the safety gate: PROCEED nominal, HOLD through gust/GNSS-degrade, RETURN_HOME on critical battery. |
| `videos/openjev-scale.mp4` | Scale + parity — LM arm streams 50 unseen dev-split postings with a live Jev-agreement counter: lands 48/50 = 96%, misses marked on screen. |
| `videos/openjev-race.mp4` | Generation vs decision — the raw base model rambles website-builder advice for 3.6s; the adapter answers typed JSON (`service_lead fit=3`) in 1.75s. Same weights. |
| `videos/openjev-polyglot.mp4` | Multilingual — 12 postings in FR/DE/ES/PT/IT/NL/PL/JA on a model trained 100% English. 8/12 right, and every "someone needs a website" lead caught across all 8 languages. |
| `videos/openjev-jevml-router.mp4` | openjev-lm × [JevML](https://github.com/gamesonrblx/JevML) — the decision model routes 5 natural-language tasks to the right ML primitive (PCA / MCMC / text-diffusion / NCA), 5/5, then each primitive runs live in Node. |
| `videos/openjev-retrain-timelapse.mp4` | Watch it learn — the real 400-step CPU retrain log replayed: loss 0.679 → 0.024 in 13.7 min, ending on the eval receipt (64/70 gold, 94.3% Jev agreement). |

## Reproduce

Scripts in `scripts/` — each prints its own header and needs only the repo venv:

```bash
.venvs/openjev/bin/python scripts/demo_scale.py        # streaming eval
.venvs/openjev/bin/python scripts/demo_browser.py      # browser use (needs HN API)
.venvs/openjev/bin/python scripts/demo_triage.py --seed && .venvs/openjev/bin/python scripts/demo_triage.py
.venvs/openjev/bin/python scripts/demo_drone.py        # UAV mission sim
.venvs/openjev/bin/python scripts/demo_race.py         # base vs adapter
.venvs/openjev/bin/python scripts/demo_polyglot.py     # 8 languages
.venvs/openjev/bin/python scripts/demo_retrain.py      # training-log timelapse
.venvs/openjev/bin/python scripts/demo_determinism.py  # 3× same output
```

`scripts/demo_route.py` routes to JevML primitives; it needs Node 22+ and a
JevML checkout with `scripts/jevml_run_primitive.ts` copied in as
`run_primitive.ts`.
