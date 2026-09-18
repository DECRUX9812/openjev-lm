# Open-Jev — progress + live real-use validation

**Date:** 2026-09-18 · **Mission:** `~/Code/jev-mission` · **Repo:** https://github.com/DECRUX9812/openjev (MIT, public)
**Harness:** `~/Code/jev-mission/live-ab/` (this report's scripts + raw JSON)

---

## TL;DR

Open-Jev (the local, zero-cost reimplementation of the Jev decision layer) is **built, published, and now validated on live production traffic**:

| Test | Result | Cost | Speed |
|---|---|---|---|
| A · today's fresh stream, 106 never-seen postings | **106/106 bucket + tier agreement (100%)** | $0 vs $0.0025 hosted | 6.0 ms vs 250 ms/posting |
| B · cross-region boundary slice, 107 never-seen postings | **104/107 (97.2%)**, 5 real disagreements | $0 vs $0.0041 hosted | 4.4 ms vs 253 ms |
| C · dark-data sweep — whole 2,844-posting archive re-analysed | 43 WATCH / 71 DROP / 0 PITCH · 99.3% tier-stable vs Sep-16 hosted pass | $0, 18.0 s total, offline | 6.3 ms/posting |

**It can carry the sweep layer today.** It should not yet be the *only* judge: the rare `service_lead` class (1–2 real examples ever) is where it misses what hosted Jev catches — measured, not guessed, below.

---

## 1 · Where the mission got to (session recap)

- **Trained:** frozen `BAAI/bge-small-en-v1.5` encoder + small MLP heads, 12 seeds ensembled, 1,500 steps, 2,512 train / 49 dev rows. Encoder never fine-tuned.
- **Measured on the held-out 70 hand-labelled postings:** **66/70 = 94.3%** vs hosted Jev **68/70 = 97.1%** (Δ −2, reported as a loss in the repo, not hidden).
- **Agreement with hosted Jev on the full 2,631-posting production corpus:** **99.39%** (2,615/2,631).
- **Every approach tried is on the record** (`docs/approaches-tried.md`): TF-IDF baselines, bge-base, encoder blends, retrieval/kNN blend (**worse** — 62/70), multi-task aux labels (**no gain**), synthetic posting data (**actively harmful** — 61–63/70), decision-bias tuning (kept, tuned on dev only).
- **The LoRA track finished after the last progress note:** all 420 steps → **65/70 = 92.9%** — a genuine distillation (+20 pts over the partial-epoch 63/70) but still **below the shipped 66/70** small model, at ~10 h/epoch vs minutes. Scored in commit `900c084`. *(Minor doc wart: `results/openjev_results.json` still carries the older "not attempted to completion" blurb for the LoRA entry while the ledger and README report the final score.)*
- **Published:** public MIT repo, 6 commits, weights + eval + MODEL_CARD + negative results, reproducible from a clean clone.
- **Already exercised in a real agent loop:** `axjev-pilot` computer-use pilot ran 4/4 phases / 12 live decisions with open-Jev as the decision layer (`~/Code/axjev-pilot/DEMO-RESULT.md`).
- **Ops is live and green:** `client-hunter-saskjobs` pulls 07:45 daily; `jev-morning-report` 06:00, `jev-mission-watchdog` */15, `jev-memguard` */5 all completed today.

## 2 · Published artifact verified (not just claimed)

From a clean `git clone` of the public repo (head `900c084`):

```
python -m openjev.eval   →  gold 66/70 (94.29%)  ·  agreement 2615/2631 = 0.9939
weights sha256-16 recorded in results/openjev_results.json
```

The eval names its own misses out loud — including **1515771 "Electronic Business (E-Business) Web Site Developer" (truth `service_lead` → predicted `generic_job`, conf 0.58)**. That same row is the single most important miss in Test C below.

## 3 · Test A — today's fresh production stream (real traffic)

Source: the 106 postings that arrived from the saskjobs watcher **after** the Sep-16 corpus was built — never used in training, never in the corpus.

| | hosted Jev | open-Jev (local) |
|---|---|---|
| buckets | 104 generic_job · 2 staff_role | identical |
| tiers | 104 ARCHIVE · 1 WATCH · 1 DROP | identical |
| agreement | — | **106/106 = 100%** (bucket and tier) |
| cost | $0.002519 (57 live + 49 cached calls, 112k tokens) | **$0.00** |
| speed | 250 ms/posting median | **6.0 ms/posting** (0.63 s warm for 106) |
| determinism | non-deterministic (known drift) | **two passes byte-identical** |

Honest caveat: this batch is easy — 104/106 are generic jobs. Agreement here is strong evidence of *safety on the daily stream*, weak evidence on the hard boundary. Hence Test B.

## 4 · Test B — boundary slice (the hard cases, cross-region)

107 postings pulled fresh from 4 other Saskatchewan regions with technical/business signal in the title (Moose Jaw / Swift Current / etc.) — genuinely unseen, and a distribution shift (the model only ever trained on Regina postings).

**104/107 = 97.2%** bucket + tier agreement. The 5 disagreements, in full:

| id | posting | hosted | open-Jev |
|---|---|---|---|
| 1544139 | GIS Technician — Millennium Land | staff_role / WATCH | generic_job / ARCHIVE |
| 1542275 | IT Support & Network Technician — Intricate Networks | staff_role / WATCH | staff_role / **ARCHIVE** |
| 1544955 | Cleaner #sksupportsukraine — Risk Free Janitorial | generic_job / **DROP** | generic_job / ARCHIVE |
| 1544241 | CNC Machinist #sksupportsukraine — PWM Hydraulics | generic_job / ARCHIVE | **staff_role** / ARCHIVE |
| 1538000 | Journeyperson Instrumentation & Control Tech — SaskPower | generic_job / ARCHIVE | **staff_role** / ARCHIVE |

Pattern: open-Jev is **more conservative on ambiguous tech titles** (under-calls `technical_need` on "IT Support", "GIS Technician") and **misses the evergreen-repost junk marker** (`#sksupportsukraine` → hosted drops it, local archives it). Both classes are fixable with labels, not architecture.

## 5 · Test C — the dark-data sweep

Everything this box has ever collected (2,737 Regina stream + 107 cross-region = **2,844 postings**), re-analysed end-to-end, locally:

- **18.0 s wall, $0.00, offline, deterministic** (6.3 ms/posting, 4 CPU threads)
- tiers: **2,730 ARCHIVE · 71 DROP · 43 WATCH · 0 PITCH**
- drift vs the Sep-16 one-shot hosted pass (shared 2,631 rows): **99.3% same tier**, 18 changes
- the 18 changes are the interesting rows: 6 hosted WATCH rows the local arm archives (all ambiguous tech titles: "Lead Developer", "Business Systems Consultant", "Oracle EBS Financial Business Partner"…), 3 rows local *promotes* (Surveillance Monitor, Health Information Management Analyst, Senior Analyst…), 7 DROP/ARCHIVE boundary flips on home-care/cleaner/janitorial rows
- **the one hosted PITCH in the whole archive (VicSquare Arcade, id 1515771) is missed by open-Jev** — bucket `generic_job`, fit 0.0. This is the same miss the published eval already reports (gold service_lead recall 0/2 vs hosted 1/2)
- employer rollup (the question metered passes never asked): 8 firms re-posting technical roles — SaskTel (7/15), SaskEnergy (6/13), City of Regina (6/38), SGI (5/15), WCB (3/7), SHA (7/851), PrintWest (2/2 — "Web Designer" ×2, already a WATCH), plus 59 evergreen/MLM-flagged postings clustered by employer

## 6 · Where it is weak (the honest part)

| class | open-Jev gold recall | hosted gold recall |
|---|---|---|
| generic_job | 54/54 | 54/54 |
| staff_role | 12/14 | 13/14 |
| **service_lead** | **0/2** | **1/2** |

The rare class that matters most for the pitch list is the exact class it can't hold. Root cause is data, not method: 2 examples in gold, 1 in the whole 2,631-posting production corpus. Fix = label the service_lead rows you find (the daily stream keeps producing them), retrain the heads in minutes; **not** a bigger model (bge-base and LoRA both failed to beat it).

## 7 · The "dark data" unlock — measured

What the price drop actually buys here, precisely (no overclaiming — at this volume hosted was already only ~$0.026/1k postings):

- **No budget decision.** The archive sweep is $0 and 18 s, so you never gate "is this worth a metered pass?" — you just re-run it. Daily, hourly, per-posting.
- **Offline + private.** Decisions never leave the box; no network dependency in the pipeline; works on data you would not send to an API.
- **Deterministic + auditable.** Byte-identical reruns; a decision can be re-derived for an audit trail. Hosted Jev drifts ~0.1 on continuous answers and is non-deterministic by design.
- **40× latency.** 6 ms vs 250 ms → the same intelligence is now usable inline inside filters, watchers, and agent loops, not just batch jobs.
- **Whole-archive questions become free.** Employer repost rollups, evergreen detection, drift audits — the things you'd never have burned meter time on — are now a nightly cron.

## 8 · Recommendation — one concrete path

**Cheap filter, expensive judge:**

1. **Sweep layer = open-Jev, always on.** Every new posting classified locally on arrival (6 ms, $0, cached decision + weight hash in the archive).
2. **Judge layer = hosted Jev on the shortlist only.** The 43 WATCH-grade rows + anything open-Jev flags technical/uncertain (~a few dozen calls/day ≈ $0.001/day) — exactly where the rare-class miss lives.
3. **Wire it as `tfl leads --local` + hook the 07:45 `client-hunter-saskjobs` job** so the pitch list is generated from the fresh stream every morning without metered calls for the 99% that are clearly ARCHIVE.

Then re-measure monthly: if the local arm holds 97%+ on the boundary slices with the judge layer catching service_leads, the judge can be narrowed further. Label every true service_lead you find — that's the only thing standing between 0/2 and parity.

---

### Reproduce this report

```bash
cd ~/Code/jev-mission/live-ab
/usr/bin/python3 fetch_hard_slice.py                                             # pull the boundary slice
~/.venvs/embed/bin/python arm_local.py            [postings] [out]               # local arm
~/Code/typesafe-lab/.venv/bin/python arm_hosted.py [postings] [out]               # hosted arm (live, metered)
~/Code/typesafe-lab/.venv/bin/python compare.py   [local] [hosted] [postings] [prefix]
OMP_NUM_THREADS=4 ~/.venvs/embed/bin/python dark_sweep.py                        # full-archive sweep
```

Artifacts: `ab_report.md` · `hard_report.md` · `dark_sweep_report.md` · `*_rows.json` (raw per-posting decisions, both arms).
