# Live A/B — hosted Jev vs open-Jev, 106 fresh postings

Source: `~/.hermes/scripts/leads.jsonl`, postings pulled 2026-09-17/18 (today's 07:45 run).
None of these ids exist in the training corpus or the gold set.

- **Bucket agreement: 106/106 = 100.0%**
- Tier agreement (same PITCH/WATCH/ARCHIVE/DROP rules): 106/106 = 100.0%

| | hosted Jev (jev-latest) | open-Jev (local) |
|---|---|---|
| buckets | {'generic_job': 104, 'staff_role': 2} | {'generic_job': 104, 'staff_role': 2} |
| tiers | {'ARCHIVE': 104, 'DROP': 1, 'WATCH': 1} | {'ARCHIVE': 104, 'DROP': 1, 'WATCH': 1} |
| latency | 250.0 ms/posting (median, network) | 56.6 ms/posting (warm, CPU) |
| cost | $0.002519 / 106 calls | $0.00 |
| determinism | repeated draws can move (documented) | byte-identical across passes |

## Disagreements (0)


## Lead worklist (PITCH/WATCH in either arm)

- **WATCH** (hosted WATCH) `1544645` Database Analyst — City of Regina · fit 1.0
