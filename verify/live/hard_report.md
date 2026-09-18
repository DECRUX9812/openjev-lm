# Live A/B — hosted Jev vs open-Jev, 106 fresh postings

Source: `~/.hermes/scripts/leads.jsonl`, postings pulled 2026-09-17/18 (today's 07:45 run).
None of these ids exist in the training corpus or the gold set.

- **Bucket agreement: 104/107 = 97.2%**
- Tier agreement (same PITCH/WATCH/ARCHIVE/DROP rules): 104/107 = 97.2%

| | hosted Jev (jev-latest) | open-Jev (local) |
|---|---|---|
| buckets | {'generic_job': 103, 'staff_role': 4} | {'generic_job': 102, 'staff_role': 5} |
| tiers | {'ARCHIVE': 103, 'WATCH': 3, 'DROP': 1} | {'ARCHIVE': 106, 'WATCH': 1} |
| latency | 253.0 ms/posting (median, network) | 41.1 ms/posting (warm, CPU) |
| cost | $0.004067 / 107 calls | $0.00 |
| determinism | repeated draws can move (documented) | byte-identical across passes |

## Disagreements (5)

- `1544139` **GIS Technician** — Millennium Land Ltd.  
  hosted: staff_role / WATCH (conf 0.91) · local: generic_job / ARCHIVE (conf 0.6635)
- `1542275` **IT Support & Network Technician** — Intricate Networks Inc.  
  hosted: staff_role / WATCH (conf 0.98) · local: staff_role / ARCHIVE (conf 0.957)
- `1544955` **Cleaner #sksupportsukraine** — RISK FREE CLEAN JANITORIAL  
  hosted: generic_job / DROP (conf 0.94) · local: generic_job / ARCHIVE (conf 0.9633)
- `1544241` **CNC Machinist - #sksupportsukraine** — PWM HYDRAULICS  
  hosted: generic_job / ARCHIVE (conf 0.96) · local: staff_role / ARCHIVE (conf 0.5739)
- `1538000` **Journeyed Trade, Instrumentation and Control Tech** — SaskPower  
  hosted: generic_job / ARCHIVE (conf 0.74) · local: staff_role / ARCHIVE (conf 0.5332)

## Lead worklist (PITCH/WATCH in either arm)

- **ARCHIVE** (hosted WATCH) `1544139` GIS Technician — Millennium Land Ltd. · fit 1.0
- **ARCHIVE** (hosted WATCH) `1542275` IT Support & Network Technician — Intricate Networks Inc. · fit 1.0
- **WATCH** (hosted WATCH) `1544050` Instructor: Computer Networking Technician — Suncrest College. · fit 1.0
