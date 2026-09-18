# FACTS — the numbers this launch is allowed to use (generated from raw run files)

- Jev (hosted, the thing being reproduced): **68/70 = 97.1%** on the 70 hand labels
- viral RLCD artifact: 54/70 = 77.1% = exactly the majority-class prior; service_lead 0/2, staff_role 3/14, confidence 0.61
- open-Jev LM (0.5B + 2.16M LoRA, 400 steps, 89 min, CPU): method A 65/70 = 92.9%, method B 92.9% (independent harness; A scored step 386, B step 400; identical buckets on all 70 rows)
  - per-class recall: service_lead 1/2, staff_role 12/14, generic_job 52/54
- 0.5B untrained: 72.9% (below the prior, zero rare-class recall)
- run-night1 (skewed corpus, 420 steps): all-70 65/70, clean-66 62/66 (4 gold rows share title+employer with its training data)
- open-Jev classifier (bge-small frozen + heads, 3 MB, 6 ms): 66/70 = 94.3%, agreement with Jev on the 2,631-row production corpus 2615/2631 = 99.39%
- corpora: train_jev 2,591 (Jev-labelled), dev 286, gold 70; hard-case 960 Jev-labelled ($0.0425); verified 13/13
