# open-Jev — standalone tweets

_Author: Ritesh Patel — independent researcher, Regina, Canada. Not affiliated with TypeSafe. {REPO} is a link placeholder. Each tweet is standalone and ready to post._

**Tweet 1 — the viral number was the prior**
The viral "reproduce Jev" artifact scored 54/70 = 77.1%. 54 of the 70 test rows are generic_job — so always answering "generic_job" scores exactly 77.1%. I trained the actual thing instead: 65/70 = 92.9%, CPU-only, $0 per call. {REPO}

**Tweet 2 — dark data**
When inference costs $0 and a decision takes 6 ms, the question stops being "can I afford to classify this?" and becomes "why not classify everything I ever collected?" Dark data doesn't need a bigger model. It needs a cheaper one. {REPO}

**Tweet 3 — the recipe**
Overnight on 6 vCPU, no GPU: Qwen2.5-0.5B + a 2.16M-param LoRA, 400 steps, 89.3 minutes. On a held-out judgment task: 65/70 = 92.9%, vs 68/70 = 97.1% for the hosted model it learned from. #localllama {REPO}

**Tweet 4 — the encoder**
A 3 MB frozen-encoder arm agrees with Jev on 2,615/2,631 postings — 99.39% — at 6.0 ms per posting. The 0.5B LM arm: 65/70 = 92.9% on the hand-labelled set (Jev: 68/70). Both open, MIT. {REPO}

**Tweet 5 — the receipts**
Two independent eval harnesses scored the same 0.5B adapter on the same 70 held-out rows: 92.9% both, agreeing row-for-row. The untrained 0.5B: 51/70 = 72.9% — below the class prior. Receipts in the repo. {REPO}
