# Data

| file | rows | what it is |
|---|---|---|
| `train_jev.jsonl` | 2,591 | training corpus; `target` = Jev's own API answer, verbatim |
| `dev_jev.jsonl` | 286 | held-out dev split used for training decisions |
| `eval_gold.jsonl` | 70 | hand-labelled gold set; held out of every training run |

Row shape: `{"id", "source", "prompt", "target", "jev", "family"}` where `target` carries the
model's full typed answer (bucket + five booleans + fit, each with a probability).

`run-final/train_jev.jsonl` is the exact artifact used by `train_v2.py`; byte-identical to
`data/train_jev.jsonl` (sha256 `77b33e573c1b46fa`, first 16 hex).

**Class balance (train):** staff_role 958 · generic_job 1,088 · service_lead 342 · junk 203.

**Why the balance matters.** An earlier overnight corpus was majority-class heavy (~70%
generic_job, ~0.5% service_lead) and the model trained on it learned exactly that: it could not
call the rare class at all. Class balance is not a statistics nicety here — it is behaviour. The
hard-case rows (960 Jev-labelled boundary postings) exist for the same reason.

**Verification.** `verify_corpus.py` re-checks every claim we make about this data: that each
target is byte-identical to a recorded Jev answer, that no eval id appears in training, that no
(title, employer) pair crosses the split, and that the gold set is disjoint. Current status:
13/13 checks pass. Run it yourself — it takes seconds and needs no network.

**Known caveat.** An early corpus (the "night one" run, kept in `runs/` for the record) shared
four (title, employer) pairs with gold rows; those ids are listed in `runs/AUDIT-DATA.md` and the
cleaned numbers (62/66) are reported next to the raw ones. The shipped corpus has zero leaks.
