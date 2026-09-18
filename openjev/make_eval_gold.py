"""openjev/make_eval_gold.py — write the 70 held-out gold postings in the COMPACT prompt format.

Standalone so the base-model eval can run before the synthetic corpus is finished.
"""
import json
from pathlib import Path

ROOT = Path("/home/decrux/Code/jev-repro-test")
LAB = Path("/home/decrux/Code/typesafe-lab")
OUT = ROOT / "openjev" / "data"

BOOLS = ["technical_need", "business_buyer", "small_firm_doable", "pay_stated", "evergreen_repost"]
KEYS = "bucket(service_lead|staff_role|generic_job|junk) " + " ".join(BOOLS) + " fit(0-4)"


def clip(s, n=90):
    return " ".join(str(s or "").split())[:n]


def prompt_for(t, e, p):
    return ("<|im_start|>system\nDecide.<|im_end|>\n<|im_start|>user\n"
            f"title: {clip(t)}\nemployer: {clip(e)}\npay: {clip(p, 40)}\n{KEYS}<|im_end|>\n"
            "<|im_start|>assistant\n")


def main():
    corpus = {r["id"]: r for r in json.load(open(LAB / "runs" / "leads_corpus_full.json"))["rows"]}
    gold = json.load(open(LAB / "data" / "leads.gold.json"))["labels"]
    out = []
    for pid, label in gold.items():
        r = corpus.get(pid)
        if not r:
            continue
        out.append({"id": pid, "title": r.get("title"), "gold": label, "jev_bucket": r.get("bucket"),
                    "prompt": prompt_for(r.get("title", ""), r.get("employer", ""), r.get("pay") or ""),
                    "target": ""})
    path = OUT / "eval_gold.jsonl"
    with path.open("w") as fh:
        for rec in out:
            fh.write(json.dumps(rec) + "\n")
    print(f"eval_gold: {len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
