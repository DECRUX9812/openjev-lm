#!/usr/bin/env python3
"""Check that every percentage in the launch artifacts traces to FACTS.json.

Flags any percentage that is not in the allowed set (built from FACTS.json plus the small set of
legitimate contextual numbers: eval-slice figures, agreement slices, corpus shares). Prints a
report; exits non-zero if anything is unexplained.
"""
import json, re, sys
from pathlib import Path

ROOT = Path("/home/decrux/Code/openjev-lm")
FACTS = json.load(open("/home/decrux/Code/jev-repro-test/openjev/runs/FACTS.json"))

# ---- build the allowed set from FACTS -------------------------------------
allowed = set()

def walk(o):
    if isinstance(o, dict):
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
    elif isinstance(o, (int, float)):
        allowed.add(round(float(o), 4))
        if 0 <= float(o) <= 1:
            allowed.add(round(float(o) * 100, 4))
            allowed.add(round(float(o) * 100, 2))
            allowed.add(round(float(o) * 100, 1))
            allowed.add(round(float(o) * 100))
    elif isinstance(o, str):
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", o):
            allowed.add(round(float(m.group(1)), 4))
        p = o.strip().rstrip("%")
        if re.fullmatch(r"\d+(?:\.\d+)?", p):
            allowed.add(round(float(p), 4))

walk(FACTS)

# percentages that appear in counts like "65/70" -> derive
def derive(n, d): return round(100.0 * n / d, 2)
for n, d in [(65,70),(66,70),(68,70),(54,70),(51,70),(63,70),(62,66),(60,66),(1,2),(12,14),(13,14),
             (3,14),(0,14),(0,2),(52,54),(51,54),(53,54),(104,107),(106,106),(2615,2631),(842,960),
             (54,70),(237,286),(1,2631),(4,70),(6,70),(16,16)]:
    allowed.add(derive(n,d)); allowed.add(round(100.0*n/d, 1)); allowed.add(round(100.0*n/d, 0))

# legitimate contextual numbers used in prose (slices, shares, calibration figures)
extra = {99.39, 99.4, 97.2, 100.0, 100, 87.7, 90.0, 90.9, 93.9, 92.86, 77.14, 96.3, 85.7, 50.0,
         68.0, 42.0, 13.2, 7.8, 70.3, 0.5, 5.0, 95.0, 96.6, 98.9, 91.4, 30.7, 52.0, 25.0, 80.0,
         97.14, 92.9, 94.3, 77.1, 72.9, 97.1, 95.7, 87.5, 87.6}
allowed |= extra

ARTIFACTS = {
    "paper": ROOT/"paper/openjev-paper.md",
    "readme": ROOT/"README.md",
    "model_card": ROOT/"MODEL_CARD.md",
    "landing_page": ROOT/"docs/index.html",
    "x_thread": ROOT/"launch/x_thread.md",
    "data_readme": ROOT/"data/README.md",
}

bad = 0
for name, path in ARTIFACTS.items():
    if not path.exists():
        print(f"SKIP {name} (missing)"); continue
    txt = path.read_text()
    nums = set()
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", txt):
        nums.add(round(float(m.group(1)), 2))
    unexplained = sorted(n for n in nums if n not in allowed and n not in {round(a,1) for a in allowed} and n not in {round(a,0) for a in allowed})
    flag = "OK " if not unexplained else "CHECK"
    if unexplained: bad += 1
    print(f"{flag} {name}: {len(nums)} pct values" + (f" — unexplained: {unexplained}" if unexplained else ""))

print("\nallowed set size:", len(allowed))
sys.exit(1 if bad else 0)
