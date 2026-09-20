"""Jev intent — given an idea (and its gated claims), choose the layout,
style pack and art archetype. Runs once per draft, exactly like the gate.

Local backend: deterministic hashed bag-of-words cosine between the idea
text (+ kept claims) and a descriptor line per option. No dependencies,
no network, $0.000000 — and the same idea always picks the same look.
Hosted backend: if JEV_API_URL + JEV_API_KEY are set, Jev itself scores
each option; local stays as fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from typing import Dict, List, Tuple

from .art import ART_BLURBS, art_names
from .layouts import LAYOUT_BLURBS, layout_names
from .model import Decision, Draft
from .styles import STYLE_PACKS, style_names

_DIM = 512


def _vec(text: str) -> List[float]:
    v = [0.0] * _DIM
    for tok in text.lower().replace("/", " ").split():
        for piece in tok.split("-"):
            if not piece:
                continue
            h = int(hashlib.blake2b(piece.encode(), digest_size=4).hexdigest(), 16)
            v[h % _DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _pick(query: str, blurbs: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
    q = _vec(query)
    scores = {name: _cos(q, _vec(f"{name.replace('-', ' ')} {blurb}"))
              for name, blurb in blurbs.items()}
    best = max(scores, key=scores.get)
    return best, scores


class LocalIntent:
    name = "jev-local"

    def decide(self, idea: str, draft: Draft, seed: int = 0) -> Decision:
        query = " ".join([idea, draft.title, draft.subtitle,
                          *[c.text for c in draft.kept_claims()]])
        layout, ls = _pick(query, LAYOUT_BLURBS)
        style, ss = _pick(query, {n: p.label for n, p in STYLE_PACKS.items()})
        art, ar = _pick(query, ART_BLURBS)
        ranked = sorted(ls.values())
        margin = ranked[-1] - ranked[-2] if len(ranked) > 1 else ranked[-1]
        return Decision(layout=layout, style=style, art=art,
                        glyphs=True, confidence=round(max(0.0, margin * 8), 3),
                        scores={"layout": ls, "style": ss, "art": ar},
                        backend="local")


class HostedIntent:
    def __init__(self):
        self.url = os.environ.get("JEV_API_URL", "")
        self.key = os.environ.get("JEV_API_KEY", "")
        self.local = LocalIntent()

    def decide(self, idea: str, draft: Draft, seed: int = 0) -> Decision:
        if not (self.url and self.key):
            return self.local.decide(idea, draft, seed)
        try:
            body = json.dumps({
                "task": "choose poster options",
                "idea": idea,
                "claims": [c.text for c in draft.kept_claims()],
                "layouts": layout_names(), "styles": style_names(),
                "art": art_names(),
            }).encode()
            req = urllib.request.Request(
                self.url, data=body,
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                v = json.loads(r.read())
            return Decision(layout=v["layout"], style=v["style"], art=v["art"],
                            glyphs=bool(v.get("glyphs", True)),
                            confidence=float(v.get("confidence", 0)),
                            backend="hosted")
        except Exception:
            return self.local.decide(idea, draft, seed)


def get_intent() -> object:
    if os.environ.get("JEV_API_KEY") and os.environ.get("JEV_API_URL"):
        return HostedIntent()
    return LocalIntent()
