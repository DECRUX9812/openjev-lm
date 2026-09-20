"""The gate. Runs once per draft — never per render.

Rule of the house: a number may only appear on the poster if that exact
token appears in the evidence document. Claims that fail are refused, not
edited. Style and layout swaps afterwards are data-only and cost $0.

Backends:
  local   — dependency-free verbatim verification (default, $0.000000)
  hosted  — Jev API when JEV_API_URL + JEV_API_KEY are set
"""
from __future__ import annotations

import os
import re
import time
import urllib.request
import json
from typing import List, Tuple

from .draft import numbers_in
from .model import Claim, Draft, GateReceipt

WORD_RE = re.compile(r"[a-zA-Z']+")
STOP = set("the a an of to in on at for and or is are was were it its with we you that this by as be from".split())


def _norm_num(tok: str) -> str:
    return tok.replace(",", "").replace("$", "").strip().rstrip(".")


def _evidence_numbers(evidence: str) -> set:
    """Every numeric token present in evidence, plus comma/$-stripped forms."""
    out = set()
    for m in numbers_in(evidence):
        out.add(m.strip())
        out.add(_norm_num(m))
    return out


def _content_words(text: str) -> set:
    return {w.lower() for w in WORD_RE.findall(text)} - STOP


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def prose_score(claim: str, evidence_sents: List[str]) -> float:
    """Best-case content-word overlap between the claim and any evidence sentence."""
    cw = _content_words(claim)
    if not cw:
        return 1.0
    best = 0.0
    for s in evidence_sents:
        sw = _content_words(s)
        if not sw:
            continue
        best = max(best, len(cw & sw) / len(cw))
    return best


def verify_claim(claim: Claim, evidence: str, ev_numbers: set,
                 ev_sents: List[str]) -> Claim:
    """Score one claim. Mutates and returns it."""
    nums = numbers_in(claim.text)
    missing = [n for n in nums
               if n.strip() not in ev_numbers and _norm_num(n) not in ev_numbers]
    claim.figure = 1.0 if not nums else (len(nums) - len(missing)) / len(nums)
    claim.prose = prose_score(claim.text, ev_sents)
    if missing:
        claim.kept = False
        claim.note = f"unverified number(s): {', '.join(missing)}"
    elif claim.prose < 0.25 and claim.kind == "claim":
        claim.kept = False
        claim.note = "no support in evidence"
    else:
        claim.kept = True
        claim.note = ""
    return claim


class LocalGate:
    """Verbatim verification against the evidence document. Costs nothing."""

    name = "jev-local"

    def decide(self, draft: Draft, evidence: str) -> Tuple[Draft, GateReceipt]:
        t0 = time.perf_counter()
        receipt = GateReceipt(model=self.name, backend="local")
        if not evidence:
            # no evidence -> nothing may carry a number
            for c in draft.claims:
                if numbers_in(c.text):
                    c.kept, c.note = False, "no evidence supplied"
                    c.figure = 0.0
                receipt.checked += 1
        else:
            ev_numbers = _evidence_numbers(evidence)
            ev_sents = _sentences(evidence)
            norm_ev = re.sub(r"\s+", " ", evidence)
            norm_ev = re.sub(r"\*\*", "", re.sub(r"`([^`]*)`", r"\1", norm_ev))
            for c in draft.claims:
                verify_claim(c, evidence, ev_numbers, ev_sents)
                receipt.checked += 1
                if c.kept and re.sub(r"\s+", " ", c.text) in norm_ev:
                    receipt.copied += 1
        receipt.kept = len(draft.kept_claims())
        receipt.refused = len(draft.refused_claims())
        receipt.cost_usd = 0.0
        receipt.elapsed_ms = (time.perf_counter() - t0) * 1000
        return draft, receipt


class HostedGate:
    """Calls the hosted Jev endpoint. Falls back to local on any failure."""

    def __init__(self, url: str = "", key: str = ""):
        self.url = url or os.environ.get("JEV_API_URL", "")
        self.key = key or os.environ.get("JEV_API_KEY", "")
        self.local = LocalGate()

    def decide(self, draft: Draft, evidence: str) -> Tuple[Draft, GateReceipt]:
        if not (self.url and self.key):
            return self.local.decide(draft, evidence)
        t0 = time.perf_counter()
        try:
            body = json.dumps({
                "claims": [c.text for c in draft.claims],
                "evidence": evidence,
            }).encode()
            req = urllib.request.Request(
                self.url, data=body,
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as r:
                verdict = json.loads(r.read())
            kept = set(verdict.get("kept", []))
            receipt = GateReceipt(model=verdict.get("model", "jev"),
                                  backend="hosted",
                                  cost_usd=float(verdict.get("cost_usd", 0.0)))
            for i, c in enumerate(draft.claims):
                c.kept = i in kept or c.text in kept
                if not c.kept:
                    c.note = "refused by jev"
                receipt.checked += 1
            receipt.kept = len(draft.kept_claims())
            receipt.refused = len(draft.refused_claims())
            receipt.elapsed_ms = (time.perf_counter() - t0) * 1000
            return draft, receipt
        except Exception:
            # never let a network hiccup fabricate a verdict
            return self.local.decide(draft, evidence)


def get_gate() -> object:
    if os.environ.get("JEV_API_KEY") and os.environ.get("JEV_API_URL"):
        return HostedGate()
    return LocalGate()
