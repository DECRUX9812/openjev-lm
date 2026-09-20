"""synthesize — turn a raw idea into a Draft whose claims are real.

The synthesizer never invents a number: candidate claims are the most
salient evidence sentences (numbers first, ranked by rarity), plus the
idea's own prose. The gate then keeps only what verifies.
"""
from __future__ import annotations

import re
from typing import List

from .draft import numbers_in
from .gate import _sentences, _content_words
from .model import Claim, Draft

MAX_CLAIMS = 8
MAX_CLAIM_CHARS = 170


def _clean(s: str) -> str:
    """Strip markdown furniture and clip to a poster-sized clause."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"[#>*_|]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -*•")
    if len(s) > MAX_CLAIM_CHARS:
        cut = s[:MAX_CLAIM_CHARS]
        # end at the last clause boundary inside the window
        for sep in (". ", ", ", "; ", " — ", " "):
            idx = cut.rfind(sep)
            if idx > MAX_CLAIM_CHARS * 0.55:
                cut = cut[:idx + (1 if sep == ". " else 0)]
                break
        s = cut.strip(" ,;—-")
        if not s.endswith(('.', '%', ')')):
            s += "…"
    return s


def _rank_evidence_sentences(idea: str, evidence: str, limit: int) -> List[str]:
    """Salience = carries numbers + shares content words with the idea."""
    iw = _content_words(idea)
    scored = []
    noisy = re.compile(r"[0-9a-f]{12,}|sha256|https?://|\.\w{2,4}\b\s|@|\$\(|\{\w+\}")
    for s in _sentences(evidence):
        s = _clean(s)
        if len(s) < 24 or len(s) > MAX_CLAIM_CHARS:
            continue
        if noisy.search(s):   # checksums, filenames, urls read as machine noise
            continue
        nums = numbers_in(s)
        overlap = len(_content_words(s) & iw)
        score = (2.0 if nums else 0.0) + min(2.0, 0.35 * len(nums)) + overlap * 0.6
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda t: -t[0])
    seen_words, out = set(), []
    skip = _content_words(idea)
    for _, s in scored:
        sw = _content_words(s)
        # drop evidence sentences that just restate the title/idea
        if skip and len(sw & skip) / max(1, len(sw)) > 0.7 and not numbers_in(s):
            continue
        key = frozenset(list(sw)[:6])
        if key & seen_words and len(s) > 120:
            continue
        seen_words |= key
        out.append(s)
        if len(out) >= limit:
            break
    return out


def synthesize(idea: str, evidence: str = "", source: str = "idea") -> Draft:
    """Idea -> Draft. `idea` becomes title (+ subtitle); evidence yields claims."""
    idea = idea.strip()
    # split "Title: subtitle" or "Title — subtitle" or "Title. Rest"
    m = re.split(r"\s*[:—–]\s*|\.\s+", idea, maxsplit=1)
    title = m[0].strip() if m else idea
    subtitle = m[1].strip() if len(m) > 1 else ""
    claims: List[Claim] = []
    if evidence:
        for s in _rank_evidence_sentences(idea, evidence, MAX_CLAIMS):
            claims.append(Claim(text=s, kind="stat" if numbers_in(s) else "claim"))
    else:
        # no evidence: idea fragments are the claims (gate will refuse numbers)
        for frag in re.split(r"[,;]\s+|\s+and\s+", subtitle or ""):
            if frag.strip():
                claims.append(Claim(text=frag.strip()))
    return Draft(title=title, subtitle=subtitle, claims=claims, source=source)
