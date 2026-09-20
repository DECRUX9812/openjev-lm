"""Draft parsing: markdown draft -> Draft; idea text -> candidate claims.

A draft file is plain markdown:

    # Title
    subtitle line
    - **92.9% accuracy** (65/70) on the gold set
    - a plain claim line
    ## Notes            <- ignored
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .model import Claim, Draft

NUM_RE = re.compile(
    r"\$?\d[\d,]*(?:\.\d+)?\s*(?:%|ms\b|s\b|MB\b|GB\b|KB\b|x\b|vCPUs?\b|B\b|M\b|K\b|k\b)?")
BULLET_RE = re.compile(r"^\s*[-*•]\s+")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+")


def numbers_in(text: str) -> List[str]:
    """All numeric tokens in a string, normalized (commas kept, unit glued)."""
    return [m.group(0).strip().rstrip(",") for m in NUM_RE.finditer(text)]


def parse_draft(markdown: str, source: str = "") -> Draft:
    title, subtitle, claims = "", "", []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line or line in ("---", "***"):
            continue
        if HEADING_RE.match(line):
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("#").strip()
            if level == 1 and not title:
                title = text
            # deeper headings are section furniture — not poster copy
            continue
        if BULLET_RE.match(line):
            text = BULLET_RE.sub("", line)
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # unbold, keep text
            claims.append(Claim(text=text, kind="stat" if numbers_in(text) else "claim"))
            continue
        if not subtitle and not claims:
            subtitle = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        else:
            claims.append(Claim(text=re.sub(r"\*\*(.+?)\*\*", r"\1", line),
                                kind="stat" if numbers_in(line) else "claim"))
    return Draft(title=title or "Untitled", subtitle=subtitle, claims=claims, source=source)


def load_draft(path: str) -> Draft:
    p = Path(path)
    return parse_draft(p.read_text(), source=str(p))
