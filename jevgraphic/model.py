"""Data model for jevgraphic — drafts, claims, gate receipts, decisions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Claim:
    """One line of poster copy. `figure`/`prose` are gate scores in [0,1]."""
    text: str
    figure: float = 0.0
    prose: float = 0.0
    kept: bool = True
    note: str = ""               # why it was refused / where the number came from
    kind: str = "claim"          # claim | stat | headline


@dataclass
class Draft:
    """Everything the canvas may draw. Produced by a markdown draft or the synthesizer."""
    title: str
    subtitle: str = ""
    claims: List[Claim] = field(default_factory=list)
    source: str = ""             # path of the draft file or 'idea'
    hero: str = ""               # optional hero stat e.g. "92.9%"

    def kept_claims(self) -> List[Claim]:
        return [c for c in self.claims if c.kept]

    def refused_claims(self) -> List[Claim]:
        return [c for c in self.claims if not c.kept]


@dataclass
class GateReceipt:
    """The gate runs once per draft; this is its proof-of-work."""
    checked: int = 0             # claims examined
    kept: int = 0
    refused: int = 0
    copied: int = 0              # claims lifted verbatim from evidence
    cost_usd: float = 0.0
    elapsed_ms: float = 0.0
    model: str = "jev-local"
    backend: str = "local"


@dataclass
class Decision:
    """Jev's creative choices for a draft. Deterministic for a given (idea, seed)."""
    layout: str
    style: str
    art: str
    glyphs: bool = True
    confidence: float = 0.0
    scores: dict = field(default_factory=dict)   # option -> score, for the receipt
    backend: str = "local"


def to_json(obj) -> str:
    return json.dumps(asdict(obj), indent=2, sort_keys=True)
