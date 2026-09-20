"""jevgraphic cli

  make "<idea>"   prompt -> gated, styled poster (SVG + PNG)
  edit <draft>    open the studio on a markdown draft
  gate <draft>    print the gate receipt only
  art             list art archetypes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .art import art_names
from .compose import compose_svg, W, H
from .draft import load_draft
from .gate import get_gate
from .intent import get_intent
from .model import to_json
from .synth import synthesize
from .svgpng import export_png


def _load_evidence(path: str) -> str:
    return Path(path).read_text() if path else ""


def cmd_make(a) -> int:
    evidence = _load_evidence(a.evidence)
    draft = synthesize(a.idea, evidence)
    gate = get_gate()
    t0 = time.perf_counter()
    draft, receipt = gate.decide(draft, evidence)
    decision = get_intent().decide(a.idea, draft, seed=a.seed)
    svg = compose_svg(draft, decision, receipt, W, H)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "-" for c in a.idea.lower())[:40].strip("-")
    svg_path = out / f"{slug}.svg"
    svg_path.write_text(svg)
    (out / f"{slug}.gate.json").write_text(to_json(receipt))
    (out / f"{slug}.decision.json").write_text(to_json(decision))
    print(f"[gate] checked={receipt.checked} kept={receipt.kept} "
          f"refused={receipt.refused} copied={receipt.copied} "
          f"cost=${receipt.cost_usd:.6f} {receipt.elapsed_ms:.0f}ms model={receipt.model}")
    print(f"[jev] layout={decision.layout} style={decision.style} art={decision.art} "
          f"confidence={decision.confidence} ({decision.backend})")
    print(f"[svg]  {svg_path}")
    if a.png:
        png_path = out / f"{slug}.png"
        how = export_png(str(svg_path), str(png_path))
        print(f"[png]  {png_path} via {how}" if how else
              "[png]  no exporter found (install chrome or rsvg-convert)")
    return 0


def cmd_gate(a) -> int:
    evidence = _load_evidence(a.evidence)
    draft = load_draft(a.draft)
    draft, receipt = get_gate().decide(draft, evidence)
    print(to_json(receipt))
    for c in draft.refused_claims():
        print(f"REFUSED: {c.text[:90]}  [{c.note}]", file=sys.stderr)
    return 0


def cmd_edit(a) -> int:
    from .studio import serve
    return serve(draft_path=a.draft, evidence_path=a.evidence,
                 host=a.host, port=a.port)


def cmd_art(a) -> int:
    print("\n".join(art_names()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jevgraphic",
                                 description="prompt-to-poster studio gated by jev")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="idea -> poster")
    m.add_argument("idea")
    m.add_argument("--evidence", default="")
    m.add_argument("--out", default="out")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--png", action="store_true")
    m.set_defaults(fn=cmd_make)

    e = sub.add_parser("edit", help="studio server on a draft")
    e.add_argument("draft")
    e.add_argument("--evidence", default="")
    e.add_argument("--host", default="127.0.0.1")
    e.add_argument("--port", type=int, default=8791)
    e.set_defaults(fn=cmd_edit)

    g = sub.add_parser("gate", help="gate a draft, print receipt")
    g.add_argument("draft")
    g.add_argument("--evidence", default="")
    g.set_defaults(fn=cmd_gate)

    ar = sub.add_parser("art", help="list art archetypes")
    ar.set_defaults(fn=cmd_art)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
