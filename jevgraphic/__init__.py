"""jevgraphic — prompt-to-poster studio gated by Jev.

Type an idea; Jev decides the layout, style pack and art archetype once;
every number on the canvas is verified verbatim against an evidence document
before it is allowed to render. Re-styling is data-only and costs nothing.

    python3 -m jevgraphic.cli make "open-Jev: local judgment at $0" \
        --evidence paper/openjev-paper.md --out out/

    python3 -m jevgraphic.cli edit draft.md --evidence paper.md --port 8791
"""

__version__ = "0.1.0"
