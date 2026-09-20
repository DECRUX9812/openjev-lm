"""compose — draft + decision + style -> one self-contained SVG poster."""
from __future__ import annotations

import hashlib


from .art import render_art
from .layouts import LAYOUTS
from .model import Decision, Draft, GateReceipt
from .styles import STYLE_PACKS, DEFAULT_STYLE
from .textures import TEXTURES

W, H = 1440, 1080


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def compose_svg(draft: Draft, decision: Decision, receipt: GateReceipt,
                width: int = W, height: int = H) -> str:
    pack = STYLE_PACKS.get(decision.style, STYLE_PACKS[DEFAULT_STYLE])
    seed = _seed(draft.title, decision.art, decision.style)
    tex = TEXTURES.get(pack.texture, lambda *a: "")(width, height, pack.ink, seed)
    art_big = render_art(decision.art, int(width * 0.62), int(height * 0.16),
                         int(width * 0.34), int(height * 0.5),
                         pack.ink, pack.accent, pack.accent2, seed)
    art_small = render_art(decision.art, int(width * 0.80), int(height * 0.02),
                           int(width * 0.15), int(height * 0.15),
                           pack.ink, pack.accent, pack.accent2, seed)
    ctx = {"draft": draft, "pack": pack, "decision": decision, "receipt": receipt,
           "glyphs": decision.glyphs, "seed": seed, "w": width, "h": height,
           "art_svg": art_big, "art_svg_small": art_small}
    body = LAYOUTS.get(decision.layout, LAYOUTS["poster"])(ctx)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{pack.font_body}">'
        f'<rect width="{width}" height="{height}" fill="{pack.bg}"/>'
        f'{tex}{body}</svg>')
