"""Layout engines. Each consumes the gate-kept claims and emits SVG body
fragments inside a shared frame. Layouts draw NOTHING unverified — the
gate already removed refused claims before we get here.

Signature: fn(ctx) -> svg string.
ctx keys: draft, pack, art_svg (rendered art group), glyphs, seed, w, h,
          receipt (GateReceipt), decision.
"""
from __future__ import annotations

import html
import math
import random
import re
from typing import Dict, List

from .draft import numbers_in

GLYPHS = ["◆", "▲", "●", "■", "✦", "◈", "ψ", "Ω"]


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def wrap(text: str, width_chars: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width_chars and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def emphasis(text: str, accent: str, ink: str) -> str:
    """Wrap numeric tokens in <tspan> accent — the figure always pops."""
    out, last = [], 0
    for m in re.finditer(r"\$?\d[\d,]*(?:\.\d+)?\s*(?:%|ms|s|MB|GB|x|B|M|K|vCPUs?)?", text):
        out.append(esc(text[last:m.start()]))
        out.append(f'<tspan fill="{accent}" font-weight="bold">{esc(m.group(0).strip())}</tspan>')
        last = m.end()
    out.append(esc(text[last:]))
    return "".join(out)


def hero_stat(draft) -> str:
    """The biggest verified number available, else the title."""
    if draft.hero:
        return draft.hero
    best = ""
    for c in draft.kept_claims():
        for n in numbers_in(c.text):
            if "%" in n:
                return n
            if not best:
                best = n
    return best


def _footer(ctx, y) -> str:
    r = ctx["receipt"]
    s = (f"{r.kept} numbers verified · {r.refused} refused · "
         f"gate ${r.cost_usd:.6f} · {r.elapsed_ms:.0f} ms · layout {ctx['decision'].layout} · "
         f"chosen by jev ({ctx['decision'].backend})")
    return (f'<text class="gate-footer" x="{ctx["w"] / 2}" y="{y}" text-anchor="middle" '
            f'font-size="15" fill="{ctx["pack"].muted}" '
            f'font-family="{ctx["pack"].font_body}">{esc(s)}</text>')


def _claim_text(ctx, c, x, y, size, wch, colour=None, weight="") -> str:
    pack = ctx["pack"]
    colour = colour or pack.ink
    parts = []
    for i, ln in enumerate(wrap(c.text, wch)):
        parts.append(
            f'<text x="{x}" y="{y + i * size * 1.32:.0f}" font-size="{size}" '
            f'fill="{colour}" font-family="{pack.font_body}" font-weight="{weight}">'
            f'{emphasis(ln, pack.accent, colour)}</text>')
    return "".join(parts), len(wrap(c.text, wch))


def _glyph(ctx, i, x, y, size) -> str:
    if not ctx["glyphs"]:
        return ""
    g = GLYPHS[i % len(GLYPHS)]
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{ctx["pack"].accent2}" '
            f'font-family="{ctx["pack"].font_body}">{g}</text>')


def poster(ctx) -> str:
    pack, w, h = ctx["pack"], ctx["w"], ctx["h"]
    d = ctx["draft"]
    P = []
    P.append(ctx["art_svg"])
    # title
    ty = h * 0.14
    for i, ln in enumerate(wrap(d.title, 34)):
        P.append(f'<text x="{w * 0.06}" y="{ty + i * 64}" font-size="58" '
                 f'font-family="{pack.font_display}" fill="{pack.ink}" '
                 f'font-weight="bold">{esc(ln)}</text>')
    ty += len(wrap(d.title, 34)) * 64
    if d.subtitle:
        for i, ln in enumerate(wrap(d.subtitle, 62)):
            P.append(f'<text x="{w * 0.06}" y="{ty + i * 26}" font-size="20" '
                     f'fill="{pack.muted}" font-family="{pack.font_body}">{esc(ln)}</text>')
        ty += len(wrap(d.subtitle, 62)) * 26 + 10
    hero = hero_stat(d)
    if hero:
        P.append(f'<text x="{w * 0.06}" y="{ty + h * 0.17}" font-size="{h * 0.17:.0f}" '
                 f'font-family="{pack.font_display}" fill="{pack.accent}" '
                 f'font-weight="bold">{esc(hero)}</text>')
        ty += h * 0.21
    yy = max(ty + 30, h * 0.56)
    for i, c in enumerate(d.kept_claims()[:5]):
        P.append(_glyph(ctx, i, w * 0.06, yy, 22))
        frag, n = _claim_text(ctx, c, w * 0.095, yy, 20, 78)
        P.append(frag)
        yy += n * 26 + 18
    P.append(_footer(ctx, h - 26))
    return "".join(P)


def grid(ctx) -> str:
    pack, w, h = ctx["pack"], ctx["w"], ctx["h"]
    d = ctx["draft"]
    P = [_banner(ctx)]
    cols = 3 if len(d.kept_claims()) > 4 else 2
    cw, ch = (w * 0.88) / cols, h * 0.62 / math.ceil(max(1, len(d.kept_claims())) / cols)
    for i, c in enumerate(d.kept_claims()[:9]):
        gx = w * 0.06 + (i % cols) * cw
        gy = h * 0.24 + (i // cols) * ch
        P.append(f'<rect x="{gx}" y="{gy}" width="{cw - 18}" height="{ch - 18}" rx="10" '
                 f'fill="{pack.paper}" stroke="{pack.muted}" stroke-opacity="0.4"/>')
        P.append(_glyph(ctx, i, gx + 16, gy + 34, 20))
        frag, _ = _claim_text(ctx, c, gx + 16, gy + 62, 17, int(cw / 9.2))
        P.append(frag)
    P.append(_footer(ctx, h - 26))
    return "".join(P)


def dashboard(ctx) -> str:
    pack, w, h = ctx["pack"], ctx["w"], ctx["h"]
    d = ctx["draft"]
    P = [_banner(ctx)]
    stats = [c for c in d.kept_claims() if numbers_in(c.text)][:4]
    rest = [c for c in d.kept_claims() if c not in stats][:4]
    tw = w * 0.88 / max(1, len(stats))
    for i, c in enumerate(stats):
        tx = w * 0.06 + i * tw
        P.append(f'<rect x="{tx}" y="{h * 0.22}" width="{tw - 14}" height="{h * 0.2}" '
                 f'rx="8" fill="{pack.paper}" stroke="{pack.muted}" stroke-opacity="0.35"/>')
        nums = numbers_in(c.text)
        P.append(f'<text x="{tx + 18}" y="{h * 0.22 + h * 0.11}" font-size="44" '
                 f'font-family="{pack.font_display}" fill="{pack.accent}" '
                 f'font-weight="bold">{esc(nums[0]) if nums else ""}</text>')
        label = c.text
        for n in nums:
            label = label.replace(n, "")
        frag, _ = _claim_text(ctx, type(c)(text=label.strip(" ()—-")), tx + 18,
                              h * 0.22 + h * 0.16, 15, int(tw / 8), colour=pack.muted)
        P.append(frag)
    yy = h * 0.52
    for i, c in enumerate(rest):
        P.append(_glyph(ctx, i, w * 0.06, yy + 6, 20))
        frag, n = _claim_text(ctx, c, w * 0.095, yy + 6, 19, 80)
        P.append(frag)
        yy += n * 25 + 16
    P.append(ctx["art_svg_small"])
    P.append(_footer(ctx, h - 26))
    return "".join(P)


def funnel(ctx) -> str:
    pack, w, h = ctx["pack"], ctx["w"], ctx["h"]
    d = ctx["draft"]
    P = [_banner(ctx)]
    claims = d.kept_claims()[:6]
    n = len(claims)
    for i, c in enumerate(claims):
        fw = w * (0.85 - 0.11 * i)
        fx = (w - fw) / 2
        fy = h * 0.22 + i * (h * 0.62 / max(1, n))
        P.append(f'<rect x="{fx}" y="{fy}" width="{fw}" height="{h * 0.62 / n - 10}" '
                 f'rx="8" fill="{pack.paper}" stroke="{pack.accent}" '
                 f'stroke-opacity="{0.9 - i * 0.12}"/>')
        frag, _ = _claim_text(ctx, c, fx + 20, fy + 34, 17, int(fw / 10.5))
        P.append(frag)
    P.append(_footer(ctx, h - 26))
    return "".join(P)


def compare(ctx) -> str:
    pack, w, h = ctx["pack"], ctx["w"], ctx["h"]
    d = ctx["draft"]
    P = [_banner(ctx)]
    claims = d.kept_claims()[:8]
    half = math.ceil(len(claims) / 2)
    for col, chunk in enumerate((claims[:half], claims[half:])):
        bx = w * 0.05 + col * w * 0.475
        P.append(f'<rect x="{bx}" y="{h * 0.22}" width="{w * 0.425}" '
                 f'height="{h * 0.62}" rx="12" fill="{pack.paper}" '
                 f'stroke="{pack.muted}" stroke-opacity="0.4"/>')
        yy = h * 0.27
        for i, c in enumerate(chunk):
            P.append(_glyph(ctx, i, bx + 20, yy, 18))
            frag, nn = _claim_text(ctx, c, bx + 52, yy, 17, int(w * 0.425 / 11))
            P.append(frag)
            yy += nn * 23 + 14
    P.append(f'<line x1="{w / 2}" y1="{h * 0.22}" x2="{w / 2}" y2="{h * 0.84}" '
             f'stroke="{pack.accent}" stroke-width="2" stroke-dasharray="8 6"/>')
    P.append(_footer(ctx, h - 26))
    return "".join(P)


def steps(ctx) -> str:
    pack, w, h = ctx["pack"], ctx["w"], ctx["h"]
    d = ctx["draft"]
    P = [_banner(ctx)]
    claims = d.kept_claims()[:6]
    P.append(f'<line x1="{w * 0.11}" y1="{h * 0.24}" x2="{w * 0.11}" '
             f'y2="{h * 0.24 + (len(claims) - 1) * h * 0.105}" '
             f'stroke="{pack.accent}" stroke-width="2"/>')
    yy = h * 0.24
    for i, c in enumerate(claims):
        P.append(f'<circle cx="{w * 0.11}" cy="{yy - 6}" r="20" fill="{pack.paper}" '
                 f'stroke="{pack.accent}" stroke-width="2"/>')
        P.append(f'<text x="{w * 0.11}" y="{yy}" text-anchor="middle" font-size="19" '
                 f'fill="{pack.accent}" font-family="{pack.font_display}" '
                 f'font-weight="bold">{i + 1}</text>')
        frag, n = _claim_text(ctx, c, w * 0.16, yy, 20, 74)
        P.append(frag)
        yy += max(h * 0.105, n * 26 + 24)
    P.append(_footer(ctx, h - 26))
    return "".join(P)


def layers(ctx) -> str:
    pack, w, h = ctx["pack"], ctx["w"], ctx["h"]
    d = ctx["draft"]
    P = [_banner(ctx)]
    claims = d.kept_claims()[:5]
    for i, c in enumerate(claims):
        lx = w * 0.07 + i * w * 0.045
        ly = h * 0.24 + i * h * 0.1
        lw = w * 0.68
        P.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{h * 0.16}" rx="12" '
                 f'fill="{pack.paper}" fill-opacity="{0.9 - i * 0.1}" '
                 f'stroke="{pack.accent}" stroke-opacity="0.5"/>')
        frag, _ = _claim_text(ctx, c, lx + 22, ly + 40, 18, int(lw / 10.5))
        P.append(frag)
    P.append(ctx["art_svg_small"])
    P.append(_footer(ctx, h - 26))
    return "".join(P)


def matrix(ctx) -> str:
    pack, w, h = ctx["pack"], ctx["w"], ctx["h"]
    d = ctx["draft"]
    P = [_banner(ctx)]
    claims = d.kept_claims()[:9]
    cols = 3
    cw, ch = w * 0.86 / cols, h * 0.58 / math.ceil(max(1, len(claims)) / cols)
    for i, c in enumerate(claims):
        mx = w * 0.07 + (i % cols) * cw
        my = h * 0.24 + (i // cols) * ch
        P.append(f'<rect x="{mx}" y="{my}" width="{cw - 12}" height="{ch - 12}" '
                 f'fill="{pack.paper}" stroke="{pack.muted}" stroke-opacity="0.45"/>')
        frag, _ = _claim_text(ctx, c, mx + 14, my + 30, 15, int(cw / 8.6))
        P.append(frag)
    P.append(_footer(ctx, h - 26))
    return "".join(P)


def stream(ctx) -> str:
    pack, w, h = ctx["pack"], ctx["w"], ctx["h"]
    d = ctx["draft"]
    P = [_banner(ctx)]
    claims = d.kept_claims()[:7]
    P.append(f'<line x1="{w * 0.07}" y1="{h * 0.3}" x2="{w * 0.93}" y2="{h * 0.3}" '
             f'stroke="{pack.accent}" stroke-width="2"/>')
    step = w * 0.86 / max(1, len(claims))
    for i, c in enumerate(claims):
        cx = w * 0.07 + i * step
        P.append(f'<circle cx="{cx}" cy="{h * 0.3}" r="9" fill="{pack.accent}"/>')
        yy = h * 0.38 + (i % 2) * h * 0.2
        frag, _ = _claim_text(ctx, c, min(cx, w * 0.75), yy, 16, 26)
        P.append(frag)
    P.append(ctx["art_svg_small"])
    P.append(_footer(ctx, h - 26))
    return "".join(P)


def _banner(ctx) -> str:
    pack, w = ctx["pack"], ctx["w"]
    d = ctx["draft"]
    P = [f'<text x="{w * 0.06}" y="{ctx["h"] * 0.115}" font-size="46" '
         f'font-family="{pack.font_display}" fill="{pack.ink}" '
         f'font-weight="bold">{esc(d.title[:60])}</text>']
    if d.subtitle:
        P.append(f'<text x="{w * 0.06}" y="{ctx["h"] * 0.165}" font-size="19" '
                 f'fill="{pack.muted}" font-family="{pack.font_body}">'
                 f'{esc(d.subtitle[:110])}</text>')
    P.append(f'<line x1="{w * 0.06}" y1="{ctx["h"] * 0.19}" x2="{w * 0.94}" '
             f'y2="{ctx["h"] * 0.19}" stroke="{pack.muted}" stroke-opacity="0.5"/>')
    # corner emblem — every layout carries the chosen art, small
    P.append(f'<g opacity="0.9">{ctx["art_svg_small"]}</g>')
    return "".join(P)


LAYOUTS: Dict[str, object] = {
    "poster": poster, "grid": grid, "dashboard": dashboard, "funnel": funnel,
    "compare": compare, "steps": steps, "layers": layers, "matrix": matrix,
    "stream": stream,
}

LAYOUT_BLURBS = {
    "poster": "one hero claim, statement poster, launch graphic, big number",
    "grid": "even grid of cards, changelog, many equal findings, roundup",
    "dashboard": "kpi tiles and rows, metrics report, benchmark numbers",
    "funnel": "narrowing stages, pipeline, conversion, reduction funnel",
    "compare": "two columns versus, before after, a/b comparison, arms",
    "steps": "ordered numbered steps, how it works, process, method",
    "layers": "stacked overlapping cards, layers of evidence, strata",
    "matrix": "compact matrix cells, table-like sweep, audit grid",
    "stream": "horizontal timeline, stream of events, history, feed",
}


def layout_names() -> List[str]:
    return list(LAYOUTS)
