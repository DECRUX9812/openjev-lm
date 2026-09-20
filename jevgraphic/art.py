"""Art catalog — seeded, procedural SVG archetypes in the Nous/Teknium
engraving-register, plus an image shelf for real assets with provenance.

Every generator: fn(x, y, w, h, ink, accent, accent2, seed) -> svg <g>.
Nothing is fetched, nothing is invented: procedural art is reproducible,
shelf art carries PROVENANCE.json.
"""
from __future__ import annotations

import base64
import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, List

Pt = tuple


def _pts(points) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def sunburst(x, y, w, h, ink, accent, accent2, seed) -> str:
    cx, cy, r = x + w / 2, y + h * 0.55, min(w, h) * 0.16
    rays, n = [], 36
    for i in range(n):
        a = math.pi * i / n + math.pi           # upper half-fan
        r1, r2 = r * 1.15, r * (2.4 if i % 2 == 0 else 1.9)
        w_half = math.pi / n * 0.42
        p = [(cx + r1 * math.cos(a - w_half), cy + r1 * math.sin(a - w_half)),
             (cx + r2 * math.cos(a), cy + r2 * math.sin(a)),
             (cx + r1 * math.cos(a + w_half), cy + r1 * math.sin(a + w_half))]
        rays.append(f'<polygon points="{_pts(p)}" fill="{accent}" opacity="0.85"/>')
    rays.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{accent2}"/>')
    rays.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
                f'stroke="{ink}" stroke-width="2"/>')
    for rr in (1.5, 2.0, 2.55):
        rays.append(f'<path d="M {cx - r * rr} {cy} A {r * rr} {r * rr} 0 0 1 '
                    f'{cx + r * rr} {cy}" fill="none" stroke="{ink}" '
                    f'stroke-width="1" opacity="0.5"/>')
    return "<g>" + "".join(rays) + "</g>"


def wave(x, y, w, h, ink, accent, accent2, seed) -> str:
    """Ukiyo scallop rows — arcs with claws."""
    rng = random.Random(seed)
    parts, rows = [], 4
    for row in range(rows):
        yy = y + h * (0.15 + 0.24 * row)
        amp = h * 0.10 * (1 - row * 0.12)
        step = w / (7 - row)
        d = f"M {x} {yy + amp}"
        for i in range(int(w / step)):
            x0, x1 = x + i * step, x + (i + 1) * step
            cxm = (x0 + x1) / 2
            d += (f" Q {x0 + step * 0.15} {yy - amp * 1.6} {cxm} {yy - amp * 0.4}"
                  f" Q {x1 - step * 0.15} {yy - amp * 1.1} {x1} {yy + amp}")
        col = accent if row % 2 == 0 else accent2
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" '
                     f'stroke-width="{3.5 - row * 0.6}" opacity="{0.9 - row * 0.15}"/>')
        # foam claws
        for i in range(int(w / step)):
            cx0 = x + i * step + step / 2 + rng.uniform(-4, 4)
            parts.append(f'<circle cx="{cx0:.1f}" cy="{yy - amp * 0.75:.1f}" '
                         f'r="{step * 0.045:.1f}" fill="{ink}" opacity="0.7"/>')
    return "<g>" + "".join(parts) + "</g>"


def caduceus(x, y, w, h, ink, accent, accent2, seed) -> str:
    cx, top, bot = x + w / 2, y + h * 0.08, y + h * 0.95
    parts = [f'<line x1="{cx}" y1="{top}" x2="{cx}" y2="{bot}" '
             f'stroke="{ink}" stroke-width="{w * 0.02}"/>']
    parts.append(f'<circle cx="{cx}" cy="{top - h * 0.02}" r="{w * 0.05}" '
                 f'fill="{accent}"/>')
    # wings
    for side in (-1, 1):
        parts.append(
            f'<path d="M {cx} {top + h * 0.06} Q {cx + side * w * 0.42} '
            f'{top - h * 0.10} {cx + side * w * 0.46} {top + h * 0.05} '
            f'Q {cx + side * w * 0.24} {top + h * 0.10} {cx} {top + h * 0.12} Z" '
            f'fill="{accent2}" opacity="0.9"/>')
    # twin snakes as phase-shifted sines
    for phase, col in ((0.0, accent), (math.pi, ink)):
        d = f"M {cx} {bot}"
        steps = 60
        for i in range(steps + 1):
            t = i / steps
            yy = bot - t * (bot - top - h * 0.12)
            xx = cx + math.sin(t * math.pi * 4 + phase) * w * 0.16 * (1 - t * 0.55)
            d += f" L {xx:.1f} {yy:.1f}"
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" '
                     f'stroke-width="{w * 0.016}"/>')
        # head
        hx = cx + math.sin(math.pi * 4 + phase) * w * 0.16 * 0.45
        parts.append(f'<circle cx="{hx:.1f}" cy="{top + h * 0.10}" '
                     f'r="{w * 0.035}" fill="{col}"/>')
    return "<g>" + "".join(parts) + "</g>"


def bust(x, y, w, h, ink, accent, accent2, seed) -> str:
    """Stylised classical profile (helmeted head) in engraving line."""
    s = min(w, h)
    cx, base = x + w * 0.52, y + h * 0.92
    prof = [(-0.30, 0.00), (-0.34, -0.10), (-0.30, -0.22), (-0.36, -0.34),
            (-0.30, -0.46), (-0.16, -0.54), (-0.02, -0.56), (0.08, -0.52),
            (0.14, -0.44), (0.20, -0.36), (0.17, -0.30), (0.24, -0.26),  # nose
            (0.15, -0.22), (0.19, -0.17), (0.13, -0.14), (0.17, -0.09),  # lips/chin
            (0.10, -0.05), (0.16, 0.00)]
    pts = [(cx + px * s, base + py * s) for px, py in prof]
    d = f"M {_pts(pts).split()[0]} " + " ".join(
        f"L {px:.1f} {py:.1f}" for px, py in pts[1:])
    parts = [f'<path d="{d}" fill="none" stroke="{ink}" stroke-width="{s * 0.012}"/>']
    # helmet crest
    crest = [(cx - 0.30 * s + i * s * 0.02, base - 0.52 * s - math.sin(i / 22 * math.pi) * s * 0.13)
             for i in range(23)]
    parts.append(f'<polyline points="{_pts(crest)}" fill="none" stroke="{accent}" '
                 f'stroke-width="{s * 0.014}"/>')
    # engraving hatches inside the silhouette
    rng = random.Random(seed)
    for i in range(14):
        yy = base - s * (0.06 + i * 0.034)
        x0 = cx - s * (0.30 - i * 0.012)
        parts.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0 + s * 0.16:.1f}" '
                     f'y2="{yy - s * 0.02:.1f}" stroke="{ink}" stroke-width="0.7" '
                     f'opacity="0.55"/>')
    # neck drape
    parts.append(f'<path d="M {cx - 0.30 * s:.1f} {base:.1f} '
                 f'Q {cx - 0.05 * s:.1f} {base + s * 0.06:.1f} '
                 f'{cx + 0.30 * s:.1f} {base:.1f}" stroke="{ink}" fill="none" '
                 f'stroke-width="{s * 0.01}"/>')
    return "<g>" + "".join(parts) + "</g>"


def mountain(x, y, w, h, ink, accent, accent2, seed) -> str:
    cx, base = x + w / 2, y + h * 0.9
    peak_y = y + h * 0.12
    parts = [f'<path d="M {x} {base} Q {cx - w * 0.12} {peak_y + h * 0.16} '
             f'{cx} {peak_y} Q {cx + w * 0.12} {peak_y + h * 0.16} '
             f'{x + w} {base} Z" fill="{accent2}" opacity="0.85"/>']
    # snow cap
    cap = f"M {cx - w * 0.09} {peak_y + h * 0.13} "
    for i in range(5):
        cap += f"L {cx - w * 0.09 + w * 0.036 * (i + 1)} {peak_y + h * (0.13 + 0.05 * (i % 2))} "
    cap += f"L {cx + w * 0.09} {peak_y + h * 0.13} Q {cx + w * 0.045} {peak_y + h * 0.07} {cx} {peak_y} Q {cx - w * 0.045} {peak_y + h * 0.07} {cx - w * 0.09} {peak_y + h * 0.13} Z"
    parts.append(f'<path d="{cap}" fill="{ink}" opacity="0.95"/>')
    # sun
    parts.append(f'<circle cx="{x + w * 0.82}" cy="{y + h * 0.2}" r="{h * 0.09}" '
                 f'fill="{accent}"/>')
    return "<g>" + "".join(parts) + "</g>"


def pixelscape(x, y, w, h, ink, accent, accent2, seed) -> str:
    rng = random.Random(seed)
    s = max(6, int(w / 40))
    parts = []
    # sky pixels
    for _ in range(int(w * h / (s * s * 30))):
        parts.append(f'<rect x="{x + rng.randrange(int(w / s)) * s}" '
                     f'y="{y + rng.randrange(int(h * 0.5 / s)) * s}" '
                     f'width="{s}" height="{s}" fill="{ink}" opacity="0.5"/>')
    # moon
    parts.append(f'<rect x="{x + w * 0.7:.0f}" y="{y + h * 0.12:.0f}" width="{s * 3}" '
                 f'height="{s * 3}" fill="{accent}"/>')
    # terrain ridge — seeded random walk, two layers
    for layer, col, hh in ((0, accent2, 0.62), (1, ink, 0.74)):
        r2 = random.Random(seed + layer)
        yy = y + h * hh
        xx = x
        while xx < x + w:
            parts.append(f'<rect x="{xx:.0f}" y="{yy:.0f}" width="{s}" '
                         f'height="{y + h - yy:.0f}" fill="{col}" '
                         f'opacity="{0.85 if layer else 0.6}"/>')
            yy += r2.choice((-1, -1, 0, 0, 1, 1)) * s
            yy = max(y + h * 0.45, min(y + h * 0.92, yy))
            xx += s
    return "<g>" + "".join(parts) + "</g>"


def stipple_burst(x, y, w, h, ink, accent, accent2, seed) -> str:
    rng = random.Random(seed)
    cx, cy, rmax = x + w / 2, y + h / 2, min(w, h) / 2
    parts = [f'<circle cx="{cx}" cy="{cy}" r="{rmax * 0.16}" fill="{accent}"/>']
    for _ in range(1400):
        a, t = rng.random() * math.tau, rng.random() ** 0.4
        r = rmax * (0.2 + 0.8 * t)
        if rng.random() < t:            # density falls off outward
            continue
        parts.append(f'<circle cx="{cx + r * math.cos(a):.1f}" '
                     f'cy="{cy + r * math.sin(a):.1f}" r="{rng.random() * 1.8:.2f}" '
                     f'fill="{ink}" opacity="{0.8 - t * 0.6:.2f}"/>')
    return "<g>" + "".join(parts) + "</g>"


def schematic(x, y, w, h, ink, accent, accent2, seed) -> str:
    rng = random.Random(seed)
    parts = []
    for _ in range(26):
        x0, y0 = x + rng.random() * w, y + rng.random() * h
        x1 = x0 + rng.choice((-1, 1)) * rng.random() * w * 0.3
        y1 = y0 + rng.choice((-1, 1)) * rng.random() * h * 0.3
        midx = x1
        parts.append(f'<path d="M {x0:.0f} {y0:.0f} L {midx:.0f} {y0:.0f} '
                     f'L {midx:.0f} {y1:.0f}" fill="none" stroke="{ink}" '
                     f'stroke-width="1.2" opacity="0.7"/>')
        parts.append(f'<circle cx="{x0:.0f}" cy="{y0:.0f}" r="3" fill="{accent}"/>')
        parts.append(f'<rect x="{midx - 4:.0f}" y="{y1 - 4:.0f}" width="8" height="8" '
                     f'fill="none" stroke="{accent2}" stroke-width="1.5"/>')
    return "<g>" + "".join(parts) + "</g>"


def seal(x, y, w, h, ink, accent, accent2, seed) -> str:
    cx, cy, r = x + w / 2, y + h / 2, min(w, h) / 2
    parts = []
    for i, rr in enumerate((1.0, 0.92, 0.6, 0.55)):
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r * rr:.1f}" fill="none" '
                     f'stroke="{accent if i % 2 == 0 else ink}" '
                     f'stroke-width="{2.5 if i < 2 else 1.2}"/>')
    n = 48
    for i in range(n):
        a = math.tau * i / n
        r0, r1 = r * 0.92, r * (0.80 if i % 4 == 0 else 0.86)
        parts.append(f'<line x1="{cx + r0 * math.cos(a):.1f}" '
                     f'y1="{cy + r0 * math.sin(a):.1f}" '
                     f'x2="{cx + r1 * math.cos(a):.1f}" '
                     f'y2="{cy + r1 * math.sin(a):.1f}" stroke="{ink}" '
                     f'stroke-width="1.4"/>')
    parts.append(f'<text x="{cx}" y="{cy + r * 0.14:.1f}" text-anchor="middle" '
                 f'font-size="{r * 0.5:.0f}" fill="{accent}" '
                 f'font-family="Georgia,serif">ϟ</text>')
    return "<g>" + "".join(parts) + "</g>"


def orbit(x, y, w, h, ink, accent, accent2, seed) -> str:
    cx, cy = x + w / 2, y + h / 2
    parts = []
    for i, (rx, ry, rot) in enumerate(((0.48, 0.16, -18), (0.44, 0.24, 22), (0.36, 0.36, 0))):
        parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{w * rx:.1f}" '
                     f'ry="{h * ry:.1f}" transform="rotate({rot} {cx} {cy})" '
                     f'fill="none" stroke="{ink}" stroke-width="1.2" opacity="0.7"/>')
        a = math.radians(rot)
        ex, ey = cx + w * rx * math.cos(a), cy + h * ry
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{min(w, h) * 0.03:.1f}" '
                     f'fill="{accent if i % 2 == 0 else accent2}"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{min(w, h) * 0.07:.1f}" '
                 f'fill="{accent}"/>')
    return "<g>" + "".join(parts) + "</g>"


def torii(x, y, w, h, ink, accent, accent2, seed) -> str:
    parts = []
    top, legw = y + h * 0.1, w * 0.09
    # curved lintel
    parts.append(f'<path d="M {x} {top + h * 0.08} Q {x + w / 2} {top - h * 0.08} '
                 f'{x + w} {top + h * 0.08} L {x + w * 0.94} {top + h * 0.16} '
                 f'Q {x + w / 2} {top + h * 0.02} {x + w * 0.06} {top + h * 0.16} Z" '
                 f'fill="{accent}"/>')
    parts.append(f'<rect x="{x + w * 0.14}" y="{top + h * 0.3}" width="{w * 0.72}" '
                 f'height="{h * 0.05}" fill="{accent}"/>')
    for sx in (0.2, 0.8 - 0.09):
        parts.append(f'<rect x="{x + w * sx}" y="{top + h * 0.12}" width="{legw}" '
                     f'height="{h * 0.8}" fill="{ink}"/>')
    parts.append(f'<rect x="{x + w * 0.5 - legw * 0.4}" y="{top + h * 0.16}" '
                 f'width="{legw * 0.8}" height="{h * 0.14}" fill="{ink}"/>')
    return "<g>" + "".join(parts) + "</g>"


def glyphs(x, y, w, h, ink, accent, accent2, seed) -> str:
    rng = random.Random(seed)
    chars = "◈◇◆△▲▽▼○◎●□■✦✧ψΩλδπ∴"
    parts = []
    for _ in range(46):
        parts.append(f'<text x="{x + rng.random() * w:.0f}" '
                     f'y="{y + rng.random() * h:.0f}" font-size="{rng.randrange(10, 30)}" '
                     f'fill="{rng.choice((ink, accent, accent2))}" '
                     f'opacity="{0.25 + rng.random() * 0.5:.2f}">'
                     f'{rng.choice(chars)}</text>')
    return "<g>" + "".join(parts) + "</g>"


ART: Dict[str, Callable] = {
    "sunburst": sunburst, "wave": wave, "caduceus": caduceus, "bust": bust,
    "mountain": mountain, "pixelscape": pixelscape, "stipple-burst": stipple_burst,
    "schematic": schematic, "seal": seal, "orbit": orbit, "torii": torii,
    "glyphs": glyphs,
}

ART_BLURBS = {
    "sunburst": "radiating sun, optimism, launch, revelation, dawn",
    "wave": "hokusai waves, flow, stream of data, persistence, ukiyo-e",
    "caduceus": "hermes staff snakes wings, messenger, medicine, classical",
    "bust": "classical statue head profile, engraving, wisdom, nous",
    "mountain": "fuji mountain snow, ascent, summit, achievement",
    "pixelscape": "pixel art landscape moon, retro game, arcade, terrain",
    "stipple-burst": "stipple starburst explosion, big bang, energy",
    "schematic": "circuit schematic traces, electronics, blueprint, wiring",
    "seal": "stamp seal rings, authenticity, verified, approved, sigil",
    "orbit": "orbits planets satellites, systems, gravity, cycles",
    "torii": "torii gate silhouette, threshold, japan, shrine",
    "glyphs": "floating glyph symbols, esoteric, cipher, arcane marks",
}


def render_art(name: str, x, y, w, h, ink, accent, accent2, seed=0) -> str:
    fn = ART.get(name, sunburst)
    return fn(x, y, w, h, ink, accent, accent2, seed)


def art_names() -> List[str]:
    return list(ART)


# --- image shelf -----------------------------------------------------------

def shelf(shelf_dir: str) -> List[dict]:
    """Real assets with provenance. Returns [{'path', 'sha', 'tags'}]."""
    d = Path(shelf_dir)
    if not d.is_dir():
        return []
    import hashlib
    prov = {}
    pj = d / "PROVENANCE.json"
    if pj.exists():
        prov = json.loads(pj.read_text())
    out = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".svg"):
            out.append({"path": str(p), "sha": hashlib.sha256(p.read_bytes()).hexdigest()[:12],
                        "tags": prov.get(p.name, {}).get("tags", "")})
    return out


def embed_image(item: dict, x, y, w, h, embed=True) -> str:
    """SVG <image> for a shelf asset. Embeds base64 so the SVG stays portable."""
    p = Path(item["path"])
    if embed and p.suffix.lower() != ".svg":
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"
                }.get(p.suffix.lower().lstrip("."), "image/png")
        href = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    else:
        href = p.resolve().as_uri()
    return (f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
            f'preserveAspectRatio="xMidYMid slice" href="{href}"/>')
