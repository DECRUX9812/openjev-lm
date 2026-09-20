"""Procedural SVG texture overlays. Each returns an SVG fragment string
sized to (w, h), drawn in `colour` at low opacity — the paper grain of a
style pack. Deterministic given a seed."""
from __future__ import annotations

import random
from typing import Callable, Dict


def grid(w: int, h: int, colour: str, seed: int = 0) -> str:
    step = 40
    parts = [f'<g stroke="{colour}" stroke-width="0.5" opacity="0.18">']
    for x in range(0, w + 1, step):
        parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{h}"/>')
    for y in range(0, h + 1, step):
        parts.append(f'<line x1="0" y1="{y}" x2="{w}" y2="{y}"/>')
    parts.append("</g>")
    # drafting crosshair marks at quarter points
    parts.append(f'<g stroke="{colour}" stroke-width="1" opacity="0.5">')
    for cx, cy in [(24, 24), (w - 24, 24), (24, h - 24), (w - 24, h - 24)]:
        parts.append(f'<line x1="{cx - 8}" y1="{cy}" x2="{cx + 8}" y2="{cy}"/>'
                     f'<line x1="{cx}" y1="{cy - 8}" x2="{cx}" y2="{cy + 8}"/>')
    parts.append("</g>")
    return "".join(parts)


def halftone(w: int, h: int, colour: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    parts = [f'<g fill="{colour}" opacity="0.14">']
    step = 14
    for y in range(0, h, step):
        for x in range(0, w, step):
            r = 1.4 * (y / h) + rng.random() * 0.3
            parts.append(f'<circle cx="{x}" cy="{y}" r="{r:.2f}"/>')
    parts.append("</g>")
    return "".join(parts)


def scanlines(w: int, h: int, colour: str, seed: int = 0) -> str:
    parts = [f'<g stroke="{colour}" stroke-width="1" opacity="0.10">']
    for y in range(0, h, 4):
        parts.append(f'<line x1="0" y1="{y}" x2="{w}" y2="{y}"/>')
    parts.append("</g>")
    return "".join(parts)


def noise(w: int, h: int, colour: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    parts = [f'<g fill="{colour}" opacity="0.08">']
    for _ in range(int(w * h / 900)):
        parts.append(f'<circle cx="{rng.randrange(w)}" cy="{rng.randrange(h)}" '
                     f'r="{rng.random() * 1.3:.2f}"/>')
    parts.append("</g>")
    return "".join(parts)


def pixels(w: int, h: int, colour: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    parts = [f'<g fill="{colour}" opacity="0.055">']
    s = 24
    for _ in range(int(w * h / (s * s * 22))):
        parts.append(f'<rect x="{rng.randrange(w // s) * s}" '
                     f'y="{rng.randrange(h // s) * s}" width="{s}" height="{s}"/>')
    parts.append("</g>")
    return "".join(parts)


def grain(w: int, h: int, colour: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    parts = [f'<g stroke="{colour}" opacity="0.07">']
    for _ in range(260):
        x, y = rng.randrange(w), rng.randrange(h)
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + rng.randrange(4, 14)}" '
                     f'y2="{y}" stroke-width="{rng.random():.2f}"/>')
    parts.append("</g>")
    return "".join(parts)


TEXTURES: Dict[str, Callable[[int, int, str, int], str]] = {
    "grid": grid, "halftone": halftone, "scanlines": scanlines,
    "noise": noise, "pixels": pixels, "grain": grain, "": lambda *a: "",
}
