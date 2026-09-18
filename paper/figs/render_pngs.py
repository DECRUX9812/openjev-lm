#!/usr/bin/env python3
"""Rasterise the open-Jev SVG figures and cards to PNG with Playwright Chromium.

Run with the playwright venv:
    /home/decrux/.venvs/playwright/bin/python \
        /home/decrux/Code/openjev-lm/paper/figs/render_pngs.py

Each SVG is inlined into a zero-margin HTML page, laid out at its exact
viewport size, checked in-browser for text overflowing the canvas, then
screenshotted at deviceScaleFactor=2 (so a 1200-wide figure lands as a
2400-pixel-wide PNG).
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

FIGS = Path("/home/decrux/Code/openjev-lm/paper/figs")
CARDS = Path("/home/decrux/Code/openjev-lm/paper/xcards")

JOBS = [
    (FIGS / "fig1_ladder.svg", FIGS / "fig1_ladder.png", 1200, 560),
    (FIGS / "fig2_perclass.svg", FIGS / "fig2_perclass.png", 1200, 560),
    (FIGS / "fig3_pipeline.svg", FIGS / "fig3_pipeline.png", 1200, 520),
    (CARDS / "card1.svg", CARDS / "card1.png", 1200, 675),
    (CARDS / "card2.svg", CARDS / "card2.png", 1200, 675),
]

HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;padding:0;background:#ffffff;}svg{display:block;}</style>
</head><body><!--SVG--></body></html>"""


def boxes(page):
    return page.evaluate(
        """() => [...document.querySelectorAll('svg text')].map(t => {
             const b = t.getBBox();
             return {t: t.textContent, x: b.x, y: b.y, r: b.x + b.width,
                     b: b.y + b.height};
           })"""
    )


def candidates():
    """Chromium binaries to try, in order: whatever Playwright ships with, any
    revision already in the ms-playwright cache, then the system Chrome."""
    import glob
    yield None  # playwright's own default
    for exe in sorted(glob.glob(
            "/home/decrux/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")):
        yield exe
    for exe in ("/usr/bin/google-chrome", "/usr/bin/google-chrome-stable",
                "/usr/bin/chromium", "/usr/bin/chromium-browser"):
        yield exe


def launch(p):
    last = None
    for exe in candidates():
        try:
            if exe is None:
                return p.chromium.launch(), "playwright default chromium"
            if not Path(exe).exists():
                continue
            return p.chromium.launch(executable_path=exe), exe
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"no usable chromium: {last}")


def main() -> int:
    failures = 0
    with sync_playwright() as p:
        browser, exe = launch(p)
        print(f"renderer: {exe}")
        for svg_path, png_path, w, h in JOBS:
            if not svg_path.exists():
                print(f"MISSING {svg_path}")
                failures += 1
                continue
            svg = svg_path.read_text(encoding="utf-8")
            page = browser.new_page(viewport={"width": w, "height": h},
                                    device_scale_factor=2)
            page.set_content(HTML.replace("<!--SVG-->", svg), wait_until="load")
            page.wait_for_timeout(120)

            bad = [b for b in boxes(page)
                   if b["r"] > w + 1 or b["b"] > h + 1 or b["x"] < -1 or b["y"] < -1]
            maxr = max(b["r"] for b in boxes(page))
            maxb = max(b["b"] for b in boxes(page))
            page.screenshot(path=str(png_path),
                            clip={"x": 0, "y": 0, "width": w, "height": h},
                            omit_background=False)
            page.close()
            size = png_path.stat().st_size
            status = "OK" if not bad else f"OVERFLOW {len(bad)}"
            if bad:
                failures += 1
                for b in bad[:6]:
                    print(f"    overflow: {b['t'][:48]!r} right={b['r']:.1f} bottom={b['b']:.1f}")
            print(f"{status:9s} {png_path.name:20s} {w}x{h} @2x -> "
                  f"{w * 2}x{h * 2}px  {size} bytes  "
                  f"(text extent: right<={maxr:.1f} bottom<={maxb:.1f})")
        browser.close()
    return failures


if __name__ == "__main__":
    sys.exit(main())
