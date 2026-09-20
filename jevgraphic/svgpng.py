"""SVG -> PNG export. Tries headless Chrome/Chromium, then rsvg-convert,
then qlmanage (macOS). Returns the exporter used or '' on failure."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome"), shutil.which("chromium"),
    shutil.which("chrome"), shutil.which("chromium-browser"),
]


def export_png(svg_path: str, png_path: str, width: int = 1440,
               height: int = 1080, scale: int = 2) -> str:
    svg, png = Path(svg_path).resolve(), Path(png_path).resolve()
    for chrome in CHROME_CANDIDATES:
        if not chrome:
            continue
        try:
            subprocess.run(
                [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                 f"--screenshot={png}", f"--window-size={width},{height}",
                 f"--force-device-scale-factor={scale}",
                 "--default-background-color=00000000", svg.as_uri()],
                check=True, capture_output=True, timeout=60)
            if png.exists():
                return "chrome"
        except Exception:
            continue
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        subprocess.run([rsvg, "-w", str(width * scale), "-h", str(height * scale),
                        "-o", str(png), str(svg)], check=True, timeout=30)
        return "rsvg-convert" if png.exists() else ""
    ql = shutil.which("qlmanage")
    if ql:  # macOS fallback
        subprocess.run([ql, "-t", "-s", str(width * scale), "-o",
                        str(png.parent), str(svg)],
                       check=True, capture_output=True, timeout=30)
        produced = png.parent / (svg.name + ".png")
        if produced.exists():
            produced.rename(png)
            return "qlmanage"
    return ""
