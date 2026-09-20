"""Style packs — palette + typography + texture. A pack is pure data:
swapping packs re-renders the canvas and calls no model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class StylePack:
    name: str
    label: str
    bg: str
    ink: str
    accent: str
    accent2: str
    muted: str
    paper: str            # card/field colour
    font_display: str     # SVG font-family stack for hero type
    font_body: str
    texture: str          # textures.py generator name or ''
    swatch: tuple         # 4 chips shown in the studio picker


MONO = "'JetBrains Mono','SF Mono',Menlo,Consolas,monospace"
SERIF = "Georgia,'Iowan Old Style','Times New Roman',serif"
SANS = "'Helvetica Neue',Helvetica,Arial,sans-serif"
DISPLAY = "'Arial Black','Helvetica Neue',Helvetica,sans-serif"

STYLE_PACKS: Dict[str, StylePack] = {p.name: p for p in [
    StylePack("blueprint-technical", "Blueprint technical · grid",
              "#081d36", "#cfe6ff", "#4db8ff", "#ffd166", "#5f7fa6", "#0a2547",
              SANS, MONO, "grid", ("#081d36", "#cfe6ff", "#4db8ff", "#0a2547")),
    StylePack("nous-poster", "Nous poster · dots",
              "#0b1220", "#e8e0cf", "#d4a017", "#8ab4f8", "#6b7280", "#141c30",
              SERIF, SERIF, "halftone", ("#0b1220", "#e8e0cf", "#d4a017", "#141c30")),
    StylePack("nous-oxide", "Nous poster · oxide + dots",
              "#101411", "#e5e0d4", "#c96f2e", "#7ba05b", "#6b7265", "#1a211a",
              SERIF, SERIF, "halftone", ("#101411", "#e5e0d4", "#c96f2e", "#1a211a")),
    StylePack("arcade-pixel", "Arcade pixel",
              "#12101f", "#f5f0ff", "#ff5e8a", "#ffd319", "#7a6f9b", "#1d1932",
              DISPLAY, MONO, "pixels", ("#12101f", "#f5f0ff", "#ff5e8a", "#ffd319")),
    StylePack("benchmark-slate", "Benchmark slate",
              "#111418", "#e6edf3", "#58d68d", "#f0b429", "#6e7b8a", "#1b2129",
              SANS, MONO, "grid", ("#111418", "#e6edf3", "#58d68d", "#1b2129")),
    StylePack("bold-graphic", "Bold graphic",
              "#f2efe6", "#16130f", "#e01e37", "#1e4fd8", "#8a8272", "#ffffff",
              DISPLAY, SANS, "", ("#f2efe6", "#16130f", "#e01e37", "#1e4fd8")),
    StylePack("chalkboard", "Chalkboard",
              "#1e2b25", "#e8efe6", "#f4e9c8", "#8fd0c2", "#5f7369", "#26362f",
              "'Comic Sans MS','Chalkboard SE',cursive", MONO, "noise", ("#1e2b25", "#e8efe6", "#f4e9c8", "#26362f")),
    StylePack("editorial-paper", "Editorial paper",
              "#f6f1e7", "#1c1a16", "#8a6d3b", "#31456b", "#9a917f", "#ffffff",
              SERIF, SERIF, "", ("#f6f1e7", "#1c1a16", "#8a6d3b", "#ffffff")),
    StylePack("interface-dark", "Interface dark",
              "#0d1117", "#e6edf3", "#58a6ff", "#f78166", "#7d8590", "#161b22",
              SANS, MONO, "grid", ("#0d1117", "#e6edf3", "#58a6ff", "#161b22")),
    StylePack("morandi-journal", "Morandi journal",
              "#e9e2d6", "#4a4640", "#a87f62", "#7c8b6f", "#8b857a", "#f2ede2",
              SERIF, SERIF, "noise", ("#e9e2d6", "#4a4640", "#a87f62", "#f2ede2")),
    StylePack("swiss-minimal", "Swiss minimal",
              "#ffffff", "#111111", "#e30613", "#111111", "#9a9a9a", "#f5f5f5",
              SANS, SANS, "", ("#ffffff", "#111111", "#e30613", "#f5f5f5")),
    StylePack("terminal-mono", "Terminal mono",
              "#0a0f0a", "#9dff9d", "#39ff6a", "#eaffea", "#3d5c3d", "#0d160d",
              MONO, MONO, "scanlines", ("#0a0f0a", "#9dff9d", "#39ff6a", "#0d160d")),
    StylePack("ui-shot", "UI shot · rule",
              "#101820", "#f0f4f8", "#f6c445", "#4ea1ff", "#5c6b7a", "#182430",
              SANS, MONO, "grid", ("#101820", "#f0f4f8", "#f6c445", "#182430")),
    StylePack("ukiyo", "Ukiyo woodblock",
              "#f4e9d4", "#1d3557", "#c0392b", "#457b9d", "#8a7f66", "#ece0c6",
              SERIF, SERIF, "grain", ("#f4e9d4", "#1d3557", "#c0392b", "#ece0c6")),
]}

DEFAULT_STYLE = "blueprint-technical"


def css_vars(pack: StylePack) -> str:
    return (f"--bg:{pack.bg};--ink:{pack.ink};--accent:{pack.accent};"
            f"--accent2:{pack.accent2};--muted:{pack.muted};--paper:{pack.paper};"
            f"--font-display:{pack.font_display};--font-body:{pack.font_body};")


def style_names() -> List[str]:
    return list(STYLE_PACKS)
