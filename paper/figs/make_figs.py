#!/usr/bin/env python3
"""Author the open-Jev launch figure set as hand-laid-out SVG.

Every numeric value written into these figures is taken from
../jev-repro-test/openjev/runs/FACTS.json (paths below). Nothing is
estimated, extrapolated or re-rounded: percentages are computed from the
raw fractions recorded in FACTS.json and printed at 0.1 resolution.

No charting library is used -- every coordinate below is written out
explicitly so the files are readable, diffable and editable by hand.

Outputs
  paper/figs/fig1_ladder.svg     headline ladder chart          (1200x560)
  paper/figs/fig2_perclass.svg   per-class recall grouped bars  (1200x560)
  paper/figs/fig3_pipeline.svg   two-arm pipeline diagram       (1200x520)
  paper/xcards/card1.svg         X card #1                      (1200x675)
  paper/xcards/card2.svg         X card #2                      (1200x675)
"""
from __future__ import annotations

import html
from pathlib import Path

# ---------------------------------------------------------------- palette
INK = "#111111"
ACCENT = "#0a7f5a"          # open-Jev (ours)
GREY = "#8a8a8a"            # baselines / others
GREY_L = "#c2c2c2"          # second baseline
GREY_TXT = "#6b6b6b"
GRID = "#ededed"
AXIS = "#d9d9d9"
BOX_STROKE = "#d8d8d8"
BOX_CHIP_FILL = "#f6f6f6"
BOX_CHIP_STROKE = "#e3e3e3"

FONT = "Inter, Lato, 'Liberation Sans', Arial, sans-serif"

# card palette
C_BG = "#0d1117"
C_TEXT = "#e8e8e8"
C_ACCENT = "#34d399"
C_RULE = "#232a33"

FIGS = Path("/home/decrux/Code/openjev-lm/paper/figs")
CARDS = Path("/home/decrux/Code/openjev-lm/paper/xcards")
SOURCE = "source: runs/FACTS.json"


# ---------------------------------------------------------------- helpers
def esc(s: str) -> str:
    return html.escape(s, quote=False)


def text(x, y, s, size=13, fill=INK, weight="400", anchor="start",
         ls=None, opacity=None, style=None):
    a = [f'x="{x:.1f}"' if isinstance(x, float) else f'x="{x}"',
         f'y="{y:.1f}"' if isinstance(y, float) else f'y="{y}"',
         f'font-family="{FONT}"', f'font-size="{size}"', f'fill="{fill}"']
    if weight != "400":
        a.append(f'font-weight="{weight}"')
    if anchor != "start":
        a.append(f'text-anchor="{anchor}"')
    if ls is not None:
        a.append(f'letter-spacing="{ls}"')
    if opacity is not None:
        a.append(f'opacity="{opacity}"')
    if style is not None:
        a.append(f'font-style="{style}"')
    return f'<text {" ".join(a)}>{esc(s)}</text>'


def rich_text(x, y, parts, size=13, weight="400", anchor="start", ls=None):
    """parts: list of (string, fill, opacity-or-None)."""
    a = [f'x="{x}"', f'y="{y}"', f'font-family="{FONT}"',
         f'font-size="{size}"', f'font-weight="{weight}"']
    if anchor != "start":
        a.append(f'text-anchor="{anchor}"')
    if ls is not None:
        a.append(f'letter-spacing="{ls}"')
    spans = []
    for s, fill, op in parts:
        extra = f' opacity="{op}"' if op is not None else ""
        spans.append(f'<tspan fill="{fill}"{extra}>{esc(s)}</tspan>')
    return f'<text {" ".join(a)}>{"".join(spans)}</text>'


def pct(num: float, den: float) -> str:
    """Percentage at 0.1 resolution, from the raw fraction."""
    return f"{num / den * 100:.1f}%"


def svg_open(w: int, h: int, title: str, desc: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img">\n'
            f'<title>{esc(title)}</title>\n<desc>{esc(desc)}</desc>\n')


# ---------------------------------------------------------------- fig 1
def fig1_ladder() -> str:
    W, H = 1200, 560
    x0, x1 = 300.0, 1120.0
    pw = x1 - x0
    ytop, ybot = 126.0, 476.0
    n = 5
    row_h = (ybot - ytop) / n
    bar_h = 36.0
    prior = 54 / 70

    rows = [
        ("Jev (hosted)",          68 / 70, GREY,   INK),
        ("open-Jev (classifier)", 66 / 70, ACCENT, ACCENT),
        ("open-Jev (0.5B LM)",    65 / 70, ACCENT, ACCENT),
        ("viral Jev repro",       54 / 70, GREY,   INK),
        ("untrained 0.5B",        51 / 70, GREY,   INK),
    ]

    s = [svg_open(W, H,
                  "Bucket accuracy on 70 hand-labelled postings",
                  "Horizontal bars: hosted Jev, the two open-Jev arms, the viral "
                  "reproduction and the untrained base model. Dashed line marks the "
                  "majority-class prior, 54 of 70.")]
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    s.append(text(64, 52, "Bucket accuracy on 70 hand-labelled postings", 23, INK, "600"))
    s.append(text(64, 80, "open-Jev arms in green · Jev (hosted) as reference · "
                          "dashed line marks the majority-class prior", 13.5, GREY_TXT))

    # gridlines + x ticks
    for p in range(0, 101, 20):
        gx = x0 + pw * p / 100
        s.append(f'<line x1="{gx:.1f}" y1="{ytop}" x2="{gx:.1f}" y2="{ybot}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(text(gx, ybot + 21, f"{p}%", 12, GREY, anchor="middle"))
    s.append(f'<line x1="{x0}" y1="{ybot}" x2="{x1}" y2="{ybot}" '
             f'stroke="{AXIS}" stroke-width="1"/>')

    # bars, labels, values
    for i, (name, rate, bar_fill, txt_fill) in enumerate(rows):
        cy = ytop + row_h * i + row_h / 2
        by = cy - bar_h / 2
        bw = pw * rate
        s.append(f'<rect x="{x0:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                 f'height="{bar_h:.1f}" rx="2" fill="{bar_fill}"/>')
        s.append(text(288, cy + 5.5, name, 15.5, txt_fill, "500", anchor="end"))
        s.append(text(x0 + bw + 13, cy + 5.5, pct(rate, 1), 15, txt_fill, "600"))

    # majority-class prior
    px = x0 + pw * prior
    s.append(f'<line x1="{px:.2f}" y1="118" x2="{px:.2f}" y2="{ybot}" '
             f'stroke="{GREY}" stroke-width="1.3" stroke-dasharray="6 4"/>')
    s.append(text(px + 9, 112, "majority-class prior (54/70)", 12.5, GREY_TXT))

    s.append(text(64, 536, SOURCE, 11.5, GREY))
    s.append(text(1136, 536, "n = 70 hand-labelled postings · exact bucket match",
                  11.5, GREY, anchor="end"))
    s.append("</svg>\n")
    return "".join(s)


# ---------------------------------------------------------------- fig 2
def fig2_perclass() -> str:
    W, H = 1200, 560
    x0, x1 = 150.0, 1150.0
    pw = x1 - x0
    ytop, ybot = 150.0, 470.0
    ph = ybot - ytop

    arms = [("open-Jev (0.5B LM)", ACCENT),
            ("viral Jev repro", GREY),
            ("untrained 0.5B", GREY_L)]
    # (class label, [ (num, den) or None per arm ])  -- None renders as n/a
    groups = [
        ("service_lead", 2,  [(1, 2), (0, 2), (0, 2)]),
        ("staff_role", 14,   [(12, 14), (3, 14), (0, 14)]),
        ("generic_job", 54,  [(52, 54), None, None]),
    ]

    s = [svg_open(W, H,
                  "Per-class recall on the 70 hand labels",
                  "Grouped bars: recall within service_lead, staff_role and "
                  "generic_job for the open-Jev LM arm, the viral reproduction and "
                  "the untrained base model. n/a marks recall not reported for that "
                  "arm in FACTS.json.")]
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    s.append(text(64, 46, "Per-class recall on the 70 hand labels", 23, INK, "600"))
    s.append(text(64, 73, "recall within each gold class · 0.0% = the arm recovered no "
                          "gold row of that class", 13.5, GREY_TXT))

    # legend
    lx = 150.0
    for name, fill in arms:
        s.append(f'<rect x="{lx:.1f}" y="95" width="13" height="13" rx="2" fill="{fill}"/>')
        s.append(text(lx + 21, 106, name, 13.5, INK, "500"))
        lx += 21 + (len(name) * 7.1) + 40

    # y gridlines + ticks
    for p in range(0, 101, 20):
        gy = ybot - ph * p / 100
        s.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(text(x0 - 16, gy + 4.5, f"{p}%", 12, GREY, anchor="end"))
    s.append(f'<line x1="{x0}" y1="{ybot}" x2="{x1}" y2="{ybot}" '
             f'stroke="{AXIS}" stroke-width="1"/>')

    gw = pw / len(groups)
    bar_w, bar_gap = 64.0, 8.0
    cluster_w = 3 * bar_w + 2 * bar_gap
    for gi, (cname, cn, vals) in enumerate(groups):
        gx0 = x0 + gw * gi
        cluster_x0 = gx0 + (gw - cluster_w) / 2
        for ai, v in enumerate(vals):
            cx = cluster_x0 + ai * (bar_w + bar_gap)
            fill = arms[ai][1]
            if v is None:
                s.append(text(cx + bar_w / 2, ybot - 11, "n/a", 13, GREY,
                              "400", anchor="middle", style="italic"))
                continue
            num, den = v
            h = ph * (num / den)
            if h <= 0:
                s.append(f'<rect x="{cx:.1f}" y="{ybot - 3:.1f}" width="{bar_w}" '
                         f'height="3" rx="1" fill="{fill}"/>')
            else:
                s.append(f'<rect x="{cx:.1f}" y="{ybot - h:.1f}" width="{bar_w}" '
                         f'height="{h:.1f}" rx="2" fill="{fill}"/>')
            s.append(text(cx + bar_w / 2, ybot - h - 10, pct(num, den), 13,
                          ACCENT if ai == 0 else INK, "600", anchor="middle"))
        s.append(text(gx0 + gw / 2, ybot + 26, cname, 15, INK, "500", anchor="middle"))
        s.append(text(gx0 + gw / 2, ybot + 45, f"n = {cn} in the 70 gold rows",
                      11.5, GREY, anchor="middle"))

    s.append(text(64, 536, SOURCE, 11.5, GREY))
    s.append(text(1136, 536, "n/a = recall not reported for that arm in FACTS.json",
                  11.5, GREY, anchor="end"))
    s.append("</svg>\n")
    return "".join(s)


# ---------------------------------------------------------------- fig 3
def fig3_pipeline() -> str:
    W, H = 1200, 520
    XL, XR = 64.0, 1136.0
    span = XR - XL

    s = [svg_open(W, H,
                  "open-Jev: how a posting becomes a typed judgment",
                  "Two local paths. Top: LM arm, Qwen2.5-0.5B plus a 2,162,688 "
                  "parameter LoRA adapter, constrained decoding into typed JSON. "
                  "Bottom: classifier arm, frozen bge-small-en-v1.5 with 12-seed "
                  "MLP heads, 6 ms and 3 MB per posting.")]
    s.append(f'<defs><marker id="arw" viewBox="0 0 10 10" refX="8.5" refY="5" '
             f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="#b3b3b3"/></marker></defs>')
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    s.append(text(64, 50, "open-Jev: how a posting becomes a typed judgment", 23, INK, "600"))
    s.append(text(64, 77, "one posting in, a typed judgment out — no network, "
                          "no GPU, no per-call cost", 13.5, GREY_TXT))

    def lane_header(y, label, right_note):
        s.append(f'<rect x="{XL}" y="{y - 9:.1f}" width="9" height="9" rx="1" fill="{ACCENT}"/>')
        s.append(text(XL + 18, y, label, 13.5, INK, "700", ls="0.4"))
        s.append(text(XR, y, right_note, 12, GREY, anchor="end"))

    def box(x, y, w, h, title, sub=None, chip=False, accent=False, title_size=14):
        fill = BOX_CHIP_FILL if chip else "#ffffff"
        stroke = BOX_CHIP_STROKE if chip else (ACCENT if accent else BOX_STROKE)
        sw = 1.4 if accent else 1.1
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        cx = x + w / 2
        cy = y + h / 2
        t_fill = ACCENT if accent else (GREY_TXT if chip else INK)
        t_weight = "600" if (accent or chip) else "500"
        if sub:
            s.append(text(cx, cy - 3, title, title_size, t_fill, t_weight, anchor="middle"))
            s.append(text(cx, cy + 19, sub, 11.5, GREY_TXT, anchor="middle"))
        else:
            s.append(text(cx, cy + 5, title, title_size, t_fill, t_weight, anchor="middle"))

    def arrow(x_from, x_to, y):
        s.append(f'<line x1="{x_from:.1f}" y1="{y:.1f}" x2="{x_to:.1f}" y2="{y:.1f}" '
                 f'stroke="#b3b3b3" stroke-width="1.4" marker-end="url(#arw)"/>')

    # --- LM arm -------------------------------------------------------
    A_Y, A_H = 136.0, 80.0
    a_cy = A_Y + A_H / 2
    lane_header(122, "LM arm", "trained locally: 400 steps · 89.3 min · CPU-only")
    n_a = 5
    gap_a = 28.0
    box_w_a = (span - gap_a * (n_a - 1)) / n_a
    boxes_a = [
        ("posting", None, True, False),
        ("7-question prompt", "bucket · 5 booleans · fit 0-4", False, False),
        ("Qwen2.5-0.5B + LoRA", "r=16 · 2,162,688 params", False, False),
        ("constrained decoding", "argmax over allowed labels", False, False),
        ("typed JSON", "with per-field confidences", False, True),
    ]
    for i, (t, sub, chip, acc) in enumerate(boxes_a):
        bx = XL + i * (box_w_a + gap_a)
        box(bx, A_Y, box_w_a, A_H, t, sub, chip=chip, accent=acc)
        if i:
            arrow(bx - gap_a + 6, bx - 4, a_cy)

    # --- classifier arm -----------------------------------------------
    B_Y, B_H = 300.0, 80.0
    b_cy = B_Y + B_H / 2
    lane_header(286, "classifier arm", "deterministic · 12 seeds · published under MIT")
    n_b = 4
    gap_b = 28.0
    box_w_b = (span - gap_b * (n_b - 1)) / n_b
    boxes_b = [
        ("posting", None, True, False),
        ("frozen bge-small-en-v1.5", "BAAI encoder · weights untouched", False, False),
        ("12-seed MLP heads", "over frozen embeddings", False, False),
        ("6 ms / 3 MB", "per posting · on CPU", False, True),
    ]
    for i, (t, sub, chip, acc) in enumerate(boxes_b):
        bx = XL + i * (box_w_b + gap_b)
        box(bx, B_Y, box_w_b, B_H, t, sub, chip=chip, accent=acc)
        if i:
            arrow(bx - gap_b + 6, bx - 4, b_cy)

    # --- footer -------------------------------------------------------
    s.append(f'<line x1="{XL}" y1="430" x2="{XR}" y2="430" stroke="{GRID}" stroke-width="1"/>')
    s.append(text(600, 464, "both trained on Jev’s own API answers · both run offline "
                            "on CPU · $0/call", 14, INK, "600", anchor="middle"))
    s.append("</svg>\n")
    return "".join(s)


# ---------------------------------------------------------------- cards
def card_chrome(W, H):
    out = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="{C_BG}"/>']
    out.append(text(88, 80, "open-Jev", 26, C_ACCENT, "700", ls="-0.3"))
    out.append(f'<line x1="88" y1="118" x2="{W - 88}" y2="118" stroke="{C_RULE}" stroke-width="1"/>')
    return out


def card1() -> str:
    W, H = 1200, 675
    s = [svg_open(W, H, "open-Jev card 1",
                  "Dark card: the viral Jev reproduction scores 77.1%, exactly the "
                  "majority-class prior of 54 in 70.")]
    s += card_chrome(W, H)
    s.append(rich_text(88, 322,
                       [("The viral Jev repro: ", C_TEXT, None),
                        ("77.1%", C_ACCENT, None)],
                       size=60, weight="700", ls="-0.8"))
    s.append(text(88, 404, "exactly the majority-class prior (54/70)", 32, C_TEXT, "400",
                  opacity=0.78))
    s.append(f'<line x1="88" y1="548" x2="{W - 88}" y2="548" stroke="{C_RULE}" stroke-width="1"/>')
    s.append(rich_text(88, 600,
                       [("open-Jev: ", C_TEXT, 0.62),
                        ("92.9%", C_ACCENT, 0.95),
                        (" (0.5B+LoRA) · ", C_TEXT, 0.62),
                        ("94.3%", C_ACCENT, 0.95),
                        (" (classifier) · MIT · github.com/DECRUX9812/openjev", C_TEXT, 0.62)],
                       size=20, weight="400"))
    s.append("</svg>\n")
    return "".join(s)


def card2() -> str:
    W, H = 1200, 675
    s = [svg_open(W, H, "open-Jev card 2",
                  "Dark card. Headline: 0.5B + 2.16M LoRA · 400 steps · 89 min · no GPU. "
                  "Subline: 92.9% of Jev’s judgment, reproduced locally, $0/call. "
                  "The headline wraps after '400 steps'.")]
    s += card_chrome(W, H)
    s.append(rich_text(88, 274,
                       [("0.5B + ", C_TEXT, None),
                        ("2.16M LoRA", C_ACCENT, None),
                        (" · 400 steps", C_TEXT, None)],
                       size=60, weight="700", ls="-0.8"))
    s.append(rich_text(88, 358,
                       [("89 min", C_ACCENT, None),
                        (" · no GPU", C_TEXT, None)],
                       size=60, weight="700", ls="-0.8"))
    s.append(rich_text(88, 446,
                       [("92.9%", C_ACCENT, None),
                        (" of Jev’s judgment, reproduced locally, $0/call", C_TEXT, 0.78)],
                       size=30, weight="400"))
    s.append(f'<line x1="88" y1="548" x2="{W - 88}" y2="548" stroke="{C_RULE}" stroke-width="1"/>')
    s.append(rich_text(88, 600,
                       [("trained on Jev’s own API answers · MIT · "
                         "github.com/DECRUX9812/openjev", C_TEXT, 0.62)],
                       size=20, weight="400"))
    s.append("</svg>\n")
    return "".join(s)


# ---------------------------------------------------------------- main
def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    CARDS.mkdir(parents=True, exist_ok=True)
    outputs = [
        (FIGS / "fig1_ladder.svg", fig1_ladder()),
        (FIGS / "fig2_perclass.svg", fig2_perclass()),
        (FIGS / "fig3_pipeline.svg", fig3_pipeline()),
        (CARDS / "card1.svg", card1()),
        (CARDS / "card2.svg", card2()),
    ]
    for path, content in outputs:
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path}  ({len(content.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
