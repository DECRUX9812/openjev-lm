#!/usr/bin/env python3
"""Build the open-Jev paper: markdown -> styled HTML -> PDF (chromium print).

Usage:  ~/.venvs/embed/bin/python build_paper.py
Figures referenced by <!--FIG:name--> markers; missing figures are skipped, not faked.
PDF: playwright chromium, A4, print backgrounds, footer with page numbers.
"""
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
FIGS = HERE / "figs"
MD = HERE / "openjev-paper.md"
HTML = HERE / "openjev-paper.html"
PDF = HERE / "openjev-paper.pdf"

CAPTIONS = {
    "fig1_ladder": ("fig1_ladder.svg", "The ladder. Bucket accuracy against 70 hand-labelled postings. The dashed line is the majority-class rate of the evaluation set (54/70) — the score of a model that always answers <code>generic_job</code>."),
    "fig2_perclass": ("fig2_perclass.svg", "Per-class recall on the same 70 rows. The whole difficulty of this task lives in the two middle bars of the left pair."),
    "fig3_pipeline": ("fig3_pipeline.svg", "Both arms, end to end. Top: the LM arm answers the typed schema through constrained decoding. Bottom: the classifier arm answers it through frozen embeddings and small heads. Both were trained only on Jev's own answers; both run offline on CPU."),
}

CSS = """
:root { --ink:#16181c; --muted:#5b6470; --rule:#e3e6ea; --accent:#0a7f5a; --bg:#fdfdfc; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin:0; background:var(--bg); color:var(--ink);
  font: 15.5px/1.72 Charter, Georgia, 'Times New Roman', serif; }
.page { max-width: 790px; margin: 0 auto; padding: 56px 40px 80px; }
h1 { font-family: Inter, 'Helvetica Neue', Arial, sans-serif; font-size: 30px; line-height:1.2;
  letter-spacing:-0.4px; margin: 0 0 10px; }
h2 { font-family: Inter, 'Helvetica Neue', Arial, sans-serif; font-size: 19px; letter-spacing:-0.2px;
  margin: 40px 0 10px; padding-top: 14px; border-top:1px solid var(--rule); }
h3 { font-family: Inter, 'Helvetica Neue', Arial, sans-serif; font-size: 16px; margin: 26px 0 8px; }
p { margin: 0 0 12px; }
em { color: var(--muted); }
strong { color: #000; }
a { color: var(--accent); text-decoration: none; }
hr { border:0; border-top:1px solid var(--rule); margin: 30px 0; }
code { font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace; font-size: 0.86em;
  background:#f2f4f6; border:1px solid var(--rule); border-radius:4px; padding:0.08em 0.32em; }
table { border-collapse: collapse; width:100%; margin: 14px 0 18px; font-size: 13.6px;
  font-family: Inter, 'Helvetica Neue', Arial, sans-serif; }
th { text-align:left; font-weight:600; border-bottom: 2px solid #cfd4da; padding: 7px 9px; }
td { border-bottom: 1px solid var(--rule); padding: 6px 9px; vertical-align: top; }
tr:last-child td { border-bottom: 1.5px solid #cfd4da; }
blockquote { margin: 14px 0; padding: 2px 16px; border-left: 3px solid var(--accent); color: var(--muted); }
figure { margin: 22px 0 26px; }
figure img, figure svg { width: 100%; height: auto; }
figcaption { font-family: Inter, 'Helvetica Neue', Arial, sans-serif; font-size: 12.4px;
  color: var(--muted); margin-top: 8px; line-height:1.5; }
.meta { font-family: Inter, 'Helvetica Neue', Arial, sans-serif; font-size: 13px; color: var(--muted);
  margin: 0 0 26px; padding-bottom: 18px; border-bottom: 1px solid var(--rule); }
.abstract { font-size: 14.4px; background:#f5f7f6; border:1px solid var(--rule); border-radius:8px;
  padding: 16px 18px 8px; margin: 0 0 10px; }
.abstract h2 { border:0; margin:0 0 6px; font-size: 13px; letter-spacing:1.2px; text-transform:uppercase; color:var(--muted); padding:0; }
@media print {
  .page { max-width:none; padding: 0; }
  h2 { page-break-after: avoid; }
  figure, table { page-break-inside: avoid; }
  a { color: var(--ink); text-decoration: none; }
}
"""


def build_html() -> str:
    md = MD.read_text()
    html = markdown.markdown(md, extensions=["tables", "sane_lists", "attr_list"])

    def fig_block(name: str, idx: int) -> str:
        fname, cap = CAPTIONS[name]
        f = FIGS / fname
        if not f.exists():
            return f"<p><em>[figure pending: {fname}]</em></p>"
        return (f'<figure><img src="figs/{fname}" alt="{name}">'
                f'<figcaption><strong>Figure {idx}.</strong> {cap}</figcaption></figure>')

    order = ["fig3_pipeline", "fig1_ladder", "fig2_perclass"]
    idx = {name: i + 1 for i, name in enumerate(order)}
    for name in order:
        html = html.replace(f"<!--FIG:{name}-->", fig_block(name, idx[name]))
    html = html.replace("<!--", "").replace("-->", "")

    # abstract box: turn the first <h2>Abstract</h2> block into a styled box
    html = html.replace("<h2>Abstract</h2>", '<div class="abstract"><h2>Abstract</h2>')
    html = html.replace("</p>\n<hr />\n<h2>1. Introduction</h2>",
                        '</div>\n<h2>1. Introduction</h2>', 1)

    # author/meta line
    html = html.replace("<p><strong>Ritesh Patel</strong> — independent researcher",
                        '<p class="meta"><strong>Ritesh Patel</strong> — independent researcher', 1)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>open-Jev: reproducing a hosted decision model's judgment locally, for nothing</title>
<style>{CSS}</style></head><body><div class="page">{html}</div></body></html>"""


def main():
    html = build_html()
    HTML.write_text(html)
    print(f"html written: {HTML} ({len(html)} bytes)")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(HTML.as_uri())
        pg.wait_for_timeout(600)
        pg.pdf(path=str(PDF), format="A4", print_background=True,
               display_header_footer=True,
               header_template='<div style="font:10px Inter,Arial;color:#8b94a1;width:100%;padding:6px 14mm 0;text-align:right;">open-Jev · September 2026</div>',
               footer_template='<div style="font:10px Inter,Arial;color:#8b94a1;width:100%;padding:0 14mm 6px;display:flex;justify-content:space-between;"><span>Ritesh Patel · DECRUX9812</span><span class="pageNumber"></span></div>',
               margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})
        b.close()
    print(f"pdf written:  {PDF} ({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
