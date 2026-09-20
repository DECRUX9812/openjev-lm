"""The studio — a Canva-like surface where the model only ever runs once.

    python3 -m jevgraphic.cli edit draft.md --evidence paper.md --port 8791

Pipeline: prompt (or draft) -> gate once -> jev picks layout/style/art once.
Every later click — style, layout, glyphs — is a data-only re-render at $0.
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .compose import compose_svg, W, H
from .draft import load_draft
from .gate import get_gate
from .intent import get_intent
from .layouts import layout_names
from .model import Draft
from .synth import synthesize
from .styles import STYLE_PACKS, style_names
from .svgpng import export_png


class State:
    def __init__(self, draft_path: str = "", evidence_path: str = ""):
        self.draft_path = draft_path
        self.evidence_path = evidence_path
        self.evidence = Path(evidence_path).read_text() if evidence_path else ""
        self.idea = ""
        self.draft: Draft = load_draft(draft_path) if draft_path else Draft(title="JEV GRAPHIC STUDIO", subtitle="type an idea")
        self.receipt = None
        self.decision = None
        self.rerender_ms = 0.0

    def run_pipeline(self):
        gate = get_gate()
        self.draft, self.receipt = gate.decide(self.draft, self.evidence)
        self.decision = get_intent().decide(self.idea or self.draft.title, self.draft)
        return self.receipt, self.decision

    def make(self, idea: str):
        self.idea = idea
        self.draft = synthesize(idea, self.evidence)
        return self.run_pipeline()

    def render(self, style=None, layout=None, glyphs=None) -> str:
        d = self.decision
        if style or layout or glyphs is not None:
            from .model import Decision
            d = Decision(layout=layout or d.layout, style=style or d.style,
                         art=d.art, glyphs=d.glyphs if glyphs is None else glyphs,
                         confidence=d.confidence, scores=d.scores, backend=d.backend)
        return compose_svg(self.draft, d, self.receipt, W, H)


INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>jev graphic studio</title>
<style>
:root{--bg:#0b0e13;--panel:#12161d;--line:#1f2630;--ink:#dfe7f0;--muted:#5f6b7a;--accent:#4ea1ff;--green:#3ddc84;--pink:#ff5e8a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px 'SF Mono',Menlo,monospace}
header{display:flex;align-items:center;gap:14px;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
.logo{color:var(--accent);font-weight:700;letter-spacing:1px;white-space:nowrap}
#prompt{flex:0 0 300px;background:#0d1119;border:1px solid var(--line);color:var(--ink);padding:7px 10px;border-radius:6px;font:inherit}
#prompt:focus{outline:1px solid var(--accent)}
.chips{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.chip{background:#161b23;border:1px solid var(--line);border-radius:14px;padding:4px 10px;font-size:11px;white-space:nowrap}
.chip b{color:var(--accent)}
button{background:#2f6fed;color:#fff;border:0;border-radius:7px;padding:8px 14px;font:inherit;cursor:pointer}
.cols{display:grid;grid-template-columns:250px 1fr 300px;gap:14px;padding:14px}
.rail{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;max-height:calc(100vh - 120px);overflow:auto}
.rail h4{margin:0 0 10px;color:var(--muted);font-size:11px;letter-spacing:1px}
.claim{padding:8px 4px;border-bottom:1px solid var(--line)}
.claim .dot{color:var(--green);margin-right:6px}
.claim.refused .dot{color:var(--pink)}
.claim .meta{color:var(--muted);font-size:10px;margin-top:3px}
#canvas{background:#0d1119;border:1px solid var(--line);border-radius:10px;display:flex;align-items:flex-start;justify-content:center;padding:18px;min-height:70vh}
#canvas svg{max-width:100%;height:auto;box-shadow:0 8px 40px #000a}
.styles{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.style{border:1px solid var(--line);border-radius:8px;padding:8px;cursor:pointer}
.style.sel{border-color:var(--accent)}
.sw{display:flex;gap:3px;margin-bottom:5px}
.sw i{width:14px;height:14px;border-radius:3px;display:block}
.style .nm{font-size:10px;color:var(--ink);line-height:1.3}
.lchips{display:flex;flex-wrap:wrap;gap:6px}
.lchip{border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer;font-size:11px}
.lchip.sel{border-color:var(--accent);color:var(--accent)}
#rerender{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);background:#10161f;border:1px solid var(--line);border-radius:16px;padding:6px 14px;color:var(--green);font-size:11px}
label.t{display:flex;gap:8px;align-items:center;margin:12px 0;color:var(--muted)}
.note{color:var(--muted);font-size:11px;line-height:1.5}
</style></head><body>
<header>
<span class="logo">JEV GRAPHIC STUDIO</span>
<input id="prompt" placeholder="type an idea, jev builds it..." autofocus>
<span id="title"></span>
<span class="chips" id="chips"></span>
<button onclick="exportPng()">Export PNG</button>
</header>
<div class="cols">
<div class="rail" id="claims"></div>
<div id="canvas"></div>
<div class="rail">
<h4>STYLE</h4><div class="styles" id="styles"></div>
<h4 style="margin-top:16px">LAYOUT</h4><div class="lchips" id="layouts"></div>
<label class="t"><input type="checkbox" id="glyphs" checked onchange="rerender()"> Glyphs on cards</label>
<p class="note">The gate ran once, before this page loaded. Style and layout are data, so swapping them calls no model and costs nothing — which is why this is instant.</p>
</div></div>
<div id="rerender"></div>
<script>
let S={decision:{},receipt:{}};
async function state(){S=await (await fetch('/state.json')).json();
 document.getElementById('title').textContent=S.draft.title;
 const r=S.receipt;
 document.getElementById('chips').innerHTML=
  `<span class=chip>kept <b>${r.kept}</b></span><span class=chip>refused <b>${r.refused}</b></span>`+
  `<span class=chip>checked <b>${r.checked}</b></span><span class=chip>copy <b>${r.copied}</b></span>`+
  `<span class=chip>gate cost <b>$${r.cost_usd.toFixed(6)}</b></span><span class=chip>gate ms <b>${r.elapsed_ms|0}</b></span>`+
  `<span class=chip>model <b>${r.model}</b></span>`;
 const cl=document.getElementById('claims');
 cl.innerHTML='<h4>GATE — DECIDED ONCE</h4>'+
  S.draft.claims.map(c=>`<div class="claim ${c.kept?'':'refused'}"><span class=dot>●</span>${c.text.slice(0,90)}`+
   `<div class=meta>figure ${c.figure.toFixed(2)} · prose ${c.prose.toFixed(2)}${c.note?' · '+c.note:''}</div></div>`).join('');
 const st=document.getElementById('styles');
 st.innerHTML=S.styles.map(s=>`<div class="style ${s==S.decision.style?'sel':''}" onclick="pick('style','${s}')">`+
  `<div class=sw>${S.swatches[s].map(c=>`<i style="background:${c}"></i>`).join('')}</div>`+
  `<div class=nm>${S.labels[s]}</div></div>`).join('');
 const ly=document.getElementById('layouts');
 ly.innerHTML=S.layouts.map(l=>`<div class="lchip ${l==S.decision.layout?'sel':''}" onclick="pick('layout','${l}')">${l}</div>`).join('');
 document.getElementById('glyphs').checked=S.decision.glyphs;
}
function pick(k,v){S.decision[k]=v;rerender();state0()}
async function state0(){const st=document.getElementById('styles');
 st.querySelectorAll('.style').forEach(e=>e.classList.remove('sel'));
 document.getElementById('layouts').querySelectorAll('.lchip').forEach(e=>e.classList.toggle('sel',e.textContent==S.decision.layout));}
async function rerender(){const q=`?style=${S.decision.style}&layout=${S.decision.layout}&glyphs=${document.getElementById('glyphs').checked?1:0}`;
 const r=await fetch('/poster.svg'+q);const t=await r.text();
 document.getElementById('canvas').innerHTML=t;
 document.getElementById('rerender').textContent=
  `re-render $0.000000 · ${r.headers.get('x-render-ms')||'?'} ms · no model called`;}
async function make(){const idea=document.getElementById('prompt').value.trim();if(!idea)return;
 document.getElementById('rerender').textContent='jev is deciding…';
 await fetch('/make',{method:'POST',body:JSON.stringify({idea})});await state();await rerender();}
document.getElementById('prompt').addEventListener('keydown',e=>{if(e.key=='Enter')make()});
function exportPng(){const svg=document.querySelector('#canvas svg');if(!svg)return;
 const x=new XMLSerializer().serializeToString(svg);const img=new Image();
 img.onload=()=>{const c=document.createElement('canvas');c.width=svg.viewBox.baseVal.width*2;c.height=svg.viewBox.baseVal.height*2;
  c.getContext('2d').drawImage(img,0,0,c.width,c.height);
  const a=document.createElement('a');a.download='jevgraphic.png';a.href=c.toDataURL('image/png');a.click();};
 img.src='data:image/svg+xml;base64,'+btoa(unescape(encodeURIComponent(x)));}
(async()=>{await state();await rerender()})();
</script></body></html>"""


def serve(draft_path: str = "", evidence_path: str = "",
          host: str = "127.0.0.1", port: int = 8791) -> int:
    state = State(draft_path, evidence_path)
    if draft_path or True:
        state.run_pipeline()

    class H(BaseHTTPRequestHandler):
        def _send(self, body, ctype="text/html", extra=None, code=200):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self._send(INDEX)
            if u.path == "/poster.svg":
                t0 = time.perf_counter()
                svg = state.render(
                    style=q.get("style", [None])[0],
                    layout=q.get("layout", [None])[0],
                    glyphs=None if "glyphs" not in q else q["glyphs"][0] == "1")
                ms = (time.perf_counter() - t0) * 1000
                return self._send(svg, "image/svg+xml",
                                  {"X-Render-Ms": f"{ms:.0f}"})
            if u.path == "/state.json":
                from dataclasses import asdict
                return self._send(json.dumps({
                    "draft": asdict(state.draft), "receipt": asdict(state.receipt),
                    "decision": asdict(state.decision),
                    "styles": style_names(), "layouts": layout_names(),
                    "labels": {n: p.label for n, p in STYLE_PACKS.items()},
                    "swatches": {n: list(p.swatch) for n, p in STYLE_PACKS.items()},
                }), "application/json")
            if u.path == "/shot.png":
                svg = state.render()
                tmp = Path("/tmp/jevgraphic_shot.svg")
                tmp.write_text(svg)
                out = Path("/tmp/jevgraphic_shot.png")
                if export_png(str(tmp), str(out)):
                    return self._send(out.read_bytes(), "image/png")
                return self._send("no exporter", "text/plain", code=500)
            return self._send("not found", "text/plain", code=404)

        def do_POST(self):
            if urlparse(self.path).path == "/make":
                n = int(self.headers.get("Content-Length", 0))
                idea = json.loads(self.rfile.read(n) or b"{}").get("idea", "")
                state.make(idea)
                return self._send(json.dumps({"ok": True}), "application/json")
            return self._send("not found", "text/plain", code=404)

        def log_message(self, *a):
            pass

    print(f"jev graphic studio → http://{host}:{port}")
    ThreadingHTTPServer((host, port), H).serve_forever()
    return 0
