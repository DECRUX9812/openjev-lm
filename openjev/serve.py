"""openjev/serve.py -- FastAPI sidecar: one adapter, three endpoints.

  POST /v1/decide     {title, employer, pay} -> staged Jev decision with
                      calibrated probabilities per field + latency.
  POST /v1/systemone  TypeSafe-shaped request {state, questions{noul|choice|score}}
                      answered by whole-answer likelihood scoring -- generic
                      questions, not just the Jev schema.
  GET  /v1/models, GET /api/info, GET /  (bundled playground)

Ported concepts (from jaredpalmer/kev): calibrated probabilities via one
fitted temperature (runs/demo/calibration.json), TypeSafe request/response
shapes, per-request latency + token accounting.

Run: python -m openjev.serve --port 8009   (or: python serve.py)
"""
from __future__ import annotations

import argparse, json, math, sys, threading, time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import eval_likelihood as E

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER = str(ROOT / "runs" / "run-final" / "adapter.pt")

STATE = {"model": None, "tok": None, "lock": threading.Lock(),
         "adapter_step": None, "temperature": 1.0, "device": "cpu",
         "calibration": {}}


def prompt_for(title: str, employer: str = "", pay: str = "") -> str:
    return ("<|im_start|>system\nDecide.<|im_end|>\n"
            "<|im_start|>user\n"
            f"title: {title}\nemployer: {employer}\npay: {pay}\n"
            "bucket(service_lead|staff_role|generic_job|junk) technical_need "
            "business_buyer small_firm_doable pay_stated evergreen_repost fit(0-4)"
            "<|im_end|>\n<|im_start|>assistant\n")


def probs(cands, T):
    """softmax over mean_logp / T -> probabilities aligned to candidate order."""
    mx = max(c["mean_logp"] for c in cands)
    exps = [math.exp((c["mean_logp"] - mx) / T) for c in cands]
    tot = sum(exps)
    return {c["value"] if "value" in c else c["label"]: exps[i] / tot
            for i, c in enumerate(cands)}


class DecideReq(BaseModel):
    title: str = ""
    employer: str = ""
    pay: str = ""
    text: str = ""          # free text -> stuffed into the title slot, on camera


class Noul(BaseModel):
    type: str = "noul"
    instructions: object = ""
    criteria: dict | None = None


class Choice(BaseModel):
    type: str = "choice"
    instructions: object = ""
    criteria: dict = {}


class Score(BaseModel):
    type: str = "score"
    instructions: object = ""
    criteria: list = []


class SysOneReq(BaseModel):
    state: object = ""
    model: str = "openjev-latest"
    questions: dict


def render(v, indent=0):
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list))
        else f"{pad}{k}: {render(x)}" for k, x in v.items())


def flat_prompt(state_text: str, instr: str) -> str:
    return ("<|im_start|>system\nDecide.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{state_text}\n{instr}<|im_end|>\n<|im_start|>assistant\n")


def score_options(prompt: str, options: list[str]):
    """Score arbitrary option strings as answer candidates; returns (labels, probs, n_tok)."""
    model, tok = STATE["model"], STATE["tok"]
    stats = {"boundary_fallbacks": 0}
    cands = [(o, o) for o in options]
    prefix = E.Prefix(model)
    scored, _ = E.score_stage(model, tok, prompt, cands, None, prefix, stats)
    ranked = E.rank(scored)
    T = STATE["temperature"]
    mx = max(c["mean_logp"] for c in ranked)
    exps = [math.exp((c["mean_logp"] - mx) / T) for c in ranked]
    tot = sum(exps)
    out = [(c["label"], exps[i] / tot) for i, c in enumerate(ranked)]
    return out


def r2(x):
    return round(float(x), 2)


def choice_confidence(p):
    K = len(p)
    return 1.0 if K == 1 else (max(p) - 1 / K) / (1 - 1 / K)


def score_confidence(p):
    L = len(p)
    mode = max(range(L), key=lambda i: p[i])
    return 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (L - 1)


app = FastAPI(title="openjev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/v1/decide")
def decide(r: DecideReq):
    title = r.title or r.text
    if not title.strip():
        raise HTTPException(422, "title or text required")
    model, tok = STATE["model"], STATE["tok"]
    stats = {"boundary_fallbacks": 0}
    with STATE["lock"]:
        t0 = time.time()
        res = E.score_row(model, tok, {"prompt": prompt_for(title, r.employer, r.pay)},
                          stats, use_enum=False)
        dt = time.time() - t0
    T = STATE["temperature"]
    st = res["staged"]
    field_probs, answers = {}, {}
    for f, fs in st["per_field"].items():
        field_probs[f] = probs(fs["candidates"], T)
        answers[f] = fs["chosen"]
    return {
        "model": "openjev-latest",
        "bucket": st["bucket"],
        "bucket_probabilities": field_probs["bucket"],
        "bucket_confidence": r2(choice_confidence(list(field_probs["bucket"].values()))),
        "fit": st["fit"],
        "fit_probabilities": field_probs["fit"],
        "bools": {b: answers[b] for b in E.BOOLS},
        "bool_probabilities": {b: field_probs[b] for b in E.BOOLS},
        "margins": {f: st["per_field"][f]["margin"] for f in st["per_field"]},
        "latency_ms": r2(dt * 1000),
        "cost_usd": 0.0,
        "temperature": T,
        "adapter_step": STATE["adapter_step"],
    }


@app.post("/v1/systemone")
def systemone(r: SysOneReq):
    """TypeSafe-shaped endpoint: typed questions over arbitrary state, answered
    by option likelihood scoring. Model was never trained on these questions."""
    state_text = render(r.state)
    answers, tok_in = {}, 0
    model, tok = STATE["model"], STATE["tok"]
    with STATE["lock"]:
        for qid, q in r.questions.items():
            qt = q.get("type", "choice")
            instr = render(q.get("instructions", ""))
            if qt == "noul":
                opts = ["no", "yes"]
            elif qt == "choice":
                opts = [f"{k}: {render(v)}" if v else str(k)
                        for k, v in (q.get("criteria") or {}).items()]
            elif qt == "score":
                opts = [render(x) for x in (q.get("criteria") or [])]
            else:
                raise HTTPException(422, f"unknown question type {qt}")
            if not opts:
                raise HTTPException(422, f"question {qid} has no options")
            t0 = time.time()
            ranked = score_options(flat_prompt(state_text, instr), opts)
            dt = time.time() - t0
            ps = [p for _, p in ranked]
            labels = [l for l, _ in ranked]
            if qt == "noul":
                yes = dict(ranked).get("yes", 0.0)
                answers[qid] = {"type": "noul", "noul": r2(yes)}
            elif qt == "choice":
                keys = list((q.get("criteria") or {}).keys())
                bykey = {k: r2(dict(ranked).get(f"{k}: {render(v)}", dict(ranked).get(k, 0.0)))
                         for k, v in (q.get("criteria") or {}).items()}
                best = max(bykey, key=bykey.get) if bykey else None
                answers[qid] = {"type": "choice", "choice": best,
                                "confidence": r2(choice_confidence(list(bykey.values()))),
                                "probabilities": bykey}
            else:
                score = sum(i * p for i, (_, p) in enumerate(ranked))
                answers[qid] = {"type": "score", "score": r2(score),
                                "confidence": r2(score_confidence(ps)),
                                "probabilities": {str(i): r2(p) for i, p in enumerate(ps)},
                                "legend": {str(i): o for i, o in enumerate(opts)}}
            tok_in += len(tok(state_text, add_special_tokens=False).input_ids)
            answers[qid]["latency_ms"] = r2(dt * 1000)
    return {"model": r.model, "answers": answers,
            "usage": {"input_tokens": tok_in}, "cost_usd": 0.0}


@app.get("/v1/models")
def models():
    return {"models": [{"id": "openjev-latest", "aliases": ["jev-latest"],
                        "base": "Qwen2.5-0.5B-Instruct",
                        "adapter_step": STATE["adapter_step"]}]}


@app.get("/api/info")
def info():
    return {"base": "Qwen2.5-0.5B-Instruct", "adapter_step": STATE["adapter_step"],
            "temperature": STATE["temperature"], "device": STATE["device"],
            "calibration": STATE["calibration"]}


@app.get("/")
def index():
    return FileResponse(HERE / "playground" / "index.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--threads", type=int, default=11)
    a = ap.parse_args()
    calibration = Path(a.adapter).resolve().parent / "calibration.json"
    cal = json.loads(calibration.read_text()) if calibration.exists() else {}
    T = cal.get("temperature", 1.0)
    model, tok, _ = E.load_model(a.model, a.threads, want_lora=True)
    state, step = E.load_lora_state(a.adapter)
    E.apply_adapter(model, state)
    STATE.update(model=model, tok=tok, adapter_step=step, temperature=T,
                 calibration=cal)
    print(f"openjev serving step {step} T={T} on :{a.port}")
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
