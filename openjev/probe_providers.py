"""Probe teacher-model access + GPU options. Prints status only, never keys."""
import json, os, re, subprocess, time, urllib.request
from pathlib import Path

ENV = Path.home() / ".hermes" / ".env"
keys = {}
for line in ENV.read_text().splitlines():
    m = re.match(r'^([A-Z0-9_]+)=(.*)$', line.strip())
    if m and m.group(2).strip():
        keys[m.group(1)] = m.group(2).strip()

def post(url, key, payload, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return body, time.time() - t0

def try_provider(name, url, key_name, model, base=None):
    key = keys.get(key_name)
    if not key:
        print(f"{name}: NO KEY ({key_name})"); return
    base = base or url
    try:
        body, dt = post(base, key, {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ok"}], "max_tokens": 8})
        txt = body["choices"][0]["message"]["content"].strip()
        print(f"{name} [{model}]: OK {dt:.1f}s -> {txt!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name} [{model}]: FAIL {type(exc).__name__}: {str(exc)[:160]}")

try_provider("qwen-cloud", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", "DASHSCOPE_API_KEY", "qwen3.8-max-preview")
try_provider("cliproxy", "http://127.0.0.1:8317/v1/chat/completions", "CLIPROXYAPI_API_KEY", "gemini-3.6-flash")
try_provider("cliproxy", "http://127.0.0.1:8317/v1/chat/completions", "CLIPROXYAPI_API_KEY", "claude-sonnet-4.6")
try_provider("opencode-go", "https://opencode.ai/zen/go/v1/chat/completions", "OPENCODE_GO_API_KEY", "deepseek-v4.1-flash")

print("\n-- GPU options --")
print("modal config exists:", (Path.home() / ".modal.toml").exists())
for py in ["python3", str(Path.home() / "Code/jev-repro-test/Qwen-2.5-1B-RLCD/.venv/bin/python")]:
    try:
        out = subprocess.run([py, "-c", "import modal,sys;print(modal.__version__)"], capture_output=True, text=True, timeout=30)
        print(f"  {py}: modal {out.stdout.strip() or out.stderr.strip()[:60]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {py}: {exc}")
