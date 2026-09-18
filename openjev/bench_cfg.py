"""Throughput bench for the CPU LoRA run: pick (batch, max_len) and project wall time."""
import json, statistics, sys, time
from pathlib import Path
sys.path.insert(0, "/home/decrux/Code/jev-repro-test/openjev")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_lora import apply_lora, make_batch, load_rows

ROOT = Path("/home/decrux/Code/jev-repro-test")
MODEL = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct")
DATA = ROOT / "openjev" / "data"
torch.set_num_threads(6)

tok = AutoTokenizer.from_pretrained(MODEL)
rows = load_rows(DATA / "train.jsonl", tok, 99999)
L = sorted(len(r["ids"]) for r in rows)
print("token lens: mean %.0f  p50 %d  p90 %d  p95 %d  p99 %d  max %d" % (
    statistics.mean(L), L[len(L)//2], L[int(.90*len(L))], L[int(.95*len(L))], L[int(.99*len(L))], L[-1]))
pad_id = tok.pad_token_id or tok.eos_token_id

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
apply_lora(model, 16, 32)
params = [p for p in model.parameters() if p.requires_grad]
model.train()

# length-bucketed batches: sort rows by length, take consecutive slices -> minimal padding
rows.sort(key=lambda r: len(r["ids"]))
q = len(rows) // 4
print("\nconfigs (length-bucketed batches, 1 accum = 1 step):")
for batch, label in ((6, "b6"), (8, "b8"), (12, "b12")):
    for rep in range(2):
        idx = torch.randint(0, len(rows) - batch, (1,)).item()
        sel = rows[idx:idx + batch]
        ids, attn, labs = make_batch(sel, pad_id)
        t0 = time.time()
        out = model(input_ids=ids, attention_mask=attn, labels=labs, use_cache=False)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        dt = time.time() - t0
        if rep == 1:
            ntok = int(attn.sum())
            print(f"  {label}  seq={ids.shape[1]:4d}  {ntok:5d} tok  {dt:6.1f}s/step  -> 200 steps {200*dt/60:5.1f}min | 400 {400*dt/60:5.1f}min | 800 {800*dt/60:5.1f}min")
