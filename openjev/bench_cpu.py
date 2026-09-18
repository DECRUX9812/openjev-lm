"""Minimal CPU fwd+bwd timing so the training step budget is grounded in reality."""
import sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(6)
m = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
m.train()
for B, L in ((1, 128), (1, 192), (2, 256)):
    ids = torch.randint(0, 1000, (B, L))
    for it in range(3):
        t0 = time.time()
        out = m(input_ids=ids, labels=ids, use_cache=False)
        out.loss.backward()
        m.zero_grad(set_to_none=True)
        dt = time.time() - t0
    flops = 6 * 0.5e9 * B * L
    print(f"batch={B} seq={L}: {dt:.2f}s micro-step -> {B*L/dt:,.0f} tok/s -> {flops/dt/1e9:,.0f} GFLOPS", flush=True)
