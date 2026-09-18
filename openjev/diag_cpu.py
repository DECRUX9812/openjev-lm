"""Diagnose the terrible CPU fwd+bwd throughput: raw GEMM rate vs training-loop rate,
thread counts, and cgroup quota."""
import os, subprocess, sys, time, torch
from transformers import AutoModelForCausalLM

print("loadavg:", open("/proc/loadavg").read().strip())
try:
    print("cgroup cpu.max:", open("/sys/fs/cgroup/cpu.max").read().strip())
except Exception as exc:  # noqa: BLE001
    print("cgroup cpu.max: n/a", exc)
print("nproc:", os.cpu_count(), "| OMP:", os.environ.get("OMP_NUM_THREADS"), "| torch threads:", torch.get_num_threads())

for nt in (3, 6):
    torch.set_num_threads(nt)
    a = torch.randn(2048, 2048); b = torch.randn(2048, 2048)
    for _ in range(2):
        t0 = time.time(); c = a @ b; dt = time.time() - t0
    gf = 2 * 2048**3 / dt / 1e9
    print(f"matmul 2048^3 threads={nt}: {dt:.2f}s -> {gf:,.0f} GFLOPS")

m = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
m.train()
for nt in (3, 6):
    torch.set_num_threads(nt)
    for B, L in ((8, 128), (4, 256)):
        ids = torch.randint(0, 1000, (B, L))
        for _ in range(2):
            t0 = time.time()
            out = m(input_ids=ids, labels=ids, use_cache=False)
            out.loss.backward(); m.zero_grad(set_to_none=True)
            dt = time.time() - t0
        print(f"fwd+bwd B={B} L={L} threads={nt}: {dt:.2f}s -> {B*L/dt:,.0f} tok/s -> {6*0.5e9*B*L/dt/1e9:,.0f} GFLOPS", flush=True)
