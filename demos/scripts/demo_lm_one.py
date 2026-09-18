"""Score one posting through the trained 0.5B LoRA adapter (staged likelihood path).

usage: demo_lm_one.py "title" "employer" "pay"
"""
import sys, time, json
sys.path.insert(0, '/home/ubuntu/repos/openjev-lm/openjev')
from dataset import prompt_for
import eval_likelihood as E

title, employer, pay = sys.argv[1], sys.argv[2], sys.argv[3]
model, tok, _ = E.load_model('/home/ubuntu/repos/openjev-lm/models/Qwen2.5-0.5B-Instruct', 8, want_lora=True)
state, step = E.load_lora_state('/home/ubuntu/repos/openjev-lm/openjev/runs/retrain-check/adapter.pt')
print(E.apply_adapter(model, state), f'(adapter step {step})')
row = {"id": "demo", "title": title, "prompt": prompt_for(title, employer, pay)}
t = time.time()
out = E.score_row(model, tok, row, {"boundary_fallbacks": 0}, use_enum=False)
s = out['staged']
print(f'\n"{title}" — {employer}')
print(f'bucket: {s["bucket"]}   fit: {s["fit"]}   bools: {s["bools"]}')
for c in s['bucket_candidates']:
    print(f'   {c["value"]:<13} sum_logp {c["sum_logp"]:>8.3f}')
print(f'({time.time()-t:.1f}s on CPU, offline)')
