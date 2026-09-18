"""openjev-lm race demo — why a decision model isn't a chat model.

Same posting, same base weights, two minds:
  LEFT:  the raw 0.5B base model free-generates an answer (streams tokens,
         rambles, takes seconds, unstructured text).
  RIGHT: the adapter answers via staged likelihood — typed JSON in ~0.3s.
Jev isn't "a small LLM that chats well" — it's a decision head. Watch why.
"""
import sys, time
sys.path.insert(0, '/home/ubuntu/repos/openjev-lm/openjev')
from dataset import prompt_for, KEYS
import eval_likelihood as E
import torch
from transformers import TextStreamer

MODEL = '/home/ubuntu/repos/openjev-lm/models/Qwen2.5-0.5B-Instruct'
ADAPTER = '/home/ubuntu/repos/openjev-lm/openjev/runs/retrain-check/adapter.pt'

POST = dict(title="Need a website for my plumbing business",
            employer="Mike's Plumbing (Austin TX)", pay="2,000 budget")

def main():
    print('one posting, two minds. same 0.5B base weights.\n')
    print(f'posting: "{POST["title"]}" — {POST["employer"]} — {POST["pay"]}\n')
    time.sleep(2)

    model, tok, _ = E.load_model(MODEL, 8, want_lora=True)
    prompt = prompt_for(POST['title'], POST['employer'], POST['pay'])
    ids = tok(prompt, return_tensors='pt').input_ids

    print('=== BASE MODEL — free generation (what everyone else ships) ===\n')
    t = time.time()
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=90, do_sample=False, streamer=streamer)
    gen_s = time.time() - t
    print(f'\n\n[{gen_s:.1f}s of generation — unstructured text, no schema guarantee]')
    time.sleep(6)

    state, step = E.load_lora_state(ADAPTER)
    print('\n=== ADAPTER — staged likelihood decision (openjev-lm) ===\n')
    print(E.apply_adapter(model, state), f'(adapter step {step})')
    t = time.time()
    row = {"id": "race", "title": POST['title'],
           "prompt": prompt}
    out = E.score_row(model, tok, row, {"boundary_fallbacks": 0}, use_enum=False)
    s = out['staged']
    dec_s = time.time() - t
    print(f'\n{dec_s:.2f}s → bucket={s["bucket"]}  fit={s["fit"]}  bools={s["bools"]}')
    for c in s['bucket_candidates']:
        print(f'   {c["value"]:<13} sum_logp {c["sum_logp"]:>8.3f}')
    print(f'\nrace: {gen_s:.1f}s of rambly text  vs  {dec_s:.2f}s of typed answer.')
    print('that gap IS the product. github.com/DECRUX9812/openjev-lm')

if __name__ == '__main__':
    main()
