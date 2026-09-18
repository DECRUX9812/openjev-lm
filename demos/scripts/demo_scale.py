"""openjev-lm scale demo — raw dev traffic through the 0.5B adapter, live.

Streams rows from data/dev_jev.jsonl (an unseen split shipped in the repo)
through the LM arm, tracking agreement vs Jev's stored answers in the margin.
On a 100-row probe this lands ~96%. Counter + $ ticker on every row.
"""
import json, sys, time

sys.path.insert(0, '/home/ubuntu/repos/openjev-lm/openjev')
import eval_likelihood as E

ROWS = '/home/ubuntu/repos/openjev-lm/data/dev_jev.jsonl'
N = 50

def main():
    rows = [json.loads(l) for l in open(ROWS)][:N]
    print(f'dev split: {len(rows)} unseen postings with Jev-stored answers — live triage\n')
    model, tok, _ = E.load_model('/home/ubuntu/repos/openjev-lm/models/Qwen2.5-0.5B-Instruct', 8, want_lora=True)
    state, step = E.load_lora_state('/home/ubuntu/repos/openjev-lm/openjev/runs/retrain-check/adapter.pt')
    print(E.apply_adapter(model, state), f'(adapter step {step})\n')
    stats = {"boundary_fallbacks": 0}
    agree = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        title = r['prompt'].split('title: ', 1)[1].split('\n', 1)[0][:52]
        t = time.time()
        out = E.score_row(model, tok, r, stats, use_enum=False)
        b = out['staged']['bucket']
        jb = (r.get('jev') or {}).get('bucket')
        agree += (b == jb)
        sec = time.time() - t
        mark = ' ' if b == jb else 'x'
        print(f'{mark} [{i+1:2d}] {title:<52} → {b:<13} (jev: {jb:<13}) '
              f'| agree {agree/(i+1)*100:5.1f}% | {sec:3.1f}s | $0.0000')
        time.sleep(0.5)
    print(f'\n{agree}/{len(rows)} agreement with the hosted model — on raw unseen traffic.')
    est = len(rows) * 90
    print(f'~{est:,} input tokens ≈ ${est*0.042/1e6:.4f} at Jev list price — here: $0.0000, offline.')
    print('and when you need bulk: the 3MB classifier does 490 postings/s. two arms, one repo.')

if __name__ == '__main__':
    main()
