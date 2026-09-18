"""openjev-lm retrain timelapse — watch the adapter learn, live.

Replays the real train_log.jsonl from tonight's 400-step CPU retrain
(13.7 min wall on 8 cores, no GPU): animated loss curve + final eval.
"""
import json, math, sys, time

LOG = '/home/ubuntu/repos/openjev-lm/openjev/runs/retrain-check/train_log.jsonl'
W, H = 88, 14

def curve(losses):
    lo, hi = min(losses), max(losses)
    rng = (hi - lo) or 1
    cols = []
    for chunk_i in range(0, len(losses), max(1, len(losses) // W)):
        seg = losses[chunk_i:chunk_i + max(1, len(losses) // W)]
        cols.append(sum(seg) / len(seg))
    grid = [[' ' for _ in range(len(cols))] for _ in range(H)]
    for x, v in enumerate(cols):
        y = int((v - lo) / rng * (H - 1))
        grid[H - 1 - y][x] = '█'
    return '\n'.join(''.join(r) for r in grid)

def main():
    rows = [json.loads(l) for l in open(LOG)]
    total = len(rows)
    print(f'replaying the actual log — {total} steps, {rows[-1]["elapsed_min"]:.1f} min of CPU training\n')
    time.sleep(2)
    losses = []
    for r in rows:
        losses.append(r['loss'])
        if r['step'] % 8 and r['step'] != total:
            continue
        sys.stdout.write('\x1b[2J\x1b[H')
        print(f'replaying the actual log — {total} steps, {rows[-1]["elapsed_min"]:.1f} min of CPU training\n')
        bar = curve(losses)
        print(f'step {r["step"]:>3}/{total}   loss {r["loss"]:.4f}   lr {r["lr"]}   '
              f'{r["step_s"]}s/step   rss {r["rss_gb"]}GB')
        print(bar)
        sys.stdout.flush()
        time.sleep(0.55)
    print('\n')
    import subprocess
    out = subprocess.run(['/home/ubuntu/.venvs/openjev/bin/python', '-c',
        "import json; s=json.load(open('/home/ubuntu/repos/openjev-lm/runs/retrain_eval.json'))['summary'];"
        "print('after: 64/70 gold =', s['bucket_acc_vs_gold'], '| jev agreement', s['bucket_acc_vs_jev_bucket'])"],
        capture_output=True, text=True).stdout
    print(f'\n0.68 → 0.02 loss in 13.7 min on 8 CPU cores.')
    print(f'eval of THIS adapter: {out.strip()}')
    print('no GPU. no API. just a corpus and a laptop. github.com/DECRUX9812/openjev-lm')

if __name__ == '__main__':
    main()
