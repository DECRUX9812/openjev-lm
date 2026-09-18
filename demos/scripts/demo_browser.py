"""openjev browser-use demo — triage a LIVE HN 'Who is hiring?' thread.

Fetches the newest Ask HN: Who is Hiring thread via the public Algolia API,
runs the 3MB local classifier on every real posting, and opens the flagged
ones in the browser. The model decides; the browser follows.
"""
import html, json, os, re, subprocess, sys, time, urllib.parse, urllib.request

sys.path.insert(0, '/home/ubuntu/repos/openjev')
from openjev.engine import OpenJev

ALG = 'https://hn.algolia.com/api/v1'

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'openjev-demo'})
    return json.load(urllib.request.urlopen(req, timeout=15))

def strip(t):
    t = re.sub(r'<a href="([^"]+)"[^>]*>.*?</a>', r'\1', t or '', flags=re.S)
    t = re.sub(r'<[^>]+>', ' ', t)
    return html.unescape(t)

def main():
    print('finding the latest "Ask HN: Who is hiring?" thread …')
    hits = get(f'{ALG}/search_by_date?tags=story&query=%22Ask%20HN:%20Who%20is%20hiring%22')['hits']
    story = next(h for h in hits
                 if re.match(r'Ask HN: Who is hiring\?\s*\(', h.get('title') or ''))
    sid, stitle = story['objectID'], story['title']
    print(f'→ {stitle}   (news.ycombinator.com/item?id={sid})\n')

    tree = get(f'{ALG}/items/{sid}')
    posts = []
    for c in tree.get('children') or []:   # top-level comments = the actual postings
        txt = strip(c.get('text'))
        if len(txt) > 60:
            posts.append((c['id'], txt))
    print(f'{len(posts)} real postings in the thread — deciding each locally\n')

    jev = OpenJev(weights_dir='/home/ubuntu/repos/openjev/weights')
    inf = 0.0
    rows = []
    for i, (cid, txt) in enumerate(posts[:24]):
        first = txt.split('\n')[0] or txt
        parts = [p.strip() for p in first.split('|')]
        title = first[:80]
        employer = parts[0][:60] if parts else ''
        t = time.time()
        d = jev.decide({'title': title, 'employer': employer, 'pay': ''})
        ms = (time.time() - t) * 1000
        inf += ms
        rows.append((cid, title, d))
        mark = {'service_lead': '★', 'junk': '✗'}.get(d.bucket, ' ')
        print(f'{mark} [{i+1:02d}] {title[:64]:<64} → {d.bucket:<13} ({d.bucket_confidence:.2f}) {ms:5.1f}ms')
        time.sleep(0.9)   # pacing for video

    from collections import Counter
    c = Counter(d.bucket for _, _, d in rows)
    print(f'\n{len(rows)} postings, {inf/len(rows):.1f}ms mean inference — API calls: 0, cost: $0')
    print('buckets:', dict(c))

    flagged = [r for r in rows if r[2].bucket != 'junk']
    pick = max(rows, key=lambda r: r[2].bucket_confidence)
    print(f'\nmodel pick → "{pick[1][:60]}"  ({pick[2].bucket} {pick[2].bucket_confidence:.2f})')
    url = f'news.ycombinator.com/item?id={pick[0]}'
    print(f'opening it in the browser: https://{url}')
    env = dict(os.environ, DISPLAY=os.environ.get('DISPLAY', ':0'))
    win = subprocess.run(['/opt/.devin/package/custom_binaries/xdotool', 'search', '--name', 'Hacker News'],
                         capture_output=True, text=True, env=env).stdout.split()[-1]
    subprocess.run(['/opt/.devin/package/custom_binaries/xdotool', 'windowactivate', '--sync', win], capture_output=True, env=env)
    time.sleep(1.0)
    subprocess.run(['/opt/.devin/package/custom_binaries/xdotool', 'key', 'ctrl+l'], capture_output=True, env=env)
    time.sleep(0.6)
    subprocess.run(['/opt/.devin/package/custom_binaries/xdotool', 'type', '--clearmodifiers', '--delay', '25', url],
                   capture_output=True, env=env)
    subprocess.run(['/opt/.devin/package/custom_binaries/xdotool', 'key', 'Return'], capture_output=True, env=env)
    time.sleep(5)
    print('\n3MB model. CPU. zero network for inference. github.com/DECRUX9812/openjev')

if __name__ == '__main__':
    main()
