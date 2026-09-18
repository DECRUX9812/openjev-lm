import sys, time
sys.path.insert(0, '/home/ubuntu/repos/openjev')
from openjev import OpenJev
j = OpenJev()
rows = [{'title': f'Posting {i}', 'employer': f'Employer {i}', 'pay': '20.00 hourly'}
        for i in range(200)]
t = time.time(); d = j.decide(rows); dt = time.time() - t
print(f'{len(d)} decisions in {dt:.2f}s = {len(d)/dt:.0f} postings/sec, $0.00 spent')
