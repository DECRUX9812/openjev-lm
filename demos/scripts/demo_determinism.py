import sys, hashlib, json
sys.path.insert(0, '/home/ubuntu/repos/openjev')
from openjev import OpenJev
j = OpenJev()
p = {'title': 'Web Designer', 'employer': 'PrintWest', 'pay': '28.00 hourly'}
h = [hashlib.sha256(json.dumps(j.decide(p).to_dict(), sort_keys=True).encode()).hexdigest()[:16]
     for _ in range(3)]
print('three runs, three hashes:', h, '-> identical:', len(set(h)) == 1)
