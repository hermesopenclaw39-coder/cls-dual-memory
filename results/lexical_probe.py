import sys, re
sys.path.insert(0, 'src')
from facts_scaled import make_facts_diverse

def toks(s):
    return set(re.findall(r"[a-z0-9']+", s.lower()))

facts = make_facts_diverse(800)
for n in [50, 100, 200, 400, 800]:
    sub = facts[:n]
    keys = [toks(t) for t, _, _ in sub]
    hit = 0
    for i, (_, q, _) in enumerate(sub):
        qt = toks(q)
        best, best_len = -1, 1e9
        for j, k in enumerate(keys):
            inter = len(qt & k)
            if inter > best or (inter == best and len(k) < best_len):
                best, best_len, pick = inter, len(k), j
        hit += int(pick == i)
    print(f'{n:5d}  lexical-containment top-1: {hit/n*100:.1f}%')