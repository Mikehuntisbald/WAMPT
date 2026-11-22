"""Independent checks of split, update budget, selected checkpoint and paired episodes."""
import json
from pathlib import Path
import sys
import numpy as np

d=Path(sys.argv[1]); p=json.loads((d/'protocol.json').read_text()); rows=[json.loads(line) for line in (d/'episodes.jsonl').read_text().splitlines()]
groups=[set(p[k]) for k in ('train','validation','test','historical_excluded')]
assert len(set.union(*groups))==sum(map(len,groups))==50==len(set(p['hashes']))
for r in rows:
    split='validation' if r['stage'].startswith('val_') else r['stage']
    assert r['trial'] in p[split] and r['init']==p['hashes'][r['trial']]
    assert 1<=r['steps']<=512 and r['success']==all(r['final'])
train=[r for r in rows if r['stage']=='train']
assert len(train)==120 and all(sum(r['trial']==k for r in train)==6 for k in groups[0])
chunks=sum((r['steps']+4)//5 for r in train)
s=json.loads((d/'selection.json').read_text())
assert s['replay']==chunks and s['updates']==chunks-255
v=[]
for i in p['checkpoints']:
    batch=[r for r in rows if r['stage']==f'val_{i}']
    assert len(batch)==len(groups[1]) and {r['trial'] for r in batch}==groups[1]
    v.append((sum(r['success'] for r in batch),sum(sum(r['final']) for r in batch),-i))
assert -max(v)[2]==s['checkpoint']
t=[r for r in rows if r['stage']=='test']
assert len(t)==60 and all(r['stage']=='test' for r in rows[-60:])
expected={(trial,trial+offset) for trial in groups[2] for offset in p['test_seed_offsets']}
a,b=[{(r['trial'],r['seed']):r for r in t if r['arm']==arm} for arm in ('base','residual')]
assert set(a)==set(b)==expected
diff=np.array([int(b[k]['success'])-int(a[k]['success']) for k in sorted(a)])
q=json.loads((d/'results.json').read_text())
assert (q['wins'],q['losses'])==(sum(diff==1),sum(diff==-1))
assert [q['base'],q[p.get('algorithm','td3')]]==[sum(r['success'] for r in items.values()) for items in (a,b)]
print(json.dumps(dict(verified=True,train_control_steps=sum(r['steps'] for r in train),replay_chunks=chunks,
    validation_successes=[x[0] for x in v],test=q),indent=2))
