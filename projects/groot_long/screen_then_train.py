"""Predeclare the gate, then stop at the first eligible LIBERO-Long task."""
from pathlib import Path
import json
import subprocess
import sys

here=Path(__file__).resolve().parent
sft=sys.argv[1:]==['--sft']
assert sft or len(sys.argv)==1
out=here/('sft_hard_v1' if sft else 'hard_task_v1')
out.mkdir(exist_ok=False)
order=[9,8,3,4,2,6,5,1,0]
if sft:order.append(7)
def dump(name,data):
    (out/name).write_text(json.dumps(data,indent=2))
def run(*args):
    subprocess.run([sys.executable,'-u',str(here/'offpolicy.py'),*map(str,args),*(['--sft'] if sft else [])],check=True)
baseinfo={}
if sft:
    from sft_assets import MODEL,REV,SHARDS
    baseinfo=dict(model=MODEL,revision=REV,weight_shards_sha256=SHARDS)
else:baseinfo=dict(base_weights_sha256='eed4991e325b906d2976e53903fe3b5dd34bdf8b23961ca605db04d02313a5df')
dump('preregister.json',dict(order=order,exclude_previously_tested=None if sft else 7,selection='first eligible in fixed order',
     gate='3-6/10 distinct layouts, then 6-12/20 after repeating all ten with fresh inference seeds; at least two of the first three failed layouts fail with the same final goal flags on both seeds, a same-seed rerun and a third seed',
     exploration=list(range(10)),train=list(range(10,30)),validation=list(range(30,35)),test=list(range(35,50)),
     test_seed_offsets=[41000,51000],reward_penalty=0.1,epochs=[40,80,120],**baseinfo))
for idx in order:
    candidate=out/f'candidate_{idx}'; run(candidate,'--screen',idx)
    gate=json.loads((candidate/'gate.json').read_text())
    print('SCREEN',gate,flush=True)
    if gate['eligible']:
        p=json.loads((candidate/'protocol.json').read_text())
        lock={**gate,**{k:p[k] for k in ['hashes','test','validation','train','test_seed_offsets']}}
        if sft:lock['base']=p['base']
        dump('selected_task.json',lock);run(out/'td3','--penalty',0.1,'--selection',out/'selected_task.json')
        break
else:
    dump('no_eligible_task.json',dict(message='No eligible original policy task. No post-training started.',order=order))
