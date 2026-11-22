"""Audit the new SAC run and pair it with the previously frozen TD3 results."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

root=Path(__file__).resolve().parent/'sft_hard_v1'
def read(path):return json.loads(path.read_text())
arm={}; audits={}; protocols={}; orders={}; val={}
for name in ('td3','sac'):
    d=root/name
    audits[name]=json.loads(subprocess.check_output([sys.executable,str(root.parent/'offpolicy_verify.py'),str(d)],text=True))
    protocols[name]=read(d/'protocol.json')
    assert protocols[name]['base']==read(root/'selected_task.json')['base']
    rows=[json.loads(x) for x in (d/'episodes.jsonl').read_text().splitlines()]
    train=[r for r in rows if r['stage']=='train']
    orders[name]=[(r['seed'],r['trial'],r['init']) for r in train]
    for i in (40,80,120):
        assert all(r['replay']==train[i-1]['replay'] for r in rows if r['stage']==f'val_{i}')
    tests=[r for r in rows if r['stage']=='test']
    for label,target in [('base',name+'_base'),('residual',name)]:
        arm[target]={(r['trial'],r['seed']):r for r in tests if r['arm']==label}
    val[name]=read(d/'selection.json')
assert orders['td3']==orders['sac']
for key in ('hashes','train','validation','test','test_seed_offsets','base','reward','scale','chunk_residual_penalty','select','lr','batch','discount','checkpoints'):
    assert protocols['sac'][key]==protocols['td3'][key],key
keys=sorted(arm['sac']);assert len(keys)==30
for r in arm.values():assert sorted(r)==keys
for k in keys:
    assert len({r[k]['init'] for r in arm.values()})==1
    for field in ('steps','success','final'):assert arm['sac_base'][k][field]==arm['td3_base'][k][field]
metrics=list(csv.DictReader((root/'sac/metrics/progress.csv').open()))
assert metrics and all(np.isfinite(float(row[k])) for k in ('train/actor_loss','train/critic_loss','train/ent_coef','train/ent_coef_loss') for row in metrics)
def compare(name,ref):
    diff=np.array([int(arm[name][k]['success'])-int(arm[ref][k]['success']) for k in keys])
    sampled=np.random.default_rng(1307).choice(diff.reshape(15,2).mean(1),(20000,15)).mean(1)*100
    return dict(wins=int((diff==1).sum()),losses=int((diff==-1).sum()),delta_pp=100*diff.mean(),ci95_pp=np.percentile(sampled,[2.5,97.5]).tolist())
names=('sac_base','td3','sac')
result=dict(success={name:sum(arm[name][k]['success'] for k in keys) for name in names},sac_vs_sft=compare('sac','sac_base'),sac_vs_td3=compare('sac','td3'),
   cheese_only={name:sum(arm[name][k]['final']==[False,True] for k in keys) for name in names},validation=audits['sac']['validation_successes'],
   selection=val['sac'],env_steps=audits['sac']['train_control_steps'],last_temperature=float(metrics[-1]['train/ent_coef']),paired_cases=[dict(layout=k[0],seed=k[1],**{n:arm[n][k]['success'] for n in names}) for k in keys])
assert np.allclose(result['sac_vs_sft']['ci95_pp'],audits['sac']['test']['layout_bootstrap95_pp'])
(root/'sac/paired_comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='paired_cases'},indent=2))
