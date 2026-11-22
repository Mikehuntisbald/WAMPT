"""Audit auxiliary SAC independently of the pure-SAC run."""
from pathlib import Path
import json
import sys
import numpy as np
root=Path(__file__).resolve().parent/'sft_hard_v1'; d=root/(sys.argv[1] if len(sys.argv)>1 else 'sac_anchor_v1')
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in (p/'episodes.jsonl').read_text().splitlines()]
p=read(d/'anchor_protocol.json'); s=read(d/'final_selection.json'); scores=read(d/'val_scores.json')
lock=read(root/'selected_task.json');r=rows(d);train=[i for i in r if i['stage']=='train']
assert len(train)==120 and all(sum(i['trial']==t for i in train)==6 for t in lock['train'])
assert all(i['init']==lock['hashes'][i['trial']] and i['success']==all(i['final']) for i in r)
chunks=sum((i['steps']+4)//5 for i in train)
assert s['updates']==4096+chunks and s['replay_final']==10359+chunks
assert s['counts']==dict(critic=[s['updates']+p['pretrain_steps']],actor=[s['updates'],s['updates']+p['pretrain_steps']],temperature=[s['updates']])
assert len(scores)==6 and s['label']==max(scores[1:],key=lambda v:v['rank'])['label']
for u in scores:
    rr=[i for i in r if i['stage']=='val_'+u['label']]
    assert len(rr)==10 and {(i['trial'],i['seed']) for i in rr}=={(t,t+o) for t in lock['validation'] for o in [30000,33000]}
    assert u['rank']==[sum(i['success'] for i in rr),sum(sum(i['final']) for i in rr)]
assert len(r)==120+60+60 and all(i['stage']=='test' for i in r[-60:])
def tests(directory,arm):
    rr=[i for i in rows(directory) if i['stage']=='test' and i['arm']==arm];assert len(rr)==30
    return {(i['trial'],i['seed']):i for i in rr}
keys=sorted((t,t+o) for t in lock['test'] for o in p['test_offsets'])
arms={'SFT':tests(d,'base'),'original SAC':tests(root/'sac','residual'),'TD3':tests(root/'td3','residual'),'SAC+BC':tests(d,'residual'),'pure SAC revision':tests(root/'sac_repair_v1','residual')}
assert all(sorted(v)==keys for v in arms.values())
for folder in ['td3','sac','sac_repair_v1']:
    other=tests(root/folder,'base')
    assert all(all(other[k][i]==arms['SFT'][k][i] for i in ('steps','final','init','success')) for k in keys)
comp={}
for name,a in arms.items():
    if name=='SAC+BC':continue
    diff=np.array([int(arms['SAC+BC'][k]['success'])-int(a[k]['success']) for k in keys]);ci=np.percentile(np.random.default_rng(1307).choice(diff.reshape(15,2).mean(1),(20000,15)).mean(1)*100,[2.5,97.5])
    comp[name]=dict(wins=int(sum(diff>0)),losses=int(sum(diff<0)),delta_pp=100*diff.mean(),ci=ci.tolist())
result=dict(success={a:sum(v[k]['success'] for k in keys) for a,v in arms.items()},vs=comp,selected=s,new_steps=sum(i['steps'] for i in train),validation=scores)
(d/'verified.json').write_text(json.dumps(result,indent=2));print(result['success'],s)
