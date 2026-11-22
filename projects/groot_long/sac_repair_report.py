"""Verify paired outcomes and actual optimizer counters against the pre-test choice."""
from pathlib import Path
import json
import sys
import numpy as np
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'sft_hard_v1'
out = root / 'sac_repair_v1'
def load(path):return json.loads(path.read_text())
def episodes(directory):return [json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
p=load(out/'repair_protocol.json'); lock=load(root/'selected_task.json');scores=load(out/'val_scores.json')
sel=load(out/'final_selection.json'); rows=episodes(out)
assert len(scores)==11 and max(scores,key=lambda r:r['rank'])['label']+'.zip'==sel['file']
assert len({*lock['train'],*lock['validation'],*lock['test']})==40
for name,warm,delay,alpha in p['arms']:
    audit=load(out/(name+'_warm_audit.json'))
    assert audit['critic_changed'] and audit['actor_changed']==(warm==0)
    for n in (2048,4096):
        score=next(s for s in scores if s['label']==f'{name}_{n}')
        assert score['counts']==dict(critic=[n],actor=[(n-warm)//delay],temperature=[(n-warm)//delay])
for s in scores:
    rr=[r for r in rows if r['stage']=='val_'+s['label']]
    assert len(rr)==10 and {(r['trial'],r['seed']) for r in rr}=={(i,i+o) for i in lock['validation'] for o in p['val_offsets']}
    assert s['rank']==[sum(r['success'] for r in rr),sum(sum(r['final']) for r in rr)]
train=[r for r in rows if r['stage']=='train']
assert len(train)==120 and all(sum(r['trial']==i for r in train)==6 for i in lock['train'])
chunks=sum((r['steps']+4)//5 for r in train)
assert chunks==sel['online_updates'] and sel['replay_final']==chunks+p['source_chunks']
for r in rows:
    assert r['init']==lock['hashes'][r['trial']] and r['success']==all(r['final'])
def test(directory,arm):
    rr=[r for r in episodes(directory) if r['stage']=='test' and r['arm']==arm];assert len(rr)==30
    return {(r['trial'],r['seed']):r for r in rr}
arms={'SFT':test(out,'base'),'SAC original':test(root/'sac','residual'),'TD3':test(root/'td3','residual'),'SAC new':test(out,'residual')}
keys=sorted(arms['SFT']);assert keys==sorted((t,t+off) for t in lock['test'] for off in p['test_offsets'])
assert all(sorted(v)==keys for v in arms.values())
for old in ('sac','td3'):
    oldbase=test(root/old,'base')
    assert all(all(oldbase[k][col]==arms['SFT'][k][col] for col in ('init','steps','final','success')) for k in keys)
by={}
for name,ref in arms.items():
    if name=='SAC new':continue
    diff=np.array([int(arms['SAC new'][k]['success'])-int(ref[k]['success']) for k in keys])
    ci=np.quantile(np.random.default_rng(1307).choice(diff.reshape(15,2).mean(1),(20000,15)).mean(1)*100,[.025,.975])
    by[name]=dict(wins=int(sum(diff>0)),losses=int(sum(diff<0)),delta_pp=100*diff.mean(),ci=ci.tolist())
result=dict(success={k:sum(r['success'] for r in v.values()) for k,v in arms.items()},vs=by,validation=scores,selected=sel,offline_choice=load(out/'offline_selection.json'),source_chunks=p['source_chunks'],new_steps=sum(r['steps'] for r in train),cheese_only={k:sum(r['final']==[False,True] for r in v.values()) for k,v in arms.items()})
(out/'verified.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='validation'},indent=2))
