"""Within-state ranking, not correlations pooled across states of different difficulty."""
from pathlib import Path
import itertools,json,sys
import numpy as np

root=Path(sys.argv[1])
pre=json.loads((root/'protocol.json').read_text()); arms=pre['arms']; critics=arms[1:]
rows=[json.loads(line) for line in (root/'branches.jsonl').read_text().splitlines()]
records=[];tol=1e-6;repeatability={}
for path in sorted(root.glob('*_*.json')):
    meta=json.loads(path.read_text())
    if 'q_heads' not in meta:continue
    data=[r for r in rows if r['state']==path.stem]
    assert len(data)==13 and len({(r['arm'],r['repeat']) for r in data})==13
    lookup={(r['arm'],r['repeat']):r for r in data}
    duplicate=lookup['zero_repeat',0];control=lookup['zero',0]
    assert all(v==control[k] for k,v in duplicate.items() if k!='arm')
    query={c:np.min(meta['q_heads'][c],axis=1) for c in critics}
    assert all(np.isfinite(v).all() for v in query.values())
    for metric in ('shaped_return','task_return','success'):
        sample=np.array([[lookup[a,i][metric] for a in arms] for i in range(3)],float)
        actual=sample.mean(0);maximum=actual.max();spread=np.ptp(actual)
        check=repeatability.setdefault(metric,dict(matches=0.,comparable=0,both_tied=0,one_tied=0))
        for a,b in itertools.combinations(range(4),2):
            gap=sample[:,a]-sample[:,b]
            for i,j in itertools.combinations(range(3),2):
                tied_i,tied_j=abs(gap[i])<=tol,abs(gap[j])<=tol
                if tied_i or tied_j:check['both_tied' if tied_i and tied_j else 'one_tied']+=1
                else:check['comparable']+=1;check['matches']+=bool(gap[i]*gap[j]>0)
        for c,q in query.items():
            wins=den=tied=stable_n=stable_w=large_n=large_w=0.
            for a,b in itertools.combinations(range(4),2):
                d=actual[a]-actual[b]
                if abs(d)<=tol:tied+=1;continue
                den+=1;dq=q[a]-q[b];correct=.5 if abs(dq)<=tol else float(d*dq>0)
                wins+=correct
                if abs(d)>=0.10:large_n+=1;large_w+=correct
                per_seed=sample[:,a]-sample[:,b]
                if np.all(per_seed>tol) or np.all(per_seed<-tol):stable_n+=1;stable_w+=correct
            pred=np.flatnonzero(q>=q.max()-tol);top=float(np.mean(actual[pred]>=maximum-tol))
            records.append(dict(state=path.stem,layout=meta['t'],metric=metric,critic=c,wins=wins,den=den,return_ties=tied,stable_n=stable_n,stable_w=stable_w,large_n=large_n,large_w=large_w,top=top,random_top=float(np.mean(actual>=maximum-tol)),informative=bool(spread>tol),regret=float(maximum-actual[pred].mean()),random_regret=float(maximum-actual.mean()),pred=[arms[i] for i in pred],q=q.tolist(),mc=actual.tolist()))
assert len(records)==len({r['state'] for r in rows})*9 and all(any(r['layout']==t for r in records) for t in pre['layouts'])
assert {(r['layout'],r['step']) for r in rows}==set(itertools.product(pre['layouts'],pre['prefix_steps']))
assert len(rows)==len(pre['layouts'])*len(pre['prefix_steps'])*(4*pre['repeats']+1)
def ratio(num,den):return float(num/den) if den else None
summary={}
for metric in ('shaped_return','task_return','success'):
    summary[metric]={}
    for c in critics:
        rr=[r for r in records if r['metric']==metric and r['critic']==c];ii=[r for r in rr if r['informative']]
        w,n=np.array([[sum(r[key] for r in rr if r['layout']==l) for l in pre['layouts']] for key in ('wins','den')])
        ix=np.random.default_rng(81122).integers(0,len(n),size=(20000,len(n)));bd=n[ix].sum(1);bw=w[ix].sum(1)
        summary[metric][c]=dict(pair_correct=w.sum(),pair_n=n.sum(),agreement=ratio(w.sum(),n.sum()),ci95_layout=np.quantile(bw[bd>0]/bd[bd>0],[.025,.975]).tolist() if np.any(bd>0) else None,eligible_states=len(ii),top_best=ratio(sum(r['top'] for r in ii),len(ii)),random_top=ratio(sum(r['random_top'] for r in ii),len(ii)),regret=float(np.mean([r['regret'] for r in rr])),random_regret=float(np.mean([r['random_regret'] for r in rr])),stable_correct=sum(r['stable_w'] for r in rr),stable_n=sum(r['stable_n'] for r in rr))
valid=[r for r in rows if r['arm']!='zero_repeat']
for metric,cc in summary.items():
    for c,values in cc.items():
        sub=[r for r in records if r['metric']==metric and r['critic']==c]
        values['posthoc_gap_ge_0.1']=[sum(r['large_w'] for r in sub),sum(r['large_n'] for r in sub)]
result=dict(states=len({r['state'] for r in rows}),rollouts=len(valid),duplicates=len(rows)-len(valid),sim_steps_all=sum(r['steps'] for r in rows),success={a:sum(r['success'] for r in valid if r['arm']==a) for a in arms},summary=summary,return_order_between_noise_repeats=repeatability)
(root/'ranking_details.json').write_text(json.dumps(records,indent=2));(root/'ranking.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
