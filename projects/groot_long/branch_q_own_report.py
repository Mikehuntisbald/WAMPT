"""Compare interpretable matched changes on identical (state, candidate-pair) units."""
import sys,itertools,json,hashlib
from pathlib import Path
import numpy as np

out=Path(sys.argv[1]); old=out.parent/'branch_q_v1'
read=lambda p:json.loads(p.read_text())
p=read(out/'protocol.json'); arms=p['arms'];critics=p['follow'];layouts=p['layouts']
assert hashlib.sha256((old/'protocol.json').read_bytes()).hexdigest()==p['parent_protocol_sha256']
prev={(r['state'],r['arm'],r['repeat']):r for line in (old/'branches.jsonl').read_text().splitlines() if (r:=json.loads(line))['arm']!='zero_repeat'}
pair_rows=[]; details=[]; tol=1e-6
def sign(x):return 0 if abs(x)<=tol else int(np.sign(x))
closed=duplicates=steps=0
for t,step in itertools.product(layouts,p['prefix_steps']):
    sid=f'{t}_{step}';meta=read(old/(sid+'.json'));assert meta['order']==arms
    first={}
    for c in critics:
        data=read(out/f'{sid}-{c}.json');idx={(r['arm'],r['repeat']):r for r in data}
        assert len(idx)==len(data)==13 and all(r['layout']==t and r['step']==step and r['state']==sid and r['follow']==c for r in data)
        assert all(idx['zero_repeat',0][k]==v for k,v in idx['zero',0].items() if k!='arm')
        duplicates+=1
        for a,i in itertools.product(arms,range(3)):
            row=idx[a,i];closed+=1;steps+=row['steps'] if 'steps' in row else row['end']-step
            h=row['first_physics'];assert first.setdefault(a,h)==h
            trace=row['trace'];assert trace[0][3]==0 and all(np.isfinite(x[1:4]).all() for x in trace)
            assert row['end']==trace[-1][0] and row['success']==all(row['final'])
            if c=='td3':assert all(x[3]==0 for x in trace)
            g,pen,logp=np.power(.99,np.arange(len(trace)))@np.asarray([x[1:4] for x in trace])
            assert np.allclose([g,g-pen,-p['alpha'][c]*logp,g-pen-p['alpha'][c]*logp],[row[k] for k in ['task','shaped','entropy','soft']],rtol=0,atol=1e-8)
        q=np.min(meta['q_heads'][c],axis=1)
        sample={'SFT':np.array([[prev[sid,a,i]['shaped_return'] for a in arms] for i in range(3)])}
        for k in ('shaped','soft','task','success'):sample[k]=np.array([[idx[a,i][k] for a in arms] for i in range(3)],float)
        mean={k:v.mean(0) for k,v in sample.items()}
        for k,v in mean.items():
            pred=q>=q.max()-tol;best=v>=v.max()-tol
            details.append(dict(layout=t,state=sid,critic=c,metric=k,q=q.tolist(),mc=v.tolist(),best=best.tolist(),chosen=pred.tolist(),top=float(best[pred].mean()),regret=float(v.max()-v[pred].mean()),bias=float((q-v).mean()),mae=float(abs(q-v).mean()),spread=float(np.ptp(v)),random_regret=float(v.max()-v.mean()),succ=sum(idx[a,i]['success'] for a,i in itertools.product(arms,range(3)))))
        for a,b in itertools.combinations(range(4),2):
            pred=sign(q[a]-q[b]);row=dict(state=sid,layout=t,critic=c,arms=[arms[a],arms[b]],pred=pred)
            for k,v in mean.items():
                d=sign(v[a]-v[b])
                row[k]=None if not d else (.5 if not pred else int(d==pred))
                row[k+'_difference']=float(v[a]-v[b])
            repeat_signs=np.sign(sample['soft'][:,a]-sample['soft'][:,b])
            row['stable']=bool(np.all(repeat_signs==1) or np.all(repeat_signs==-1))
            row['repeat_soft_gaps']=(sample['soft'][:,a]-sample['soft'][:,b]).tolist()
            pair_rows.append(row)

ix=np.random.default_rng(81122).integers(0,len(layouts),(20000,len(layouts)))
report={}
for c in critics:
    rr=[r for r in pair_rows if r['critic']==c];ds=[r for r in details if r['critic']==c]
    def stats(key,base=None):
        pairs=[r for r in rr if r[key] is not None and (base is None or r[base] is not None)]
        tot=np.asarray([sum(r[key]-(r[base] if base else 0) for r in pairs if r['layout']==l) for l in layouts])
        cnt=np.asarray([sum(r['layout']==l for r in pairs) for l in layouts]);den=cnt[ix].sum(1);valid=den>0
        return dict(score=float(tot.sum()),n=int(cnt.sum()),ci95=(np.quantile(tot[ix].sum(1)[valid]/den[valid],[.025,.975]).tolist() if valid.any() else None))
    def flips(a,b):
        return {name:sum(r[a]==x and r[b]==y for r in rr) for name,x,y in [('fixed',0,1),('still_wrong',0,0),('new_error',1,0),('still_right',1,1)]}
    s={}
    for key in ('SFT','shaped','soft','task','success'):
        d=[r for r in ds if r['metric']==key];inform=[r for r in d if r['spread']>tol]
        s[key]=dict(**stats(key),top=sum(r['top'] for r in inform),inform_states=len(inform),mean_regret=np.mean([r['regret'] for r in d]),random_regret=np.mean([r['random_regret'] for r in d]),mean_Q_minus_MC=np.mean([r['bias'] for r in d]),mae_of_MC_mean=np.mean([r['mae'] for r in d]))
    s['old_to_own_environment']=flips('SFT','shaped');s['own_environment_to_soft']=flips('shaped','soft')
    s['old_to_aligned']=flips('SFT','soft');s['aligned_minus_old']=stats('soft','SFT')
    stable=[r for r in rr if r['stable'] and r['soft'] is not None]
    s['stable_correct_n']=[sum(r['soft'] for r in stable),len(stable)]
    substantive=[r for r in rr if abs(r['soft_difference'])>=.1 and r['soft'] is not None]
    s['sensitivity_MC_gap_ge_0.1']=[sum(r['soft'] for r in substantive),len(substantive)]
    agreement_n=agreement_w=0
    for r in rr:
        for a,b in itertools.combinations(map(sign,r['repeat_soft_gaps']),2):
            if a and b:agreement_n+=1;agreement_w+=int(a==b)
    s['cross_repeat_soft_sign']=[agreement_w,agreement_n]
    report[c]=s
dump=lambda name,obj:(out/name).write_text(json.dumps(obj,indent=2))
dump('statewise.json',details);dump('pairwise.json',pair_rows)
dump('results.json',dict(states=len(layouts)*len(p['prefix_steps']),branches=closed,duplicates=duplicates,control_steps_without_duplicates=steps,report=report))
print('Verified:',closed,'formal branches,',duplicates,'duplicate controls.',flush=True)
for c,s in report.items():print(c,[(k,s[k]['score'],s[k]['n']) for k in ['SFT','shaped','soft']],s['old_to_aligned'],flush=True)
names={'sac':'原 SAC','new_sac':'改进 SAC','td3':'TD3'}
def frac(m):return f"{m['score']:g}/{m['n']}"
lines=['# 自身策略续跑的 Q 排序实验','',f'{closed} 条正式续跑、{duplicates} 条零修正重复对照。重复分支的完整奖励/熵序列、终止条件、首块及最终物理哈希一致。以每状态每动作三次续跑的平均值判断相对顺序；Q 是线上双头较小值。','',
'| Critic | SFT 接手（含代价） | 自身接手（含代价，不含熵） | 自身接手且目标对齐 | 自身目标下首选最佳 | 相对旧列的准确率差及按布局配对 95% 区间 |',
'|---|---:|---:|---:|---:|---:|']
for c,s in report.items():
    delta=s['aligned_minus_old'];ci=np.array(delta['ci95'])*100
    lines.append(f"| {names[c]} | {frac(s['SFT'])} | {frac(s['shaped'])} | {frac(s['soft'])} | {s['soft']['top']:g}/{s['soft']['inform_states']} | {100*delta['score']/delta['n']:+.1f} pp [{ci[0]:+.1f}, {ci[1]:+.1f}] |")
lines+=['','| Critic | 旧错→新对 | 旧错→仍错 | 旧对→新错 | 旧对→仍对 | Q−MC 均值 | Q 首选的平均回报损失 / 随机选择期望损失 |',
        '|---|---:|---:|---:|---:|---:|---:|']
for c,s in report.items():
    f=s['old_to_aligned'];v=s['soft']
    lines.append(f"| {names[c]} | {f['fixed']} | {f['still_wrong']} | {f['new_error']} | {f['still_right']} | {v['mean_Q_minus_MC']:+.3f} | {v['mean_regret']:.3f} / {v['random_regret']:.3f} |")
lines+=['','TD3 使用部署 actor 的确定性输出，没有训练 Bellman 目标的平滑噪声。SAC 使用各检查点自己的随机 actor、tanh 修正后的三维联合 log 密度、折扣后未来熵；固定首块不含熵。每块均扣执行残差的训练代价。成功与 soft return 不同：熵和成本会累积，动作提前结束也会改变二者。','',
        '两个反事实对照：SFT→自身不含熵观察换后续策略（含 SAC 探索）的作用；自身不含熵→自身含熵只改变同轨迹的评分。正确/错误以 MC 估计为参照，不视为无噪声真值。8 个训练侧布局、每布局两个状态、3 次推理噪声，区间只按布局聚类；未重复训练，不能将全部仍错项直接定性为 critic 缺陷。不同 critic 各自对应的目标也不同。','',
        '数据：results.json、pairwise.json、statewise.json、各状态各策略的逐块迹线。复现说明：../../BRANCH_Q_OWN_REPRO.md。']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
