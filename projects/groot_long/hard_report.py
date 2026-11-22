"""Independently check screening evidence, split lock, replay and final pairs."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np

here=Path(__file__).resolve().parent; root=here/(sys.argv[1] if len(sys.argv)>1 else 'hard_task_v1')
def read(f):return json.loads(f.read_text())
def rows(f):return [json.loads(line) for line in f.read_text().splitlines()]
pre=read(root/'preregister.json')
table=[]; passed=[]
for idx in pre['order']:
    d=root/f'candidate_{idx}'
    gate=read(d/'gate.json'); data=rows(d/'episodes.jsonl'); protocol=read(d/'protocol.json')
    if 'revision' in pre:assert read(d/'loaded_sft.json')['revision']==protocol['base']['revision']==pre['revision']
    by={(r['stage'],r['trial']):r for r in data}
    assert len(by)==len(data) and all(r['trial'] in pre['exploration'] and r['init']==protocol['hashes'][r['trial']] and r['arm']=='base' and r['replay']==0 for r in data)
    primary=[by['screen',i] for i in pre['exploration']];score=sum(r['success'] for r in primary)
    assert score==gate['successes']
    eligible=False;repeated=[];confirmed=None
    if 3<=score<=6:
        confirmed=sum(by['confirm',i]['success'] for i in pre['exploration'])
        for r in [r for r in primary if not r['success']][:3]:
            i=r['trial']; seq=[by[st,i] for st in ['screen','replica','confirm','repeat_noise']]
            assert [s['seed'] for s in seq]==[61000+i,61000+i,71000+i,81000+i]
            if all(not s['success'] and s['final']==r['final'] for s in seq):repeated.append(i)
        eligible=6<=score+confirmed<=12 and len(repeated)>=2
    assert eligible==gate['eligible'];table.append(dict(task=gate['task'],primary=score,confirm=confirmed,repeated=repeated))
    if eligible:passed.append(idx);break
if not passed:
    assert (root/'no_eligible_task.json').is_file() and not (root/'td3').exists()
    if 'revision' in pre:
        (root/'verified.json').write_text(json.dumps(dict(screen=table,selected=None,trained=False),indent=2))
    print('No eligible task',table);sys.exit(0)
lock=read(root/'selected_task.json');assert lock['idx']==passed[0]
d=root/'td3';train_protocol=read(d/'protocol.json')
if 'revision' in pre:assert read(d/'loaded_sft.json')['revision']==lock['base']['revision']==train_protocol['base']['revision']==pre['revision']
assert all(lock[k]==train_protocol[k] for k in ['hashes','test','validation','train','test_seed_offsets'])
assert all(lock[k]==pre[k] for k in ['test','validation','train','test_seed_offsets'])
audit=json.loads(subprocess.check_output([sys.executable,str(here/'offpolicy_verify.py'),str(d)],text=True))
rr=rows(d/'episodes.jsonl'); train=[r for r in rr if r['stage']=='train']
assert len(train)==120 and all(r['replay']>0 for r in train)
for i in (40,80,120):
    assert all(r['replay']==train[i-1]['replay'] for r in rr if r['stage']==f'val_{i}')
pairs={}
for r in rr:
    if r['stage']=='test':
        pairs.setdefault((r['trial'],r['seed']),{})[r['arm']]=r
details=[dict(trial=k[0],seed=k[1],base=p['base']['success'],residual=p['residual']['success'],base_final=p['base']['final'],residual_final=p['residual']['final']) for k,p in sorted(pairs.items())]
delta=np.array([int(v['residual'])-int(v['base']) for v in details]).reshape(15,2).mean(1)
ci=np.quantile(np.random.default_rng(1307).choice(delta,(20000,15)).mean(1),[.025,.975])*100
assert np.allclose(ci,audit['test']['layout_bootstrap95_pp'])
report=dict(screen=table,task=lock['task'],split_lock_sha256=hashlib.sha256((root/'selected_task.json').read_bytes()).hexdigest(),audit=audit,pairs=details)
(root/'verified.json').write_text(json.dumps(report,indent=2))
t=audit['test'];s=read(d/'selection.json')
text=f"""# LIBERO-Long 筛选与离策略后训练

任务：{lock['task']}。探索初筛 {score}/10，换策略噪声 {confirmed}/10；布局 {repeated} 在同种子复跑、另一种子确认及第三种子复跑时均失败且末态目标标志一致。这是失败现象复现，不是根因诊断。

| 策略 | 测试成功 | 布局 | 每布局完整评估 |
|---|---:|---:|---:|
| {pre.get('model','冻结 GR00T RL-Long-Step300')} | {t['base']}/30 | 15 | 2 |
| GR00T + 验证选中的残差 TD3 | {t['td3']}/30 | 同上 | 同上 |

配对胜/负 {t['wins']}/{t['losses']}，净差 {t['diff_pp']:+.1f} 个百分点，按布局配对 bootstrap 95% 区间 [{ci[0]:.1f}, {ci[1]:.1f}]。每臂一个训练配置，区间不含跨训练种子不确定性。这是单项任务，不能代表整个套件。

训练 120 个真实仿真回合，{audit['train_control_steps']} 个控制步、{audit['replay_chunks']} 条动作块 transition，critic 更新 {s['updates']} 次、actor 更新 {s['actor_updates']} 次（总训练预算）；第 {s['checkpoint']} 回合权重由验证选定，三个验证点（各 5 个布局）分别成功 {audit['validation_successes']} 次。原模型不改权重，离策略 TD3 修正 XYZ，奖励含已满足谓词数的变化、双目标/全目标成功奖励及 0.1 倍归一化修正均方代价。测试成功仅由原任务的官方全目标判定决定。训练/验证/测试初态哈希不相交，训练 20 个布局，模型选择 5 个，独立测试 15 个，另 10 个布局只筛选。公开模型原训练与官方布局是否重合未知。

证据：preregister.json、selected_task.json、各 candidate_*/gate.json 和 episodes.jsonl、td3/selection.json、td3/results.json、verified.json 和 td3/*.mp4。运行：从工作区用 work/envs/gr00t_rl/bin/python projects/groot_long/screen_then_train.py （需要全新输出目录）；事后核验：work/envs/gr00t_rl/bin/python projects/groot_long/hard_report.py。
"""
(root/'RESULTS.md').write_text(text);print(text)
if 'revision' in pre:
    def tally(arm):
        ended=[tuple(pair[arm]['final']) for pair in pairs.values()]
        return dict(sorted((str(key),ended.count(key)) for key in set(ended)))
    stage=dict(predicate_order=lock['goals'],base=tally('base'),td3=tally('residual'))
    (root/'final_goal_counts.json').write_text(json.dumps(stage,indent=2))
    with (root/'RESULTS.md').open('a') as f:
        print('\n同任务在历史 RL-Step300 实验中评估过，因此这不是项目从未见过的测试布局。测试基线是本轮实际重新执行的 SFT-Long。完整复现步骤见 ../SFT_HARD_REPRO.md；运行筛选时需添加 --sft，运行核验时指定 sft_hard_v1。末态子目标分布见 final_goal_counts.json，仅描述现象。',file=f)
