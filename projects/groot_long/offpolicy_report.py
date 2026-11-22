"""Create a report only after both fixed-budget experiments have finished."""
from pathlib import Path
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
rows=[]; arms=[]
for folder, label in [('td3_run','TD3'),('td3_cost','TD3 + 修正代价')]:
    directory=root/folder
    checked=json.loads(subprocess.check_output([sys.executable,str(root/'offpolicy_verify.py'),str(directory)],text=True))
    result=checked['test']
    entries=[json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
    tests={(e['trial'],e['seed']):e for e in entries if e['stage']=='test' and e['arm']=='base'}
    arms.append(tests)
    sel=json.loads((directory/'selection.json').read_text())
    line=[label,checked['train_control_steps'],sel['actor_updates'],' / '.join(map(str,checked['validation_successes'])),sel['checkpoint'],f"{result['td3']}/30",f"{result['diff_pp']:+.1f}",f"{result['wins']}/{result['losses']}",'至'.join(f'{v:.1f}' for v in result['layout_bootstrap95_pp'])]
    rows.append('| '+' | '.join(map(str,line))+' |')
assert set(arms[0])==set(arms[1])
for k in arms[0]:
    assert all(arms[0][k][field]==arms[1][k][field] for field in ['success','final','steps','init'])
base=sum(e['success'] for e in arms[0].values())
doc=f"""# 双物体入篮：离策略 RL 实测

同一个冻结的 GR00T Long-Step300，加一个学习 XYZ 微调的 TD3 网络。原策略在 15 个独立测试布局、每个布局两个噪声种子的基线是 **{base}/30**。基线在两条实验中分别重新执行，两份结果逐条核对。测试布局未参与本地训练或选权重。

| 组合策略 | 训练控制步 | actor 更新 | 三次验证成功数（各10） | 选中回合 | 无人成功 | 相对基线 pp | 配对胜/负 | 布局 bootstrap 95% 区间 pp |
|---|---:|---:|---|---:|---:|---:|---:|---|
""" + '\n'.join(rows) + """

第 40 回合的不加代价修正器在全部 3,522 条训练状态上都给出几乎相同的极限平移；三个轴满足 |输出|>0.95 的比例都是 100%。该批 replay 的正奖励只有 26 条。这个测试直接说明它当时没有根据状态灵活地改变平移方向，但无法仅据此归因于 critic 过估计或视觉误差。加代价臂在读取任一测试成绩前确定。

两臂均为一个训练种子，测试区间仅覆盖 15 个布局的采样不确定性。先前 4/5 的试跑来自另一组布局，不能用作这里的基线。详见 OFFPOLICY.md 的代码版本、动作范围、奖励、数据划分及复现方式。
"""
(root/'OFFPOLICY_RESULTS.md').write_text(doc)
print(doc)
