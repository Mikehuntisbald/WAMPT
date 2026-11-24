# 配对复现 PPO 与改进 SAC

从仓库根目录运行。只复核已保存结果需要 Python 和 NumPy。脚本逐条核对 60 个布局的种子、初态及稳定后的物理和双相机指纹，再用相同的配对布局重算成功数和区间。

```bash
python projects/groot_long/paired_ppo_report.py
(cd weights && sha256sum -c SHA256SUMS)
```

GPU 仿真环境和冻结基座按[仓库首页](../../README.md)准备。训练独立执行 17 轮 PPO，保留第 6、12、17 轮权重。每轮在线收集 1,186 个五步动作块，再训练 10 个 epoch。验证用 5 个布局、每个布局两个推理种子，按成功数、满足目标数、较早轮次排序。测试使用封存的初态与每个控制步的共同 GR00T 噪声，SAC 与原始 SFT 的历史同布局记录放在 evidence/paired_ppo_v1/test。两个物体都入篮记为成功。

重新训练并评估使用新的输出目录。

```bash
export OMP_NUM_THREADS=2
export PAIRED_PPO_OUT="$PWD/runs/paired_ppo"
py=work/envs/gr00t_rl/bin/python
"$py" projects/groot_long/paired_ppo.py train
"$py" projects/groot_long/paired_ppo.py eval
```

只复跑已发布 PPO 的仿真评估，可将三个已发布权重复制到一个新的运行目录。

```bash
export OMP_NUM_THREADS=2
export PAIRED_PPO_OUT="$PWD/runs/paired_ppo_published"
mkdir -p runs
mkdir "$PAIRED_PPO_OUT" && cp weights/paired_ppo_v1/*.zip "$PAIRED_PPO_OUT/"
work/envs/gr00t_rl/bin/python projects/groot_long/paired_ppo.py eval
```

[训练配置](../../evidence/paired_ppo_v1/protocol.json)、[验证选择](../../evidence/paired_ppo_v1/selection.json)和[配对统计](../../evidence/paired_ppo_v1/results.json)均来自原实验。provenance.json 记录原实验工作区的源码指纹；本仓库将共用的 GR00T 和仿真函数集中在 paired_ppo_env.py，并通过仓库相对路径读取发布的证据和权重。样例视频见 [PPO 末轮](../../evidence/paired_ppo_v1/test/ppo_17/962400.mp4)与[改进 SAC](../../evidence/paired_ppo_v1/test/sac/962400.mp4)。
