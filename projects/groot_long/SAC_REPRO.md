# 双物体入篮：SAC 与 TD3 的离策略比较

基座是同一个公开 GR00T-SFT-Long；从零初始化残差 actor/critic，不从 TD3 权重接着训练。选定任务和双方配对的 init hash/种子来自既有 sft_hard_v1/selected_task.json。训练布局 10–29 各六次，验证 30–34，测试 35–49 每个两次。测试集在早先的 TD3 实验中已评估，SAC 只按验证选权重。本次不通过重筛任务或更换种子来挑结果。

同前次使用 Stable-Baselines3 实现（DSRL 仓库所附的 SB3，版本见 OFFPOLICY.md）。冻结基座，从 replay 训练 XYZ 残差，步长上限 ±0.12；沿用同一观察、batch 128、学习率 3e-4、折扣 0.99、修正代价 0.1、120 回合及验证选择规则。SAC 使用 tanh 高斯策略，初始均值 0、压缩前标准差 0.22，不叠加外部噪声，alpha 从 0.1 自动学习，目标熵 -3。评估使用残差分布的确定性输出；基座的生成噪声仍按配对种子。两种算法都每个动作块更新一次 critic；TD3 每两次更新一次 actor，SAC 每次更新 actor 和温度。相同回合预算不等于相同控制步数或参数更新次数。

从工作区根目录执行，输出目录不能已存在：

```bash
OMP_NUM_THREADS=2 work/envs/gr00t_rl/bin/python -u projects/groot_long/offpolicy.py \
  projects/groot_long/sft_hard_v1/sac --sft --algorithm sac --penalty 0.1 \
  --selection projects/groot_long/sft_hard_v1/selected_task.json
work/envs/gr00t_rl/bin/python projects/groot_long/sac_report.py
```

后者核对训练布局顺序、两次 SFT 基线、测试初态、噪声种子、验证选择、replay 与更新数，分别计算 SAC 相对 SFT 和相对 TD3 的布局配对 bootstrap 区间。不把同一布局的两个种子当两个独立布局。见 sft_hard_v1/sac/paired_comparison.json、episodes.jsonl 及首个测试布局的视频。
