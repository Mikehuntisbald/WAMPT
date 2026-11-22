# 对齐后续策略和 SAC 熵项的反事实排序实验

读取 branch_q_v1 保存的 8 个训练侧布局、16 个状态、四个确定性首块候选以及三个线上双 Q 的分数。后续 GR00T 噪声与上次实验配对。每次从官方初态重放保存的前缀，同时核对物理状态、warm start、控制量、两路相机、目标标志；首动作和缓存 Q 经过复核。四个候选顺序为零、原 SAC、改进 SAC、TD3。每个首块为五个控制步。后续用自己的 actor 对每个新 GR00T 提议做修正，无重新训练。两个 SAC 用 tanh 压缩的高斯采样和同次采样产生的联合 log_prob；另设 CPU 采样种子，四个首块干预间共享噪声。TD3 使用部署时的确定性均值，不附加训练时的 target policy smoothing 噪声。因此 TD3 的部署后续策略虽已对齐，不是精确地复刻带 target smoothing 的训练算子。

对于固定的首动作 a0，令每个动作块 r_j 为目标数量增量、终端成功奖励 2 和该块归一化残差均方代价 -0.1 MSE 之和。任务列只计前两项；shaped 列额外扣每块残差代价。soft 列 = shaped + sum_{j>=1} gamma^j (-alpha log pi(a_j|s_j))，其中 gamma=0.99，alpha 取各 checkpoint 的 exp(log_ent_coef)，TD3 的熵项为零。首动作是外部指定且 Q 已条件化，不加首块熵；无终端后 bootstrap。连续分布的 -log pi 可以为负。log_prob 对象是训练使用的归一化三维残差，不是缩放后笛卡尔动作或基础 VLA 的随机变量。比较使用样本回报而非另一个 critic 的估计。

每个状态 × 每种后续策略 × 四个首块 × 三组噪声正式运行，另每种后续策略/状态对第一组噪声重复零修正。每组 13 条完整结果通过重复性验证后原子落盘，可从最后完整组继续。每个文件保存每块末步、任务奖励、残差成本、log_prob 和末态目标。不会把同一个布局的两时刻或多条噪声分支当独立布局。

复现：从工作区运行以下命令。单进程可完成全部；可指定不重叠的 --layouts 分片计算。

```bash
OMP_NUM_THREADS=2 work/envs/gr00t_rl/bin/python -u projects/groot_long/branch_q_own.py projects/groot_long/sft_hard_v1/branch_q_own_v1
work/envs/gr00t_rl/bin/python projects/groot_long/branch_q_own_report.py projects/groot_long/sft_hard_v1/branch_q_own_v1
```

分析分三列：原来 SFT 接手的含代价 MC；自己的 actor 接手的含代价 MC；同一批新轨迹加上熵后的 MC。两两排名先在每个状态内对每动作的三次 MC 取均值，然后比较方向。六对中回报平局不判对错、Q 平局记半分，固定容差 1e-6。主对比是旧列和目标匹配列在**相同动作对**上的对错翻转；报告修复、仍错和新增错误，按布局配对 bootstrap。报告逐状态 top-1、regret、Q 相对 MC 的平均偏差和误差（有限样本 MC 并非精确真值）。每种 critic 的目标回报不同，不能仅比较各自正确率就将三者当成相同难度的考试。训练侧状态、仅一个训练种子、只有三组噪声意味着不能完全归因所有错排。
