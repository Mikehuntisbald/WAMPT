# 同状态分支：Q 能否选对首个动作块？

局部问题：给定完全相同的仿真状态、相机和 SFT 的五步原始提议，选择零残差、原 SAC、第 120 回合的改进 SAC、TD3 之一，之后统一使用冻结 SFT（零残差）。不加载教师，不更新参数。普通 SAC、改进 SAC 和 TD3 均用确定性 actor 产生干预；各自使用训练出来的 online critic 双头较小值评价全部四个残差。零残差没有自己的独立 critic。所有 checkpoint 及哈希见输出中的 protocol.json。

训练侧公开布局 10–17，先用零残差 SFT 走到第 30 和 120 步，按预先指定时间分支，若已终止则不选。初态按官方文件恢复，重放 15 步 settle 和同一串控制动作；避免把只含 qpos/qvel 的快照当成完整控制器。每条分支首步前校验模拟器状态、warmstart、ctrl 以及两路图像哈希。首块使用预先缓存的同一份 GR00T 动作提议与四份 XYZ residual，首块之后的每次推理重新输入真实分支图像和本体状态。相同状态、相同 repeat 的各臂使用相同推理噪声序列；不同 repeat 不共享噪声。每个状态 3 组 repeat，外加一条重复的零残差控制。重复控制的终态物理哈希、每块目标序列哈希、完成步数、全部回报和成功标志必须相同。保留下来的 prefix 和 action/feature 数组可独立检查。

上限 512 个控制步，从原 episode 开始计时；不是从分叉位置再算 512。每个 chunk 折扣 0.99；任务奖励 = 目标满足数的增量 + 2 × 官方成功。主指标再扣 0.1 × 首块归一化残差的均方；之后零残差无代价。按三次续跑的 MC 均值排序；一个状态内四选二共 6 对，MC 差≤1e-6 不作为有序对，Q 平局记半对；报按布局 bootstrap 区间。另报去修正代价的任务回报、二元成功率、赢家是否在 Q 的首选中（实际并列最佳都算对），以及同一状态能选到的最好平均回报减去 Q 所选动作回报。三次同号的严格比较只是敏感性检查，不是独立样本。不得据数值绝对大小跨 critic 对比。新旧 SAC 训练目标还含后续动作熵；各 critic 原先针对各自策略，而此次固定 SFT 续跑。此实验测共同 continuation 下的决策相关性，不直接测各自策略 soft-Q 绝对值的无偏程度。

从项目根目录运行，创建新的输出目录：

```bash
OMP_NUM_THREADS=2 work/envs/gr00t_rl/bin/python -u projects/groot_long/branch_q.py projects/groot_long/sft_hard_v1/branch_q_v1
work/envs/gr00t_rl/bin/python projects/groot_long/branch_q_report.py projects/groot_long/sft_hard_v1/branch_q_v1
```
