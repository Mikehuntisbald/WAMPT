# 从 SFT-Long 开始的独立 LIBERO-Long 实验

本机 GR00T 环境、RLinf 和 Stable-Baselines3 版本沿用 README.md / OFFPOLICY.md。使用公开的 `RLinf/RLinf-Gr00t-SFT-Long`，revision `2e6892a170c0220de88b2580d13df84e413c13b7`。只取两片 safetensors、索引、配置与训练数据归一化元数据。下载到数据盘独立目录，不复用以前的 Spatial 或 RL 权重。sft_assets.py 对两片核对官方 LFS SHA256，对小文件核对 Git blob；offpolicy.py 从每片抽查三个具名参数加载后的数值。初筛、训练采样和最终比较都读取同一权重。

从工作区根目录复现（需要已有 gr00t_rl 环境及 LIBERO 资产；screen_then_train.py 拒绝覆盖已有运行）：

```bash
work/envs/gr00t_rl/bin/python projects/groot_long/sft_assets.py
OMP_NUM_THREADS=2 work/envs/gr00t_rl/bin/python -u projects/groot_long/screen_then_train.py --sft
work/envs/gr00t_rl/bin/python projects/groot_long/hard_report.py sft_hard_v1
```

与原先筛选相同的顺序和门槛。以前已用 RL-Step300 训练过的第 7 项排在最后，仅当之前都不资格时才用新 SFT 筛。全套任务都未替换指令、改变场景或缩短时限。布局 0–9 只筛选；各任务其余布局事先分为训练 10–29、验证 30–34 和测试 35–49；入选时绑定完整初态哈希和测试种子。只按验证结果选第 40/80/120 回合候选，不从测试反馈继续调参。

第 7 项最终入选。它的测试布局虽未参与本次 SFT 残差训练、筛选或模型选择，但在历史 **RL-Step300** 实验中评估过，不能称为整个项目从未见过的新场景。此前的 30/30 与 29/30 不能作为本实验 SFT 起点或结果；两臂均须重新运行。

算法是冻结 SFT 加 TD3 残差策略。残差每 5 步读取两路图像缩略图、机器人状态、SFT 动作，修正 XYZ，最大 ±0.12 个控制器动作单位；不改夹爪与旋转。训练奖励用官方目标谓词完成数的变化、终端成功奖励以及 0.1 倍修正均方代价；测试只用官方成功判定。训练期实际采集，再从 replay 随机采样，即在线采集的离策略 RL。最终只解释同任务、同布局、同推理种子的逐对胜负；15 个布局每个 2 个种子，不计为 30 个独立场景。公开 SFT 训练覆盖的布局未知；仅保证本地拆分不交叉。
