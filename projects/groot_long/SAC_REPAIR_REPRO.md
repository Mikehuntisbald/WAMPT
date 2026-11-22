# SAC 复现实验：先学 Q，再动策略

目标是本次锁定条件下的 **27/30** 点估计；达到同一分数也不是统计等效或跨场景复现。此前原 SAC 为 13/30，TD3 为 27/30。沿用同一个 SFT、XYZ 动作范围、奖励和 30 对测试条件，不加载 TD3 模型、轨迹或动作标签。项目已看过这份测试集；它适合历史配对，不是全新的盲测。

**分析依据。** 第 40 回合两个方法用来更新 critic 的次数接近，但 SAC actor 更新次数约为两倍。之前真实闭环互换方向/幅度的结果说明只放大 SAC 的确定性输出不够。先检验 early critic 和 actor 更新频率；低初始熵温度是单独的后续对照。

**设计。** 直接从上轮 SAC 的 120 个训练回合取 replay，不含验证或测试。四个独立初始化的 SAC 都在相同的 10,359 条记录上执行 4,096 次原生 SB3 critic 更新，使用相同的 minibatch 和采样噪声。对照组同步更新；预热组前 512 步冻结 actor 和温度；延迟组预热后每两步才更新 actor 与温度；第四组另将自动学习的 alpha 初值从 0.1 调至 0.01。保持原生 SAC 的随机策略、soft Bellman target、双 Q 和 target network 更新。脚本检查参数摘要与 Adam 步数；冻结时真实 optimizer.step 被跳过，不只是给学习率置零。

离线阶段在 2,048 / 4,096 步验证，按 5 个固定验证布局 × 两组固定噪声选一个起点。随后恢复该 SAC 的 optimizer、temperature 和训练 replay，再从 20 个训练布局新采 120 个在线回合。离线候选及在线第 40/80/120 回合均参与最终验证选择，仅对胜出者运行一次最终测试。排名按成功数、末态目标数、先出现者；不根据测试成绩再搜索。固定回合数会有不同的交互步数与更新预算；不能将相对上轮的收益全部归因于预热。这是一个训练种子的实验。

**复现。** 依赖已下载的 SFT-Long、上轮 SAC 的训练 replay（见运行协议的 SHA256），与原 TD3 及 SAC 历史配对档案。创建新输出目录；核验脚本默认读取 sac_repair_v1。

```bash
OMP_NUM_THREADS=2 work/envs/gr00t_rl/bin/python -u projects/groot_long/offpolicy.py \
  projects/groot_long/sft_hard_v1/sac_repair_v1 --repair --sft --algorithm sac --penalty 0.1 \
  --selection projects/groot_long/sft_hard_v1/selected_task.json
work/envs/gr00t_rl/bin/python projects/groot_long/sac_repair_report.py
```

运行目录内 repair_protocol.json 优先于通用 protocol.json 描述两阶段预算。val_scores.json 是逐候选验证与真实优化器步数，offline_selection.json / final_selection.json 在对应阶段结束时写入，episodes.jsonl 是闭环记录，verified.json 是事后独立重算结果。部署只需要选中 SAC 的压缩包和冻结 SFT，不需要 replay 或 TD3。

## 在纯 SAC 验证退化后增加的两组（先于本轮最终测试）

- `--anchored`，独立输出 sac_anchor_v1：仅原 SAC 的训练集，按完整回合折扣回报预拟合双 Q；用其中成功回合的执行动作拟合随机 SAC actor 的均值，作为参考而非逐步专家真值。1024 个预训练步；后续 SAC actor 梯度另加权重 0.5 的参考 MSE。无需教师，也不构成标准纯 SAC。
- `--transfer`，独立输出 sac_transfer_v1：让既有 td3_40 只在原 SAC 训练 replay 的 10,359 个状态提供确定性动作，离线一次性缓存标签，随后释放教师。随机初始化 SAC，无网络权重复制；8,192 个预拟合步，SAC 阶段参考 MSE 权重 10。它是蒸馏辅助 SAC，不是 SAC 自主发现了 TD3 的策略。
- 两者各完整执行 4096 步离策略 SAC 与 120 次在线训练回合；选取验证最佳的 SAC 检查点（纯 BC/蒸馏预拟合点只作诊断，不可入选）。从第 4096 步继续在线的路径事先固定。组间没有等算力或等先验；两组都会报告。命令同上，将 --repair 替换为相应选项并指定新目录，核验分别运行 sac_anchor_report.py 或 sac_anchor_report.py sac_transfer_v1。
