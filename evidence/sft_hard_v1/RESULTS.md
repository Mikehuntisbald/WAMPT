# LIBERO-Long 筛选与离策略后训练

任务：put both the alphabet soup and the cream cheese box in the basket。探索初筛 6/10，换策略噪声 4/10；布局 [4, 8] 在同种子复跑、另一种子确认及第三种子复跑时均失败且末态目标标志一致。这是失败现象复现，不是根因诊断。

| 策略 | 测试成功 | 布局 | 每布局完整评估 |
|---|---:|---:|---:|
| RLinf/RLinf-Gr00t-SFT-Long | 10/30 | 15 | 2 |
| GR00T + 验证选中的残差 TD3 | 27/30 | 同上 | 同上 |

配对胜/负 17/0，净差 +56.7 个百分点，按布局配对 bootstrap 95% 区间 [40.0, 73.3]。每臂一个训练配置，区间不含跨训练种子不确定性。这是单项任务，不能代表整个套件。

训练 120 个真实仿真回合，54294 个控制步、10925 条动作块 transition，critic 更新 10670 次、actor 更新 5335 次（总训练预算）；第 40 回合权重由验证选定，三个验证点（各 5 个布局）分别成功 [4, 0, 0] 次。原模型不改权重，离策略 TD3 修正 XYZ，奖励含已满足谓词数的变化、双目标/全目标成功奖励及 0.1 倍归一化修正均方代价。测试成功仅由原任务的官方全目标判定决定。训练/验证/测试初态哈希不相交，训练 20 个布局，模型选择 5 个，独立测试 15 个，另 10 个布局只筛选。公开模型原训练与官方布局是否重合未知。

证据：preregister.json、selected_task.json、各 candidate_*/gate.json 和 episodes.jsonl、td3/selection.json、td3/results.json、verified.json 和 td3/*.mp4。运行：从工作区用 work/envs/gr00t_rl/bin/python projects/groot_long/screen_then_train.py （需要全新输出目录）；事后核验：work/envs/gr00t_rl/bin/python projects/groot_long/hard_report.py。

同任务在历史 RL-Step300 实验中评估过，因此这不是项目从未见过的测试布局。测试基线是本轮实际重新执行的 SFT-Long。完整复现步骤见 ../SFT_HARD_REPRO.md；运行筛选时需添加 --sft，运行核验时指定 sft_hard_v1。末态子目标分布见 final_goal_counts.json，仅描述现象。
