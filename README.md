# WAMPT · GR00T 残差后训练

在 LIBERO-Long 双物体入篮任务上冻结 GR00T N1.5 SFT-Long，每五个控制步由 SAC 或 TD3 对 XYZ 动作施加修正。仓库包含任务筛选、在线采集的离策略训练、原始训练 replay、部署权重，以及从相同状态分支检验 critic 排序的实验。

| 策略 | 成功 / 30 | 部署权重（与共同的 GR00T 基座一起加载） |
|---|---:|---|
| 原始 SFT | 10 | [官方 SFT-Long 固定版本](https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Long/tree/2e6892a170c0220de88b2580d13df84e413c13b7) |
| 原 SAC | 13 | [sac_40.zip](weights/sac/sac_40.zip) |
| TD3 | 27 | [td3_40.zip](weights/td3/td3_40.zip) |
| 改进 SAC | 29 | [online_120.zip](weights/sac_repair_v1/online_120.zip) |

15 个布局各用两个推理种子，只有一个训练种子。改进 SAC 相对原 SAC 为 16 胜 0 负，相对 TD3 为 2 胜 0 负。布局在此前实验中用过，改进 SAC 增加了训练交互。将未来策略和熵项对齐后，改进 SAC 的 critic 在 96 个动作对里排对了 46 个。详见[训练结果](evidence/sft_hard_v1/sac_repair_v1/RESULTS.md)、[逐状态续跑结果](evidence/sft_hard_v1/branch_q_own_v1/RESULTS.md)和[示例视频](evidence/sft_hard_v1/sac_repair_v1/test-residual-41035.mp4)。

## 同一交互预算下的 PPO 与 SAC

在相同的 GR00T-SFT-Long 上，两个残差策略都只修正 XYZ，训练交互预算均为 20,162 个动作块。60 个固定测试布局上，原始 SFT 成功 19 次，验证选中的 PPO（第 12 轮）成功 18 次，用满预算的 PPO（第 17 轮）成功 38 次，改进 SAC 成功 55 次。对比两个末轮策略，19 个布局只有 SAC 成功，2 个只有 PPO 成功；差值为 +28.3 个百分点，按布局配对 bootstrap 的 95% 区间为 [+15.0,+41.7]。每种方法只跑了一个训练种子，SAC 做过更多调参。第 12、17 轮在 10 个验证回合上同为 4 次成功，按预定规则选取较早的一轮。[逐项结果](evidence/paired_ppo_v1/results.json) · [复现步骤](projects/groot_long/PAIRED_PPO_REPRO.md)

## 在 CPU 上复核已记录的结果

从仓库根目录执行。两个报告脚本只需 Python 和 NumPy，读取实际 episode 或分支迹线，重新计算成功数、配对区间和 Q 排序。

```bash
(cd weights && sha256sum -c SHA256SUMS)
proj=projects/groot_long
ev=evidence/sft_hard_v1
python "$proj/sac_repair_report.py" "$ev"
python "$proj/branch_q_own_report.py" "$ev/branch_q_own_v1"
```

## 加载权重和评估

历史环境使用 Linux、Python 3.11、PyTorch 2.8.0+cu128、MuJoCo 3.3.7、robosuite 1.4.1、rlinf-libero 0.1.3、Gymnasium 1.4.0 和 NumPy 1.26.4。MuJoCo 需要可用的 EGL。已有兼容环境可直接使用。以下命令在新目录准备两个固定版本的代码依赖，使用 RLinf 的安装器。运行安装器前请先满足其系统依赖并配置 uv 和 CUDA。GR00T 源版本为 NVIDIA/Isaac-GR00T 的 `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`。

```bash
mkdir -p work/repos/dsrl work/envs
git clone https://github.com/RLinf/RLinf.git work/repos/RLinf
git -C work/repos/RLinf checkout --detach f11ed2277e086e843ffad155d7ab89ae878e77a5
git clone https://github.com/ajwagen/stable-baselines3-dsrl.git work/repos/dsrl/stable-baselines3
git -C work/repos/dsrl/stable-baselines3 checkout --detach 10e5d311b5c42fe571ff3eeb72f622ca616a149d
venv="$PWD/work/envs/gr00t_rl"
(cd work/repos/RLinf && UV_TORCH_BACKEND=cu128 bash requirements/install.sh embodied --model gr00t --env libero --venv "$venv" --torch 2.8.0 --no-root --no-apex --no-natten)
```

从仓库根目录下载共同基座。可用环境变量 `WAMPT_SFT_DIR` 指向已有文件，下载器核对固定 revision 及两个 shard 的 SHA256。训练和评估需要 LIBERO 的 BDDL、资源与初态文件已按该版本的路径配置。确认结果目录为新目录后运行四组配对评估。

```bash
py=work/envs/gr00t_rl/bin/python
proj=projects/groot_long
"$py" "$proj/sft_assets.py"
(cd weights && sha256sum -c SHA256SUMS)
OMP_NUM_THREADS=2 "$py" "$proj/offpolicy.py" runs/published --sft --evaluate-published \
  --penalty 0.1 --selection evidence/sft_hard_v1/selected_task.json
```

每组使用同一批 30 个初始条件，输出 `runs/published/episodes.jsonl`、`results.json` 和样例视频。权重文件为 Stable-Baselines3 原始 zip，分别用 SAC.load 或 TD3.load 读取。冻结基座用 RLinf 的 GR00T 加载器读取。

## 重新训练与分支实验

[任务筛选与 TD3](projects/groot_long/SFT_HARD_REPRO.md) → [原 SAC](projects/groot_long/SAC_REPRO.md) → [改进 SAC](projects/groot_long/SAC_REPAIR_REPRO.md) → [SFT 续跑](projects/groot_long/BRANCH_Q_REPRO.md) → [自身策略续跑与熵](projects/groot_long/BRANCH_Q_OWN_REPRO.md)。各阶段配方均从仓库根目录执行。训练写入 projects/groot_long/sft_hard_v1，已发布的历史记录单独存放在 evidence 中。

如只重跑改进 SAC，可以直接从首次实验保存的 replay 开始。在干净的运行目录准备筛选记录及原 SAC 的训练日志后，执行改进 SAC 文档里的训练命令。复核历史成绩时直接使用上面的 CPU 命令。

```bash
p=projects/groot_long/sft_hard_v1
e=evidence/sft_hard_v1
mkdir -p "$p/sac"
cp "$e/selected_task.json" "$p/"
cp "$e/sac/episodes.jsonl" "$p/sac/"
gzip -dc "$e/sac/replay.pkl.gz" > "$p/sac/replay.pkl"
(cd "$p/sac" && echo '4a65984a2d0ec6b4ad3ff1da56e10ad6a682bb0a8881cfe45e17f760c3739537  replay.pkl' | sha256sum -c -)
```

如只重跑两轮分支，在新的运行目录将 weights 下三个子目录按原层级复制到 projects/groot_long/sft_hard_v1，并从 evidence/sft_hard_v1 复制 selected_task.json，再按两篇分支文档顺序启动。发布记录中将历史权重路径写成相对路径，并同步更新子实验对父协议的校验；权重、奖励和逐步迹线来自原始实验。
