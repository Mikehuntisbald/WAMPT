# GR00T N1.5 / LIBERO-Long: two objects into one basket

This is an autonomous inference run of RLinf's published Long Step300 policy. No local training, demonstrations, human corrections, or scripted subtask switches. Give the full instruction at every policy call. Use upstream LIBERO goal predicates for measurement only. A transient first insertion does not count as task success: both objects must be inside simultaneously.

From project root, with the existing GR00T/CUDA environment and LIBERO assets configured:

```bash
cd projects/groot_long
base=https://huggingface.co/RLinf/RLinf-Gr00t-RL-Long-Step300/resolve/2c3d290c05741051a2ff12c750de460503c077f1
mkdir -p model/experiment_cfg
for f in config.json experiment_cfg/metadata.json model.safetensors; do
  test -f "model/$f" || { curl --fail --retry 3 -L "$base/$f" -o "model/$f.part" && mv "model/$f.part" "model/$f"; } || exit 1
done
../../work/envs/gr00t_rl/bin/python -u run.py
```

RLinf checkout: f11ed2277e086e843ffad155d7ab89ae878e77a5; local distributed-worker edits are not used by this single-process evaluator. Load the official N1.5 policy builder and observation/action transforms. BF16, action-only evaluation, four flow denoising steps per 5 × 7 action chunk, fresh current front/wrist views and robot proprioception every chunk. This tests multi-stage task execution; it does not establish long-context memory. No generated video or world predictions are used. Trial IDs 0–4 are fixed before running; 15 open-gripper settling steps, then up to 512 policy control steps at 20 Hz. Video playback uses the same frame rate. Stop on official success or episode end. Each NPZ stores executed actions and the simulator's goal predicate values at every step.
