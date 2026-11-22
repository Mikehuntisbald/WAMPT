"""Evaluate one LIBERO-Long instruction on the first five official layouts."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "work/repos/RLinf"
sys.path.insert(0, str(REPO))
os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", TOKENIZERS_PARALLELISM="false")
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from rlinf.envs.sim.libero.utils import get_libero_image, get_libero_wrist_image, quat2axisangle
from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model


def main():
    torch.set_num_threads(2)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    idx, = [i for i in range(suite.n_tasks) if "both the alphabet soup and the cream cheese" in suite.get_task(i).language]
    instruction = suite.get_task(idx).language
    states = suite.get_task_init_states(idx)
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(idx), camera_heights=256, camera_widths=256)
    goals = env.env.parsed_problem["goal_state"]
    print(instruction, goals, flush=True)
    cfg = OmegaConf.load(REPO / "examples/embodiment/config/model/gr00t.yaml")
    cfg.model_path = str(HERE/"model")
    cfg.add_value_head = False
    with (HERE/"model/model.safetensors").open("rb") as source:
        assert hashlib.file_digest(source,"sha256").hexdigest()=="eed4991e325b906d2976e53903fe3b5dd34bdf8b23961ca605db04d02313a5df"
    policy = get_model(cfg).cuda()
    policy.eval()
    out = HERE/"rollouts"
    out.mkdir(exist_ok=True)
    rows = []
    for trial in range(5):
        seed = 20261120+trial
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); env.seed(seed)
        env.reset(); obs = env.set_init_state(states[trial])
        for _ in range(15):
            obs, _, _, _ = env.step(np.array([0.]*6+[-1.]))
        actions, flags = [], []
        with imageio.get_writer(out/f"trial{trial}.mp4", fps=20) as video:
            for step in range(512):
                if step % 5 == 0:
                    state = np.r_[obs["robot0_eef_pos"],quat2axisangle(obs["robot0_eef_quat"].copy()),obs["robot0_gripper_qpos"]]
                    feed = dict(main_images=torch.from_numpy(get_libero_image(obs).copy())[None], wrist_images=torch.from_numpy(get_libero_wrist_image(obs).copy())[None],states=torch.from_numpy(state)[None],task_descriptions=[instruction])
                    with torch.inference_mode():
                        chunk,_ = policy.predict_action_batch(feed,mode="eval")
                    chunk = chunk.cpu().numpy()
                    assert chunk.shape == (1,5,7) and np.isfinite(chunk).all()
                action = chunk[0,step%5]
                obs, _, done, _ = env.step(action)
                actions.append(action.copy()); flags.append([bool(env.env._eval_predicate(g)) for g in goals])
                video.append_data(obs["agentview_image"][::-1,::-1].copy())
                success = bool(env.check_success())
                assert success == all(flags[-1])
                if success or done: break
        np.savez_compressed(out/f"trial{trial}.npz", actions=actions, predicates=flags)
        row=dict(trial=trial,seed=seed,initial_state_sha256=hashlib.sha256(np.asarray(states[trial]).tobytes()).hexdigest(),steps=len(flags),success=success,final=flags[-1],first_true_steps=[next((i+1 for i,f in enumerate(flags) if f[j]),None) for j in range(len(goals))])
        rows.append(row); print(row,flush=True)
        (out/"results.json").write_text(json.dumps(dict(task=instruction,goals=goals,rows=rows),indent=2))
    env.close()


if __name__=="__main__":
    main()
