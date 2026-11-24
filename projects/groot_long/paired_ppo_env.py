"""Frozen GR00T and LIBERO rollout routines used in the paired PPO experiment."""
import hashlib
import json
import imageio.v2 as imageio
import numpy as np
import torch

from branch_q import (HERE, REPO, DIR, rng, sha, benchmark, OffScreenRenderEnv, OmegaConf,
                      get_model, get_libero_image, get_libero_wrist_image, quat2axisangle, SCALE)
from offpolicy import write

REPOSITORY = HERE.parents[1]
BANK = REPOSITORY / 'evidence/paired_ppo_v1'
LOCK = json.loads((REPOSITORY/'evidence/sft_hard_v1/selected_task.json').read_text())
CFG = json.loads((BANK/'protocol.json').read_text())
VALIDATION_OFFSETS = (30000, 33000)

def load_model():
    cfg = OmegaConf.load(REPO / 'examples/embodiment/config/model/gr00t.yaml')
    cfg.model_path = str(DIR)
    cfg.add_value_head = False
    model = get_model(cfg).cuda()
    model.eval()
    return model.requires_grad_(False)


def images_state(obs):
    return (get_libero_image(obs).copy(), get_libero_wrist_image(obs).copy(),
            np.r_[obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat'].copy()),
                  obs['robot0_gripper_qpos']].astype('float32'))


def feed(rgb, wrist, state):
    return dict(main_images=torch.as_tensor(rgb), wrist_images=torch.as_tensor(wrist),
                states=torch.as_tensor(state), task_descriptions=[LOCK['task']] * len(rgb))


@torch.inference_mode()
def predict(model, observation, seed):
    rng(seed)
    batch = feed(*(a[None] for a in observation))
    action, _ = model.predict_action_batch(batch, mode='eval')
    action = action[0].cpu().numpy()
    assert action.shape == (5, 7) and np.isfinite(action).all()
    return action


def residual_features(observation, action):
    rgb, wrist, state = observation
    def pool(im):
        return im.reshape(8,32,8,32,3).mean((1,3)).ravel()/127.5-1
    return np.r_[state,action.ravel(),pool(rgb),pool(wrist)].astype('float32')


def correct(agent, observation, action):
    feature = residual_features(observation, action)
    residual = agent.predict(feature, deterministic=True)[0]
    label = action.copy()
    label[:, :3] += SCALE * residual
    return label


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def setup():
    suite=benchmark.get_benchmark_dict()['libero_10']()
    idx=LOCK['idx']
    assert suite.get_task(idx).language==LOCK['task']
    env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(idx),camera_heights=256,camera_widths=256)
    return env,suite.get_task_init_states(idx)

def snapshot(env,obs):
    return [sha(x) for x in (env.get_sim_state(),env.sim.data.qacc_warmstart,env.sim.data.ctrl,
                              obs['agentview_image'],obs['robot0_eye_in_hand_image'])]

def settle(env,state,seed,rng_seed=None):
    rng(seed if rng_seed is None else rng_seed)
    env.seed(seed)
    env.reset()
    obs=env.set_init_state(state)
    for _ in range(15):obs,*_=env.step(np.r_[np.zeros(6),-1.])
    return obs


def rollout(arms,validation,*,agents,result_dir):
    torch.set_num_threads(2);env,states=setup();base=load_model()
    if validation:
        cases=[(t+o,states[t]) for o in VALIDATION_OFFSETS for t in LOCK['validation']]
    else:
        records=json.loads((BANK/'initial/manifest.json').read_text())
        cases=[]
        for rec in records:
            s=rec['seed'];state=np.load(BANK/'initial'/f'{s}.npy')
            assert sha(state)==rec['hash']
            cases.append((s,state))
    for i,(seed,state) in enumerate(cases):
        for arm in (arms if i%2==0 else arms[::-1]):
            dest=result_dir/('val' if validation else 'test')/arm;dest.mkdir(exist_ok=True,parents=True)
            target=dest/f'{seed}.json'
            if target.exists():continue
            obs=settle(env,state,seed)
            settled=snapshot(env,obs)
            pos=0
            video=imageio.get_writer(dest/f'{seed}.mp4',fps=20) if not validation and i==0 else None
            while pos<512:
                view=images_state(obs)
                # Match historical validation noise/reset (one RNG stream per episode)
                # by explicitly calling the standard forward without re-seeding every chunk there.
                if validation:
                    with torch.inference_mode():
                        act,_=base.predict_action_batch(feed(*(x[None] for x in view)),mode='eval')
                        actions=act[0].numpy()
                else:
                    actions=predict(base,view,seed*1000+pos)
                if agents[arm] is not None:actions=correct(agents[arm],view,actions)
                for a in actions:
                    obs,_,done,_=env.step(a);pos+=1
                    if video:video.append_data(get_libero_image(obs).copy())
                    if done or pos==512:break
                if done or env.check_success():break
            if video:video.close()
            flags=[bool(env.env._eval_predicate(g)) for g in env.env.parsed_problem['goal_state']]
            assert all(flags)==bool(env.check_success())
            write(target,dict(seed=seed,initial=sha(state),settled=settled,success=all(flags),goals=flags,steps=pos))
        if (i+1)%10==0:print(i+1,arms,flush=True)
    env.close()
