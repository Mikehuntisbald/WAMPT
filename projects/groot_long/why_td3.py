"""Post-hoc diagnostic: shared-state probes and norm/direction swaps on validation only."""
from offpolicy import HERE, REPO, benchmark, OffScreenRenderEnv, OmegaConf, get_model, get_libero_image, get_libero_wrist_image, quat2axisangle, SAC, TD3, SCALE
from sft_assets import DIR
import json
import random
from collections import Counter
import torch
import numpy as np

torch.set_num_threads(2)
root=HERE/'sft_hard_v1'
out=root/'why_td3';out.mkdir(exist_ok=False)
def load_rows(name):return [json.loads(l) for l in (root/name/'episodes.jsonl').read_text().splitlines()]
rows={k:load_rows(k) for k in ('td3','sac')}
agents={k:cls.load(root/k/f'{k}_40.zip',device='cpu') for k,cls in [('td3',TD3),('sac',SAC)]}
common={}
for src in agents:
    prefix=[r for r in rows[src] if r['stage']=='train'][:40]
    n=prefix[-1]['replay']
    agents[src].load_replay_buffer(root/src/'replay.pkl')
    buf=agents[src].replay_buffer;assert not buf.full and buf.size()>=n
    obs=buf.observations[:n,0]
    acts={k:a.predict(obs,deterministic=True)[0] for k,a in agents.items()}
    norms={k:np.linalg.norm(v,axis=1) for k,v in acts.items()}
    qs={}
    with torch.no_grad():
        for k,a in agents.items():
            def q(action):
                return torch.cat(a.critic(torch.from_numpy(obs),torch.from_numpy(action)),dim=1).min(1).values.numpy()
            gap=q(acts['td3'])-q(acts['sac'])
            qs[k]=float((gap>0).mean())
    common[src]=dict(n=n,successes_first40=sum(r['success'] for r in prefix),positive_reward_chunks=int((buf.rewards[:n]>0).sum()),mean_norm={k:float(v.mean()) for k,v in norms.items()},mean_cosine=float(np.mean((acts['td3']*acts['sac']).sum(1)/np.maximum(norms['sac']*norms['td3'],1e-12))),critic_fraction_favoring_td3=qs)
keys=lambda r:(r['trial'],r['seed'])
test={k:{keys(r):r for r in vals if r['stage']=='test' and r['arm']=='residual'} for k,vals in rows.items()}
wins=[k for k in test['td3'] if test['td3'][k]['success'] and not test['sac'][k]['success']]
losses=[k for k in test['td3'] if not test['td3'][k]['success'] and test['sac'][k]['success']]
summary=dict(shared_probes=common,winning_layouts=sorted({t for t,s in wins}),td3_only=len(wins),sac_only=len(losses),sac_final_among_td3_only=dict(Counter(str(test['sac'][k]['final']) for k in wins)))
(out/'shared.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
suite=benchmark.get_benchmark_dict()['libero_10'](); idx=7; states=suite.get_task_init_states(idx)
env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(idx),camera_heights=256,camera_widths=256)
cfg=OmegaConf.load(REPO/'examples/embodiment/config/model/gr00t.yaml');cfg.model_path=str(DIR);cfg.add_value_head=False
base=get_model(cfg).cuda();base.eval();base.requires_grad_(False)
conditions=('td3','sac','td3_direction_sac_norm','sac_direction_td3_norm')
def episode(t,condition):
    seed=30000+t;np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);env.seed(seed);env.reset();obs=env.set_init_state(states[t])
    for _ in range(15):obs,*_=env.step(np.r_[np.zeros(6),-1.])
    steps=0;clip_n=chunks=0
    while steps<512:
        rgb=get_libero_image(obs).copy();wrist=get_libero_wrist_image(obs).copy()
        state=np.r_[obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']]
        with torch.inference_mode():
            chunk,_=base.predict_action_batch(dict(main_images=torch.from_numpy(rgb)[None],wrist_images=torch.from_numpy(wrist)[None],states=torch.from_numpy(state)[None],task_descriptions=[suite.get_task(idx).language]),mode='eval')
        act=chunk[0].cpu().numpy().copy()
        pool=lambda im:im.reshape(8,32,8,32,3).mean((1,3)).ravel()/127.5-1
        f=np.r_[state,act.ravel(),pool(rgb),pool(wrist)].astype('float32')
        u={k:a.predict(f,deterministic=True)[0] for k,a in agents.items()}
        if condition in agents:r=u[condition]
        else:
            direction,other=condition.split('_direction_');other=other.removesuffix('_norm')
            r=u[direction]*np.linalg.norm(u[other])/max(np.linalg.norm(u[direction]),1e-12)
            clip_n+=int(np.abs(r).max()>1);r=r.clip(-1,1)
        chunks+=1;act[:,:3]+=SCALE*r
        for a in act:
            obs,_,done,_=env.step(a);steps+=1
            if done or steps==512:break
        success=bool(env.check_success())
        if done or success:break
    final=[bool(env.env._eval_predicate(g)) for g in env.env.parsed_problem['goal_state']]
    if condition in agents:
        ref=next(r for r in rows[condition] if r['stage']=='val_40' and r['trial']==t)
        assert (steps,success,final)==(ref['steps'],ref['success'],ref['final']),(t,condition,steps,ref)
    return dict(trial=t,seed=seed,condition=condition,steps=steps,success=success,final=final,clipped=clip_n,chunks=chunks)
for t in range(30,35):
    for c in conditions:
        row=episode(t,c)
        with (out/'ablation.jsonl').open('a') as log: print(json.dumps(row),file=log)
    print('validation layout finished',t,flush=True)
env.close()
