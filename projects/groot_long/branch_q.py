"""First-action counterfactuals, full-horizon common continuation, no training."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from offpolicy import (HERE, REPO, benchmark, OffScreenRenderEnv, OmegaConf, get_model,
                       get_libero_image, get_libero_wrist_image, quat2axisangle, TD3, SAC, SCALE)
from sft_assets import DIR
import numpy as np
import torch

ARM=('zero','sac','new_sac','td3')
ROOT=HERE/'sft_hard_v1'
WEIGHTS={'sac':ROOT/'sac/sac_40.zip','new_sac':ROOT/'sac_repair_v1/online_120.zip','td3':ROOT/'td3/td3_40.zip'}
def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(f,x):
    f.write_text(json.dumps(x,indent=2))
def rng(s):
    np.random.seed(s);random.seed(s);torch.manual_seed(s)
def main(out):
    out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2)
    protocol=dict(layouts=list(range(10,18)),prefix_steps=[30,120],repeats=3,arms=ARM,
                  follow='frozen SFT + zero residual, receding 5-step chunks',horizon=512,
                  gamma=.99,penalty=.1,prefix='frozen SFT seed schedule, open-loop prefix replay from official reset',
                  metrics='statewise mean MC; pairwise match excluding return ties (abs <= 1e-6), top1 with tie credit, regret, task-only sensitivity; bootstrap by layout',
                  checkpoints={k:[str(v.relative_to(ROOT)),hashlib.sha256(v.read_bytes()).hexdigest()] for k,v in WEIGHTS.items()})
    dump(out/'protocol.json',protocol)
    suite=benchmark.get_benchmark_dict()['libero_10']();idx=7;states=suite.get_task_init_states(idx)
    lock=json.loads((ROOT/'selected_task.json').read_text());assert set(protocol['layouts'])<=set(lock['train'])
    env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(idx),camera_heights=256,camera_widths=256)
    goals=env.env.parsed_problem['goal_state'];instruction=suite.get_task(idx).language
    cfg=OmegaConf.load(REPO/'examples/embodiment/config/model/gr00t.yaml');cfg.model_path=str(DIR);cfg.add_value_head=False
    base=get_model(cfg).cuda();base.eval();base.requires_grad_(False)
    agents={k:(TD3 if k=='td3' else SAC).load(w,device='cpu') for k,w in WEIGHTS.items()}
    def flags():return [bool(env.env._eval_predicate(g)) for g in goals]
    def reset(t):
        rng(711000+t);env.seed(711000+t);env.reset();obs=env.set_init_state(states[t])
        for _ in range(15):obs,*_=env.step(np.r_[np.zeros(6),-1.])
        return obs
    def propose(obs,seed):
        rng(seed);rgb=get_libero_image(obs).copy();wrist=get_libero_wrist_image(obs).copy()
        state=np.r_[obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']]
        with torch.inference_mode():
            chunk,_=base.predict_action_batch(dict(main_images=torch.from_numpy(rgb)[None],wrist_images=torch.from_numpy(wrist)[None],states=torch.from_numpy(state)[None],task_descriptions=[instruction]),mode='eval')
        act=chunk[0].cpu().numpy().copy()
        def pool(im):return im.reshape(8,32,8,32,3).mean((1,3)).ravel()/127.5-1
        feature=np.r_[state,act.ravel(),pool(rgb),pool(wrist)].astype('float32');return feature,act
    def fingerprint(obs):
        # Include hidden controller-relevant dynamics and both policy camera observations.
        return [sha(env.get_sim_state()),sha(env.sim.data.qacc_warmstart),sha(env.sim.data.ctrl),sha(obs['agentview_image']),sha(obs['robot0_eye_in_hand_image'])]
    for t in protocol['layouts']:
        assert sha(states[t])==lock['hashes'][t]
        obs=reset(t);prefix=[];snapshots={}
        for step in range(0,max(protocol['prefix_steps'])+1,5):
            if step in protocol['prefix_steps']:
                snapshots[step]=(fingerprint(obs),flags(),*propose(obs,810000+t*1000+step))
            if step==max(protocol['prefix_steps']):break
            _,act=propose(obs,710000+t*1000+step)
            for a in act:
                obs,_,done,_=env.step(a);prefix.append(a.copy())
                if done:break
            if done:break
        for step,(fp,initial,f,first) in snapshots.items():
            act={k:a.predict(f,deterministic=True)[0] for k,a in agents.items()};act['zero']=np.zeros(3,np.float32)
            aa=torch.from_numpy(np.stack([act[k] for k in ARM]));ff=torch.from_numpy(np.repeat(f[None],4,axis=0))
            with torch.inference_mode():qs={k:torch.cat(a.critic(ff,aa),dim=1).numpy().tolist() for k,a in agents.items()}
            state_id=f'{t}_{step}'
            np.savez_compressed(out/(state_id+'.npz'),prefix=np.asarray(prefix[:step]),feature=f,first=first,residuals=aa.numpy())
            dump(out/(state_id+'.json'),dict(t=t,step=step,initial=initial,hashes=fp,q_heads=qs,order=ARM))
            for rep in range(protocol['repeats']):
                control=None
                for k in (('zero','zero_repeat','sac','new_sac','td3') if rep==0 else ARM):
                    obs=reset(t)
                    for a in prefix[:step]:obs,_,done,_=env.step(a);assert not done
                    assert fingerprint(obs)==fp and flags()==initial,(t,step,rep,k,'prefix mismatch')
                    pos=step;chunk_i=0;gain=0.;hist=[];prior=initial;rr=act['zero' if k=='zero_repeat' else k]
                    while pos<512:
                        if chunk_i==0:
                            chunk=first.copy();chunk[:,:3]+=SCALE*rr
                        else:
                            _,chunk=propose(obs,9000000+t*100000+step*500+rep*120+chunk_i)
                        for a in chunk:
                            obs,_,done,_=env.step(a);pos+=1
                            if done or pos==512:break
                        cur=flags();success=bool(env.check_success());assert success==all(cur)
                        r=sum(cur)-sum(prior)+2*int(success)
                        gain+=(.99**chunk_i)*r;hist.append((pos,cur));prior=cur;chunk_i+=1
                        if success or done:break
                    row=dict(state=state_id,layout=t,step=step,repeat=rep,arm=k,steps=pos-step,success=success,final=cur,task_return=gain,shaped_return=gain-.1*float(np.mean(rr**2)),trajectory_hash=sha(np.asarray([[p,*g] for p,g in hist])),end_physics=sha(env.get_sim_state()))
                    if k=='zero':control=row
                    if k=='zero_repeat':assert all(row[key]==control[key] for key in row if key!='arm'),(row,control)
                    with (out/'branches.jsonl').open('a') as log:print(json.dumps(row),file=log)
            print('finished state',state_id,flush=True)
    env.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);main(parser.parse_args().out)
