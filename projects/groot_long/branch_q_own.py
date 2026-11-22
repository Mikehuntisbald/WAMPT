"""Own-policy MC return for the exact cached first block and fork used in branch_q."""
import argparse
import json
import hashlib
import os
import torch
import numpy as np
from branch_q import (ROOT,WEIGHTS,ARM,sha,rng,dump,HERE,REPO,benchmark,OffScreenRenderEnv,
                      OmegaConf,get_model,get_libero_image,get_libero_wrist_image,quat2axisangle,
                      SCALE,TD3,SAC,DIR)

OLD=ROOT/'branch_q_v1'
def main(out,layouts=None):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    old=json.loads((OLD/'protocol.json').read_text())
    agents={k:(TD3 if k=='td3' else SAC).load(w,device='cpu') for k,w in WEIGHTS.items()}
    for k,w in WEIGHTS.items():assert hashlib.sha256(w.read_bytes()).hexdigest()==old['checkpoints'][k][1]
    alpha={k:(float(a.log_ent_coef.detach().exp()) if k!='td3' else 0.) for k,a in agents.items()}
    protocol=dict(parent_protocol_sha256=hashlib.sha256((OLD/'protocol.json').read_bytes()).hexdigest(),layouts=old['layouts'],prefix_steps=old['prefix_steps'],repeats=old['repeats'],arms=ARM,alpha=alpha,follow=list(agents),discount=.99,cost=.1,horizon=512,td3='deployed deterministic actor, no training target smoothing',
                  primary='min online twin-Q versus mean own-policy soft return; compare same critic and same state/arm pairs with previous shaped SFT MC; paired layout bootstrap, 20000 draws',
                  sac_entropy='-alpha*joint squashed residual logpi once per future chunk, from chunk 1, no fixed first action entropy',
                  control='duplicate zero first action at repeat 0 for each continuation and state')
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==json.loads(json.dumps(protocol))
    else:dump(out/'protocol.json',protocol)
    suite=benchmark.get_benchmark_dict()['libero_10']();idx=7;states=suite.get_task_init_states(idx)
    lock=json.loads((ROOT/'selected_task.json').read_text())
    env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(idx),camera_heights=256,camera_widths=256)
    goals=env.env.parsed_problem['goal_state'];instruction=suite.get_task(idx).language
    cfg=OmegaConf.load(REPO/'examples/embodiment/config/model/gr00t.yaml');cfg.model_path=str(DIR);cfg.add_value_head=False
    base=get_model(cfg).cuda();base.eval();base.requires_grad_(False)
    def flags():return [bool(env.env._eval_predicate(g)) for g in goals]
    def reset(t):
        rng(711000+t);env.seed(711000+t);env.reset();obs=env.set_init_state(states[t])
        for _ in range(15):obs,*_=env.step(np.r_[np.zeros(6),-1.])
        return obs
    def propose(obs,seed):
        rng(seed);rgb=get_libero_image(obs).copy();wrist=get_libero_wrist_image(obs).copy()
        state=np.r_[obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']]
        chunk,_=base.predict_action_batch(dict(main_images=torch.from_numpy(rgb)[None],wrist_images=torch.from_numpy(wrist)[None],states=torch.from_numpy(state)[None],task_descriptions=[instruction]),mode='eval')
        act=chunk[0].cpu().numpy().copy()
        def pool(im):return im.reshape(8,32,8,32,3).mean((1,3)).ravel()/127.5-1
        return np.r_[state,act.ravel(),pool(rgb),pool(wrist)].astype('float32'),act
    first_physics={}
    for t in old['layouts'] if layouts is None else layouts:
        assert t in old['layouts']
        assert sha(states[t])==lock['hashes'][t]
        for step in old['prefix_steps']:
            sid=f'{t}_{step}';meta=json.loads((OLD/(sid+'.json')).read_text())
            with np.load(OLD/(sid+'.npz')) as snap:prefix,f,first,acts=[snap[n] for n in ('prefix','feature','first','residuals')]
            assert meta['order']==list(ARM) and len(prefix)==step
            for name,agent in agents.items():
                path=out/f'{sid}-{name}.json'
                if path.exists():continue
                q=torch.cat(agent.critic(torch.from_numpy(np.repeat(f[None],4,axis=0)),torch.from_numpy(acts)),dim=1).numpy()
                assert np.allclose(q,meta['q_heads'][name],rtol=0,atol=1e-6)
                assert np.array_equal(agent.predict(f,deterministic=True)[0],acts[ARM.index(name)])
                rows=[]
                for rep in range(old['repeats']):
                    for arm in (('zero','zero_repeat',*ARM[1:]) if rep==0 else ARM):
                        obs=reset(t)
                        for a in prefix:obs,_,done,_=env.step(a);assert not done
                        fp=[sha(env.get_sim_state()),sha(env.sim.data.qacc_warmstart),sha(env.sim.data.ctrl),sha(obs['agentview_image']),sha(obs['robot0_eye_in_hand_image'])]
                        assert fp==meta['hashes'] and flags()==meta['initial']
                        prior=meta['initial'];pos=step;j=0;trace=[];key='zero' if arm=='zero_repeat' else arm
                        while pos<512:
                            seed=9000000+t*100000+step*500+rep*120+j
                            lp=0.
                            if j==0:chunk=first.copy();res=acts[ARM.index(key)]
                            else:
                                feat,chunk=propose(obs,seed)
                                if name=='td3':res=agent.predict(feat,deterministic=True)[0]
                                else:
                                    torch.random.default_generator.manual_seed(100000000+seed)
                                    action,logp=agent.actor.action_log_prob(torch.from_numpy(feat[None]))
                                    res=action[0].numpy();lp=float(logp.item())
                            assert np.isfinite(lp) and np.isfinite(res).all() and np.abs(res).max()<=1.
                            chunk[:,:3]+=SCALE*res
                            for a in chunk:
                                obs,_,done,_=env.step(a);pos+=1
                                if done or pos==512:break
                            cur=flags();success=bool(env.check_success());assert success==all(cur)
                            if j==0:
                                h=sha(env.get_sim_state());k=(sid,key)
                                assert first_physics.setdefault(k,h)==h
                            reward=sum(cur)-sum(prior)+2*int(success)
                            trace.append([pos,int(reward),float(.1*np.mean(res**2)),lp,cur]);prior=cur;j+=1
                            if done or success:break
                        rewards=np.array([v[1:4] for v in trace]);task,cost,logp_sum=.99**np.arange(j)@rewards
                        row=dict(state=sid,layout=t,step=step,follow=name,repeat=rep,arm=arm,end=pos,success=success,final=cur,trace=trace,
                                 first_physics=h,end_physics=sha(env.get_sim_state()),task=task,shaped=task-cost,entropy=-alpha[name]*logp_sum,soft=task-cost-alpha[name]*logp_sum)
                        if arm=='zero_repeat':assert all(row[n]==rows[0][n] for n in row if n!='arm')
                        rows.append(row)
                tmp=path.with_suffix('.tmp');dump(tmp,rows);os.replace(tmp,path)
                print('verified state/continuation',sid,name,flush=True)
    env.close()

if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--layouts',type=int,nargs='+')
    args=p.parse_args()
    with torch.inference_mode():main(args.out,args.layouts)
