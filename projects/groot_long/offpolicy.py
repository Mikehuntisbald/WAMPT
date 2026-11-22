"""Residual TD3 on a frozen GR00T. One replay transition per policy chunk."""
from run import HERE, REPO, OffScreenRenderEnv, benchmark, get_model, get_libero_image, get_libero_wrist_image, quat2axisangle, OmegaConf
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
sys.path.insert(0,str(HERE.parents[1]/'work/repos/dsrl/stable-baselines3'))
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.logger import configure

TRAIN=list(range(5,25)); VAL=list(range(25,35)); TEST=list(range(35,50))
SEED=1307
SCALE=.12

class Spaces(gym.Env):
    action_space=gym.spaces.Box(-1.,1.,(3,),np.float32)
    observation_space=gym.spaces.Box(-np.inf,np.inf,(8+35+2*8*8*3,),np.float32)


def write(path, data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def main(out, penalty=0., screening=None, selection=None, sft=False, algorithm='td3', repair=False, anchored=False, transfer=False, confirm=False, evaluate_published=False):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    suite=benchmark.get_benchmark_dict()['libero_10']()
    global TRAIN, VAL, TEST
    if screening is not None or selection is not None:
        TRAIN=list(range(10,30)); VAL=list(range(30,35)); TEST=list(range(35,50))
        idx=screening if screening is not None else selection['idx']
    else:
        idx,=[i for i in range(suite.n_tasks) if 'both the alphabet soup and the cream cheese' in suite.get_task(i).language]
    states=suite.get_task_init_states(idx)
    hashes=[hashlib.sha256(np.asarray(s).tobytes()).hexdigest() for s in states]
    assert len(set(hashes))==len(states)==50
    protocol=dict(train=TRAIN,validation=VAL,test=TEST,historical_excluded=list(range(5)),hashes=hashes,seed=SEED,episodes=120,checkpoints=[40,80,120],select='validation successes, then final predicates, then earlier checkpoint',scale=SCALE,reward='delta in number of satisfied goals + 2 * success; episodic 512-step horizon',batch=128,lr=.0003,discount=.99,chunk=5,exploration_std=.22,validation_seed_offset=30000,test_seed_offsets=[41000,51000])
    protocol['chunk_residual_penalty']=penalty
    protocol['algorithm']=algorithm
    if algorithm=='sac':
        protocol.update(exploration_std=None,initial_pre_tanh_std=.22,sac_entropy='auto_0.1',target_entropy=-3,external_action_noise=False)
    if sft:
        from sft_assets import DIR, REV, MODEL, SHARDS
        assets=json.loads((DIR/'verified.json').read_text())
        assert (assets['model'],assets['revision'])==(MODEL,REV)
        assert all(assets['files'][k]==v for k,v in SHARDS.items())
        protocol['base']=assets
        if selection is not None:assert selection['base']==assets
    if screening is not None or selection is not None:
        protocol.update(task=suite.get_task(idx).language,task_id=idx,historical_excluded=list(range(10)))
    if selection is not None:
        assert selection['hashes']==hashes and selection['train']==TRAIN and selection['validation']==VAL and selection['test']==TEST
    write(out/'protocol.json',protocol)
    env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(idx),camera_heights=256,camera_widths=256)
    goals=env.env.parsed_problem['goal_state']
    cfg=OmegaConf.load(REPO/'examples/embodiment/config/model/gr00t.yaml')
    cfg.model_path=str(DIR if sft else HERE/'model');cfg.add_value_head=False
    if not sft:
        with (HERE/'model/model.safetensors').open('rb') as source:
            assert hashlib.file_digest(source,'sha256').hexdigest()=='eed4991e325b906d2976e53903fe3b5dd34bdf8b23961ca605db04d02313a5df'
    base=get_model(cfg).cuda();base.eval();base.requires_grad_(False)
    if sft:
        from safetensors import safe_open
        index=json.loads((DIR/'model.safetensors.index.json').read_text())['weight_map']
        live=dict(base.named_parameters()); checked=[]
        for shard in SHARDS:
            keys=sorted(k for k in live if index.get(k)==shard and live[k].numel()>128)
            assert keys
            with safe_open(DIR/shard,framework='pt') as src:
                for k in (keys[0],keys[len(keys)//2],keys[-1]):
                    actual=live[k].detach().flatten()[:128].cpu()
                    expected=src.get_tensor(k).flatten()[:128].to(actual.dtype)
                    assert torch.equal(actual,expected), k
                    checked.append(k)
        write(out/'loaded_sft.json',dict(revision=REV,checked_parameters=checked,method='128 parameter values from each named tensor matched after dtype conversion'))
    agent_class={'td3':TD3,'sac':SAC}[algorithm]
    extra=dict(ent_coef='auto_0.1',target_entropy=-3.) if algorithm=='sac' else {}
    algo=agent_class('MlpPolicy',Spaces(),buffer_size=50000,batch_size=128,learning_rate=.0003,gamma=.99,device='cpu',policy_kwargs=dict(net_arch=[128,128]),seed=SEED,**extra)
    with torch.no_grad():
        if algorithm=='sac':
            algo.actor.mu.weight.zero_();algo.actor.mu.bias.zero_()
            algo.actor.log_std.weight.zero_();algo.actor.log_std.bias.fill_(np.log(.22))
        else:
            for actor in [algo.actor,algo.actor_target]:
                actor.mu[-2].weight.zero_();actor.mu[-2].bias.zero_()
    algo.set_logger(configure(str(out/'metrics'),['csv']))
    rng=np.random.default_rng(SEED)
    ids=np.concatenate([rng.permutation(TRAIN) for _ in range(6)])
    def propose(obs):
        rgb=get_libero_image(obs).copy(); wrist=get_libero_wrist_image(obs).copy()
        state=np.r_[obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']]
        with torch.inference_mode():
            chunk,_=base.predict_action_batch(dict(main_images=torch.from_numpy(rgb)[None],wrist_images=torch.from_numpy(wrist)[None],states=torch.from_numpy(state)[None],task_descriptions=[suite.get_task(idx).language]),mode='eval')
        actions=chunk[0].cpu().numpy()
        def pool(im):return im.reshape(8,32,8,32,3).mean((1,3)).ravel()/127.5-1
        features=np.r_[state,actions.ravel(),pool(rgb),pool(wrist)].astype('float32')
        assert np.isfinite(features).all() and actions.shape==(5,7)
        return features,actions
    def episode(trial,seed,arm,stage):
        training=stage=='train'
        np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);env.seed(seed)
        env.reset();obs=env.set_init_state(states[trial])
        for _ in range(15):obs,*_=env.step(np.r_[np.zeros(6),-1.])
        feature,chunk=propose(obs)
        steps=0; flags=[bool(env.env._eval_predicate(g)) for g in goals]; any_goal=any(flags); total_reward=0.
        save_video=(stage=='test' and trial==TEST[0]) or stage in ('replica','repeat_noise')
        video=imageio.get_writer(out/f'{stage}-{arm}-{seed}.mp4',fps=20) if save_video else None
        while True:
            residual=algo.predict(feature,deterministic=not(training and algorithm=='sac'))[0] if arm!='base' else np.zeros(3,np.float32)
            if training and algorithm=='td3': residual=np.clip(residual+rng.normal(0,.22,3),-1,1).astype(np.float32)
            act=chunk.copy()
            if arm!='base': act[:,:3]+=SCALE*residual
            prior=sum(flags)
            for a in act:
                obs,_,done,_=env.step(a);steps+=1
                if video:video.append_data(get_libero_image(obs).copy())
                if done or steps>=512:break
            flags=[bool(env.env._eval_predicate(g)) for g in goals];any_goal|=any(flags)
            success=bool(env.check_success());assert success==all(flags)
            end=bool(done or success or steps>=512)
            reward=sum(flags)-prior+2*int(success)-penalty*float(np.mean(residual**2));total_reward+=reward
            nxt,new_chunk=(np.zeros_like(feature),None) if end else propose(obs)
            if training:
                algo.replay_buffer.add(feature[None],nxt[None],residual[None],np.array([reward]),np.array([end]),[{}])
                algo.num_timesteps+=1
                if algo.num_timesteps>=256:algo.train(1,128)
            if end:break
            feature,chunk=nxt,new_chunk
        if video:video.close()
        row=dict(stage=stage,arm=arm,trial=int(trial),seed=seed,init=hashes[trial],steps=steps,success=success,final=flags,any_goal=any_goal,reward=total_reward,replay=algo.replay_buffer.size() if algo.replay_buffer else 0)
        with (out/'episodes.jsonl').open('a') as f:print(json.dumps(row),file=f)
        return row
    if evaluate_published:
        assert sft and selection is not None and penalty == 0.1
        published = HERE.parents[1] / 'weights'
        checkpoints = [
            ('base', None, None),
            ('sac', SAC, 'sac/sac_40.zip'),
            ('new_sac', SAC, 'sac_repair_v1/online_120.zip'),
            ('td3', TD3, 'td3/td3_40.zip'),
        ]
        totals = {}
        for label, model_class, filename in checkpoints:
            if filename:
                algo = model_class.load(published / filename, device='cpu')
            rows = [episode(t, offset + t, label, 'test')
                    for t in TEST for offset in protocol['test_seed_offsets']]
            totals[label] = sum(row['success'] for row in rows)
        write(out / 'results.json', dict(episodes_per_arm=len(rows), success=totals))
        env.close()
        return
    if confirm:
        assert sft and selection is not None
        root=HERE/'sft_hard_v1'
        write(out/'confirm_protocol.json',dict(offset=91000,layouts=TEST,arms=['td3','sac_new'],reuse_layouts=True,new_noise_only=True,weights=['td3_40.zip','online_120.zip']))
        for alg,method,weights in [('td3',TD3,root/'td3/td3_40.zip'),('sac_new',SAC,root/'sac_repair_v1/online_120.zip')]:
            algo=method.load(weights,device='cpu')
            for t in TEST:episode(t,91000+t,alg,'confirm')
        env.close();return
    if repair or anchored or transfer:
        assert sft and algorithm=='sac' and selection is not None and penalty==.1
        if anchored or transfer:
            from sac_anchor import experiment
        else:
            from sac_repair import experiment
        def activate(agent):
            nonlocal algo
            algo=agent
        experiment(out,Spaces,episode,activate,ids,**({'transfer':True} if transfer else {}))
        env.close();return
    if screening is not None:
        rows=[episode(t,61000+t,'base','screen') for t in range(10)]
        score=sum(r['success'] for r in rows);eligible=False;rep=[]
        if 3<=score<=6:
            confirm=[episode(t,71000+t,'base','confirm') for t in range(10)]
            for failure in [r for r in rows if not r['success']][:3]:
                t=failure['trial']; a=episode(t,61000+t,'base','replica'); b=episode(t,81000+t,'base','repeat_noise')
                rep.append(not a['success'] and not b['success'] and not confirm[t]['success'] and failure['final']==a['final']==b['final']==confirm[t]['final'])
            eligible=(6<=score+sum(r['success'] for r in confirm)<=12 and sum(rep)>=2)
        write(out/'gate.json',dict(idx=idx,task=suite.get_task(idx).language,successes=score,eligible=eligible,repeated=rep,goals=goals))
        env.close();return
    best=None
    for i,trial in enumerate(ids,1):
        episode(trial,100000+i,'residual','train')
        if i%10==0:
            algo.logger.dump(algo.num_timesteps);print('training',i,algo.num_timesteps,algo._n_updates,flush=True)
        if i in protocol['checkpoints']:
            algo.save(out/f'{algorithm}_{i}');algo.save_replay_buffer(out/'replay.pkl')
            assert all(torch.isfinite(p).all() for p in algo.policy.parameters())
            val=[episode(t,30000+t,'residual',f'val_{i}') for t in VAL]
            rank=(sum(r['success'] for r in val),sum(sum(r['final']) for r in val),-i)
            if best is None or rank>best:best=rank
            print('validation',i,rank,flush=True)
    selected=-best[2];write(out/'selection.json',dict(checkpoint=selected,validation_successes=best[0],updates=algo._n_updates,actor_updates=algo._n_updates if algorithm=='sac' else algo._n_updates//2,replay=algo.replay_buffer.size()))
    algo=agent_class.load(out/f'{algorithm}_{selected}',device='cpu')
    paired=[]
    for trial in TEST:
        for offset in protocol['test_seed_offsets']:
            pair=[episode(trial,offset+trial,arm,'test') for arm in ['base','residual']]
            paired.append([int(r['success']) for r in pair])
        print('test layout',trial,flush=True)
    pairs=np.array(paired);d=(pairs[:,1]-pairs[:,0]).reshape(len(TEST),2).mean(1)
    boot=np.random.default_rng(SEED).choice(d,(20000,len(TEST))).mean(1)
    write(out/'results.json',dict(n=len(pairs),layouts=len(TEST),base=int(pairs[:,0].sum()),**{algorithm:int(pairs[:,1].sum())},wins=int((pairs[:,1]>pairs[:,0]).sum()),losses=int((pairs[:,1]<pairs[:,0]).sum()),diff_pp=100*d.mean(),layout_bootstrap95_pp=(100*np.percentile(boot,[2.5,97.5])).tolist(),selected=selected))
    env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--penalty',type=float,default=0.)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--screen',type=int);mode.add_argument('--selection',type=Path)
    p.add_argument('--sft',action='store_true')
    p.add_argument('--algorithm',choices=('td3','sac'),default='td3')
    p.add_argument('--repair',action='store_true')
    p.add_argument('--anchored',action='store_true')
    p.add_argument('--transfer',action='store_true')
    p.add_argument('--confirm',action='store_true')
    p.add_argument('--evaluate-published', action='store_true')
    args=p.parse_args();assert sum([args.anchored,args.repair,args.transfer,args.confirm,args.evaluate_published])<=1
    main(args.out.resolve(),args.penalty,args.screen,json.loads(args.selection.read_text()) if args.selection else None,args.sft,args.algorithm,args.repair,args.anchored,args.transfer,args.confirm,args.evaluate_published)
