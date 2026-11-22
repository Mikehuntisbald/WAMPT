"""SAC with a successful-behavior auxiliary loss and empirical return initialization."""
from sac_repair import ROOT,SOURCE,dump,counts
import hashlib,json
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import SAC,TD3
from stable_baselines3.common.logger import configure

def experiment(out,spaces,episode,activate,ids,transfer=False):
    seed=1307
    with SOURCE.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='4a65984a2d0ec6b4ad3ff1da56e10ad6a682bb0a8881cfe45e17f760c3739537'
    agent=SAC('MlpPolicy',spaces(),seed=seed,device='cpu',policy_kwargs=dict(net_arch=[128,128]),batch_size=128,learning_rate=.0003,gamma=.99,ent_coef='auto_0.01',target_entropy=-3.)
    agent.load_replay_buffer(SOURCE); buf=agent.replay_buffer;n=buf.size();assert n==10359
    rr=[q for l in (ROOT/'sac/episodes.jsonl').read_text().splitlines() if (q:=json.loads(l))['stage']=='train']
    good=[];returns=np.zeros(n,np.float32);start=0
    for row in rr:
        assert row['trial'] in range(10,30)
        end=row['replay'];assert not buf.dones[start:end-1,0].any() and buf.dones[end-1,0]
        if row['success']:good.extend(range(start,end))
        for j in reversed(range(start,end)):returns[j]=buf.rewards[j,0]+(.99*returns[j+1] if j+1<end else 0.)
        start=end
    assert start==n
    presteps=8192 if transfer else 1024;bc_weight=10. if transfer else .5
    dump(out/'anchor_protocol.json',dict(offline_source='original SAC training replay only',source_episodes=len(rr),successful_episodes=sum(r['success'] for r in rr),success_transition_count=len(good),mc_discount=.99,pretrain_steps=presteps,sac_updates_offline=4096,bc_weight=bc_weight,initial_alpha=.01,online_epochs=120,select='successes then final flags then earliest; validation 30..34 x offsets 30000/33000',test_trials=list(range(35,50)),test_offsets=[41000,51000],teachers='td3_40, train replay labels only' if transfer else False))
    rng=np.random.default_rng(seed);good=np.asarray(good); states=torch.from_numpy(buf.observations[:n,0]); actions=torch.from_numpy(buf.actions[:n,0]);ret=torch.from_numpy(returns)
    with torch.no_grad():
        agent.actor.mu.weight.zero_();agent.actor.mu.bias.zero_();agent.actor.log_std.weight.zero_();agent.actor.log_std.bias.fill_(np.log(.22))
    if transfer:
        good=np.arange(n);teacher=TD3.load(ROOT/'td3/td3_40.zip',device='cpu')
        targets=torch.from_numpy(teacher.predict(states.numpy(),deterministic=True)[0])
        del teacher
    else:targets=actions
    def clone_loss():
        ind=rng.choice(good,128);return (agent.actor(states[ind],deterministic=True)-targets[ind]).square().mean()
    for _ in range(presteps):
        ind=rng.integers(n,size=128);pred=agent.critic(states[ind],actions[ind])
        loss=sum(torch.nn.functional.smooth_l1_loss(q[:,0],ret[ind]) for q in pred)
        agent.critic.optimizer.zero_grad();loss.backward();agent.critic.optimizer.step()
        bc=clone_loss();agent.actor.optimizer.zero_grad();bc.backward();agent.actor.optimizer.step()
    agent.critic_target.load_state_dict(agent.critic.state_dict())
    agent.set_logger(configure(str(out/'metrics_anchor'),['csv']));activate(agent);agent.num_timesteps=n
    scores=[];best=(-1,-1);best_label=None
    def validate(label,eligible=True):
        nonlocal best,best_label
        rows=[episode(t,t+off,'residual','val_'+label) for off in (30000,33000) for t in range(30,35)]
        rank=sum(r['success'] for r in rows),sum(sum(r['final']) for r in rows)
        scores.append(dict(label=label,rank=rank,optim=counts(agent),sac_updates=agent._n_updates))
        dump(out/'val_scores.json',scores);agent.save(out/(label+'.zip'))
        if eligible and rank>best:best,best_label=rank,label
        print('anchor validation',label,rank,flush=True)
    validate('bc_only',False)
    # Add the supervised gradient just before native SAC applies its actor gradient.
    def anchor_hook(optimizer,args,kwargs):(bc_weight*clone_loss()).backward()
    handle=agent.actor.optimizer.register_step_pre_hook(anchor_hook)
    for i in range(1,4097):
        SAC.train(agent,1,128)
        if i%256==0:agent.logger.dump(i)
        if i in (2048,4096):validate(f'offline_{i}')
    for i,t in enumerate(ids,1):
        episode(int(t),200000+i,'residual','train')
        if i%10==0:agent.logger.dump(agent.num_timesteps);print('anchor online',i,flush=True)
        if i in (40,80,120):validate(f'online_{i}')
    handle.remove()
    dump(out/'final_selection.json',dict(label=best_label,validation=best,updates=agent._n_updates,counts=counts(agent),replay_final=buf.size()))
    agent=SAC.load(out/(best_label+'.zip'),device='cpu');activate(agent)
    for t in range(35,50):
        for offset in (41000,51000):
            for arm in ('base','residual'):episode(t,t+offset,arm,'test')
