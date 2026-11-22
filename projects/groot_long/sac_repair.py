"""Matched replay SAC ablations, validation-only selection, then online continuation."""
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

ROOT=Path(__file__).resolve().parent/'sft_hard_v1'
SOURCE=ROOT/'sac/replay.pkl'
ARMS=[('control',0,1,.1),('warm',512,1,.1),('warm_delay',512,2,.1),('warm_delay_low',512,2,.01)]
STEPS=4096
def dump(path,data):path.write_text(json.dumps(data,indent=2))
def digest(module):
    h=hashlib.sha256()
    for tensor in module.parameters():h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def train_one(agent,warm,delay):
    tick=agent._n_updates+1
    with ExitStack() as scope:
        if tick<=warm or (tick-warm)%delay:
            scope.enter_context(patch.object(agent.actor.optimizer,'step',return_value=None))
            scope.enter_context(patch.object(agent.ent_coef_optimizer,'step',return_value=None))
        SAC.train(agent,1,128)
def counts(agent):
    def steps(opt):return sorted({int(s['step']) for s in opt.state.values() if 'step' in s}) or [0]
    return dict(critic=steps(agent.critic.optimizer),actor=steps(agent.actor.optimizer),temperature=steps(agent.ent_coef_optimizer))
def experiment(out,spaces,episode,activate,ids):
    seed=1307
    with SOURCE.open('rb') as src:sha=hashlib.file_digest(src,'sha256').hexdigest()
    assert sha=='4a65984a2d0ec6b4ad3ff1da56e10ad6a682bb0a8881cfe45e17f760c3739537'
    history=[json.loads(l) for l in (ROOT/'sac/episodes.jsonl').read_text().splitlines()]
    train=[r for r in history if r['stage']=='train']
    assert len(train)==120 and set(r['trial'] for r in train)==set(range(10,30))
    n=sum((r['steps']+4)//5 for r in train)
    dump(out/'repair_protocol.json',dict(source_sha=sha,source_chunks=n,source_episodes=120,arms=ARMS,common_offline_updates=STEPS,offline_validation_at=[2048,4096],val_offsets=[30000,33000],validation_trials=list(range(30,35)), online_epochs=120,online_validation_at=[40,80,120],selection='successes, final predicates, earlier candidate in listed order; retain offline checkpoint on ties',test_offsets=[41000,51000],test_trials=list(range(35,50)),point_target='at least 27/30, not a statistical equivalence test',teachers=False))
    best=None; best_file=None; best_params=None; scores=[]
    def validate(agent,label):
        activate(agent)
        rows=[episode(t,off+t,'residual','val_'+label) for off in (30000,33000) for t in range(30,35)]
        rank=(sum(r['success'] for r in rows),sum(sum(r['final']) for r in rows))
        scores.append(dict(label=label,rank=rank,counts=counts(agent)))
        dump(out/'val_scores.json',scores);print('validation',label,rank,flush=True);return rank
    for name,warm,delay,alpha in ARMS:
        agent=SAC('MlpPolicy',spaces(),device='cpu',seed=seed,ent_coef=f'auto_{alpha}',target_entropy=-3.,buffer_size=50000,batch_size=128,learning_rate=.0003,gamma=.99,policy_kwargs=dict(net_arch=[128,128]))
        with torch.no_grad():
            agent.actor.mu.weight.zero_();agent.actor.mu.bias.zero_();agent.actor.log_std.weight.zero_();agent.actor.log_std.bias.fill_(np.log(.22))
        actor0=digest(agent.actor);crit0=digest(agent.critic);alpha0=agent.log_ent_coef.item()
        agent.load_replay_buffer(SOURCE);assert agent.replay_buffer.size()==n and not agent.replay_buffer.full
        agent.set_logger(configure(str(out/name),['csv']))
        for i in range(1,STEPS+1):
            # Pair replay indices and reparameterization noise across all four arms, including after validation.
            np.random.seed(72000+i);torch.manual_seed(82000+i)
            train_one(agent,warm,delay)
            if i==512:
                assert digest(agent.critic)!=crit0
                if warm:assert digest(agent.actor)==actor0 and agent.log_ent_coef.item()==alpha0
                dump(out/(name+'_warm_audit.json'),dict(actor_changed=digest(agent.actor)!=actor0,critic_changed=True,counts=counts(agent)))
            if i%256==0:agent.logger.dump(i)
            if i in (2048,4096):
                label=f'{name}_{i}'; path=out/(label+'.zip');agent.save(path)
                rank=validate(agent,label)
                if best is None or rank>best:best,best_file,best_params=rank,path,(warm,delay)
    warm,delay=best_params
    dump(out/'offline_selection.json',dict(file=best_file.name,rank=best,warm=warm,delay=delay))
    agent=SAC.load(best_file,device='cpu');agent.load_replay_buffer(SOURCE);agent.set_logger(configure(str(out/'online'),['csv']))
    agent.num_timesteps=n
    online_start=agent._n_updates
    def continued(gradient_steps,batch_size):
        assert gradient_steps==1 and batch_size==128
        train_one(agent,warm,delay)
    activate(agent)
    with patch.object(agent,'train',side_effect=continued):
        for i,t in enumerate(ids,1):
            episode(int(t),200000+i,'residual','train')
            if i%10==0:agent.logger.dump(agent.num_timesteps);print('online',i,'updates',agent._n_updates-online_start,flush=True)
            if i in (40,80,120):
                label=f'online_{i}';rank=validate(agent,label)
                # Serialize without the process-local scheduling wrapper.
                path=out/(label+'.zip');agent.save(path,exclude=['train'])
                if rank>best:best,best_file=rank,path
    dump(out/'final_selection.json',dict(file=best_file.name,validation_rank=best,counts_last=counts(agent),online_updates=agent._n_updates-online_start,replay_final=agent.replay_buffer.size()))
    agent=SAC.load(best_file,device='cpu');activate(agent)
    for t in range(35,50):
        for offset in (41000,51000):
            for arm in ('base','residual'):episode(t,t+offset,arm,'test')
        print('paired layout',t,flush=True)
