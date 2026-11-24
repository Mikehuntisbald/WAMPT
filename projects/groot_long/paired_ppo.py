"""Compare native SB3 PPO to the archived improved residual SAC on LIBERO-Long."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from paired_ppo_env import (BANK, CFG, LOCK, REPOSITORY, VALIDATION_OFFSETS, digest, feed,
                            images_state, load_model, residual_features, rollout, setup, settle)
from offpolicy import SCALE, Spaces, write
from stable_baselines3 import PPO
from paired_ppo_report import read_rows, summarize


OUT = Path(os.environ.get('PAIRED_PPO_OUT', Path(__file__).resolve().parent / 'paired_ppo_v1')).resolve()
SEED = 1307
ROLLOUT = 1186
ROUNDS = 17
CANDIDATES = (6, 12, 17)
SAC_SHA = '642db9dc4704d42bdc08916a850b433435c74066bf1aedab92c6e7929331ee51'


def append(path, row):
    with path.open('a') as f:
        print(json.dumps(row), file=f)


class ChunkEnv(Spaces):
    def __init__(self):
        self.env, self.states = setup()
        self.base = load_model()
        self.goals = self.env.env.parsed_problem['goal_state']
        self.episode = 0
        self.ids = []
        self.layout_rng = np.random.default_rng(SEED)

    @torch.inference_mode()
    def propose(self, obs):
        view = images_state(obs)
        actions, _ = self.base.predict_action_batch(feed(*(x[None] for x in view)), mode='eval')
        self.chunk = actions[0].cpu().numpy()
        feature = residual_features(view, self.chunk)
        assert self.chunk.shape == (5, 7) and np.isfinite(feature).all()
        return feature

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.episode == 120:
            self.layout_rng = np.random.default_rng(SEED)
        if not self.ids:
            self.ids = list(self.layout_rng.permutation(LOCK['train']))
        self.layout = int(self.ids.pop(0))
        self.episode += 1
        self.env_seed = (100000 + self.episode if self.episode <= 120 else 200000 + self.episode - 120)
        obs = settle(self.env, self.states[self.layout], self.env_seed)
        self.count = sum(self.env.env._eval_predicate(g) for g in self.goals)
        self.steps = 0
        self.score = 0.
        return self.propose(obs), {}

    def step(self, residual):
        actions = self.chunk.copy()
        actions[:, :3] += SCALE * residual
        for action in actions:
            obs, _, done, _ = self.env.step(action)
            self.steps += 1
            if done or self.steps >= 512:
                break
        flags = [bool(self.env.env._eval_predicate(g)) for g in self.goals]
        success = bool(self.env.check_success())
        assert success == all(flags)
        reward = sum(flags) - self.count + 2 * int(success) - .1 * float(np.mean(residual ** 2))
        self.count = sum(flags)
        self.score += reward
        end = bool(done or success or self.steps >= 512)
        if end:
            append(OUT/'episodes.jsonl', dict(layout=self.layout, seed=self.env_seed, steps=self.steps,
                                              success=success, goals=flags, reward=self.score))
        feature = np.zeros(self.observation_space.shape, np.float32) if end else self.propose(obs)
        return feature, reward, end, False, {}

    def close(self):
        self.env.close()


class LoggedPPO(PPO):
    def train(self):
        super().train()
        iteration = self.num_timesteps // ROLLOUT
        assert all(p.isfinite().all() for p in self.policy.parameters())
        updates = int(next(iter(self.policy.optimizer.state.values()))['step'])
        append(OUT/'training.jsonl', dict(iteration=iteration, chunks=self.num_timesteps,
                                          optimizer_steps=updates, **{k:float(v) for k,v in self.logger.name_to_value.items()}))
        if iteration in CANDIDATES:
            self.save(OUT/f'ppo_{iteration}')
        print('PPO', iteration, self.num_timesteps, updates, flush=True)


def train():
    OUT.mkdir(parents=True, exist_ok=False)
    assert digest((REPOSITORY/'weights/sac_repair_v1/online_120.zip')) == SAC_SHA
    sac_run = json.loads((REPOSITORY/'evidence/sft_hard_v1/sac_repair_v1/final_selection.json').read_text())
    assert ROLLOUT * ROUNDS == sac_run['replay_final']
    write(OUT/'protocol.json', dict(task=LOCK['task'], train_layouts=LOCK['train'],
          test_manifest_sha256=digest(BANK/'initial/manifest.json'), sac_sha256=SAC_SHA,
          base_revision=LOCK['base']['revision'], seed=SEED, chunks=ROLLOUT*ROUNDS,
          rollout_chunks=ROLLOUT, candidates=CANDIDATES, validation='successes, total goals, earlier round',
          lr=.0003, batch_size=128, epochs=10, gamma=.99, gae=.95, clip=.2, entropy=0.,
          topology=[128,128], activation='ReLU', initial_std=.22, scale=SCALE,
          reward='delta predicates + 2*success - 0.1*mean(residual**2)', horizon=512,
          timeout_bootstrap=False, deterministic_eval=True, test_seeds=CFG['test_seeds']))
    env = ChunkEnv()
    agent = LoggedPPO('MlpPolicy', env, seed=SEED, device='cpu', n_steps=ROLLOUT, batch_size=128,
                      n_epochs=10, learning_rate=.0003, gamma=.99, gae_lambda=.95, clip_range=.2,
                      ent_coef=0., policy_kwargs=dict(net_arch=[128,128], activation_fn=torch.nn.ReLU,
                                                      log_std_init=float(np.log(.22))))
    with torch.no_grad():
        agent.policy.action_net.weight.zero_()
        agent.policy.action_net.bias.zero_()
    try:
        agent.learn(ROLLOUT*ROUNDS)
    finally:
        env.close()


def evaluate():
    agents = {f'ppo_{k}': PPO.load(OUT/f'ppo_{k}.zip',device='cpu') for k in CANDIDATES}
    rollout(list(agents), True, agents=agents, result_dir=OUT)
    val = [o+t for o in VALIDATION_OFFSETS for t in LOCK['validation']]
    ranks = {}
    for k in CANDIDATES:
        rows = read_rows(OUT/'val'/f'ppo_{k}', val)
        ranks[k] = [sum(r['success'] for r in rows), sum(sum(r['goals']) for r in rows)]
    selected = max(CANDIDATES, key=lambda k: (ranks[k],-k))
    arm = f'ppo_{selected}'
    write(OUT/'selection.json', dict(validation=ranks, selected=selected, sha256=digest(OUT/(arm+'.zip'))))
    # Always include the final endpoint as a secondary equal-training-budget comparison.
    final_arm = f'ppo_{ROUNDS}'
    rollout(list(dict.fromkeys([arm,final_arm])), False, agents=agents, result_dir=OUT)
    write(OUT/'results.json', summarize(BANK, OUT, selected, final_arm))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['train','eval'])
    args=p.parse_args()
    torch.set_num_threads(2)
    (train if args.stage=='train' else evaluate)()
