"""Recompute paired success rates and layout bootstrap intervals from episode records."""
import json
from pathlib import Path

import numpy as np

SEED = 1307


def read_rows(directory, seeds):
    return [json.loads((directory/f'{seed}.json').read_text()) for seed in seeds]


def summarize(bank, out, selected, final_arm='ppo_17'):
    seeds = json.loads((bank/'protocol.json').read_text())['test_seeds']
    arm = f'ppo_{selected}'
    arms = {name:read_rows(folder, seeds) for name,folder in
            [('sft', bank/'test/base'),('sac', bank/'test/sac'),('ppo', out/'test'/arm),('ppo_final', out/'test'/final_arm)]}
    for i, seed in enumerate(seeds):
        for rows in arms.values():
            assert rows[i]['seed']==seed and rows[i]['initial']==arms['sac'][i]['initial']
            assert rows[i]['settled']==arms['sac'][i]['settled']
    scores = {name:np.array([int(r['success']) for r in rows]) for name,rows in arms.items()}
    pairs = {}
    for a,b in [('sac','ppo'),('ppo','sft'),('sac','sft'),('sac','ppo_final'),('ppo_final','sft')]:
        d = scores[a] - scores[b]
        boot = np.random.default_rng(SEED).choice(d,(20000,len(d))).mean(1)
        pairs[f'{a}_minus_{b}'] = dict(wins=int((d>0).sum()), losses=int((d<0).sum()),
                                        diff_pp=round(100*d.mean(),2),
                                        ci95_pp=np.round(100*np.percentile(boot,[2.5,97.5]),2).tolist())
    return dict(n=len(seeds),success={k:int(v.sum()) for k,v in scores.items()},
                                   paired=pairs, selected=selected, matched_settled_fingerprints=len(seeds))


if __name__ == '__main__':
    bank = Path(__file__).resolve().parents[2]/'evidence/paired_ppo_v1'
    selected = json.loads((bank/'selection.json').read_text())['selected']
    result = summarize(bank, bank, selected)
    assert result == json.loads((bank/'results.json').read_text())
    print(json.dumps(result, indent=2))
