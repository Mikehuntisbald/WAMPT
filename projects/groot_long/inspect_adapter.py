"""Quantify saturation on training states, without simulation or test feedback."""
import sys
from pathlib import Path
import json
import numpy as np
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[1]/'work/repos/dsrl/stable-baselines3'))
from stable_baselines3 import TD3

d=Path(sys.argv[1]);model=TD3.load(d/f'td3_{sys.argv[2]}',device='cpu')
model.load_replay_buffer(d/'replay.pkl')
buf=model.replay_buffer;n=buf.size()
if len(sys.argv)>3:
    n=int(sys.argv[3]);assert n<=buf.size() and not buf.full
x=model.predict(buf.observations[:n,0],deterministic=True)[0]
print(json.dumps(dict(n=n,checkpoint=sys.argv[2],mean_xyz=x.mean(0).tolist(),std_xyz=x.std(0).tolist(),
    fraction_abs_gt_095_per_axis=(np.abs(x)>.95).mean(0).tolist(),
    replay_positive_reward=int((buf.rewards[:n]>0).sum()),replay_negative_reward=int((buf.rewards[:n]<0).sum())),indent=2))
