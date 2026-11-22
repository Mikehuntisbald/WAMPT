import json
from pathlib import Path
import subprocess
import numpy as np

directory = Path(__file__).resolve().parent/"rollouts"
data = json.loads((directory/"results.json").read_text())
rows = data["rows"]
assert len(rows)==5 and [r["trial"] for r in rows]==list(range(5))
assert len({r["initial_state_sha256"] for r in rows})==5
for row in rows:
    prefix = directory/f"trial{row['trial']}"
    with np.load(prefix.with_suffix(".npz")) as trace:
        flags, actions = trace["predicates"], trace["actions"]
    n = row["steps"]
    assert 1<=n<=512 and flags.shape==(n,2) and actions.shape==(n,7) and np.isfinite(actions).all()
    assert row["final"]==flags[-1].tolist() and row["success"]==bool(flags[-1].all())
    assert not flags[:-1].all(1).any()
    assert row["first_true_steps"]==[int(x[0])+1 if len(x) else None for x in (np.flatnonzero(col) for col in flags.T)]
    frames = subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",prefix.with_suffix(".mp4")],text=True)
    assert int(frames)==n
print(f"Verified {len(rows)} distinct layouts, {sum(r['steps'] for r in rows)} actions and video frames, {sum(r['success'] for r in rows)} completions.")
