"""Fetch only inference files at the published Long SFT revision."""
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

REV='2e6892a170c0220de88b2580d13df84e413c13b7'
MODEL='RLinf/RLinf-Gr00t-SFT-Long'
DIR=Path(os.environ.get('WAMPT_SFT_DIR', Path(__file__).resolve().parent / 'model_sft')).expanduser()
SHARDS={'model-00001-of-00002.safetensors':'86c39b245870e78419cd2d7d76f577eef5e6abbe8ae265bc54999e4e00ded0fc','model-00002-of-00002.safetensors':'4ff99b30f4d41035ff8398ab0b1059cd4df0d43a06d1fadf70729ac4206f6bfb'}

if __name__=='__main__':
    DIR.mkdir(parents=True,exist_ok=True)
    resp=requests.get(f'https://huggingface.co/api/models/{MODEL}/revision/{REV}?blobs=true',timeout=20);resp.raise_for_status();meta=resp.json()
    assert meta['sha']==REV
    def fetch(f):
        name=f['rfilename'];path=DIR/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            part=path.with_suffix('.part')
            subprocess.run(['curl','-fsSL','--retry','3','-C','-',f'https://huggingface.co/{MODEL}/resolve/{REV}/{name}','-o',part],check=True)
        else:part=path
        assert part.stat().st_size==f['size']
        with part.open('rb') as src:sha=hashlib.file_digest(src,'sha256').hexdigest()
        if name in SHARDS:assert sha==SHARDS[name]==f['lfs']['sha256']
        else:assert hashlib.sha1(f"blob {f['size']}\0".encode()+part.read_bytes()).hexdigest()==f['blobId']
        if not path.exists():part.rename(path)
        return name,sha
    wanted=set(SHARDS)|{'config.json','experiment_cfg/metadata.json','model.safetensors.index.json'}
    with ThreadPoolExecutor(3) as ex:files=dict(ex.map(fetch,[f for f in meta['siblings'] if f['rfilename'] in wanted]))
    assert set(files)==wanted
    (DIR/'verified.json').write_text(json.dumps(dict(model=MODEL,revision=REV,files=files),indent=2))
