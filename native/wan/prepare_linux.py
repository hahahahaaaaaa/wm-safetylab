"""Verify the downloaded official snapshot and seal this host's Wan runtime."""
import hashlib, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[1]
OUT = ROOT/'results/wan-linux-20261004'
NATIVE = ROOT/'native/wan'
CHECKPOINT = WORKSPACE/'weights/Wan2.1-T2V-1.3B'
PYTHON = ROOT/'.venv-wan/bin/python'
REVISION = '37ec512624d61f7aa208f7ea8140a131f93afc9a'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def write_once(path, text):
    if path.exists():
        if path.read_text() != text: raise ValueError('Existing file differs: '+str(path))
    else: path.write_text(text,encoding='utf8')

info = json.loads((OUT/'model-info.official.json').read_text())
assert info['sha'] == REVISION and info['id'] == 'Wan-AI/Wan2.1-T2V-1.3B'
required = {'config.json','diffusion_pytorch_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth','LICENSE.txt'}
files = [x for x in info['siblings'] if x['rfilename'] in required or x['rfilename'].startswith('google/umt5-xxl/')]
verified = []
for item in files:
    name = item['rfilename']; p = CHECKPOINT/name
    assert p.is_file() and p.stat().st_size==item['size'],name
    value=sha(p)
    if item.get('lfs'): assert value==item['lfs']['sha256'],name
    else:
        body=p.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==item['blobId'],name
    verified.append({'file':name,'size_bytes':p.stat().st_size,'sha256':value})
    print('Verified '+name,flush=True)
manifest = {'repository':info['id'],'revision':REVISION,'files':{i['file']:i['sha256'] for i in verified}}
write_once(NATIVE/'weights-lock.json',json.dumps(manifest,indent=2))
freeze = subprocess.check_output([str(PYTHON),'-m','pip','freeze','--all'],text=True)
write_once(NATIVE/'requirements-native.lock',freeze)
runtime = {'python':str(PYTHON),'source_dir':str(WORKSPACE/'repos/Wan2.1-9737cba9c1c3c4d04b33fcad41c111989865d315'),
 'source_lock':str(NATIVE/'source-lock.linux.json'),'checkpoint_dir':str(CHECKPOINT),
 'weights_lock':str(NATIVE/'weights-lock.json'),'dependency_lock':str(NATIVE/'requirements-native.lock'),'cuda_visible_devices':'0'}
write_once(NATIVE/'runtime.json',json.dumps(runtime,indent=2))
(OUT/'assets-verified.json').write_text(json.dumps({'status':'passed','metadata_source':'https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B?blobs=true','revision':REVISION,'files':verified},indent=2))
print(json.dumps({'status':'passed','files':len(verified),'bytes':sum(x['size_bytes'] for x in verified),'runtime':str(NATIVE/'runtime.json')}),flush=True)