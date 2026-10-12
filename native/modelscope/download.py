"""Download only the fixed FP16 ModelScope snapshot, with upstream LFS hash checks."""
import hashlib,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import httpx
ROOT=Path(__file__).resolve().parents[2]
REPO='ali-vilab/text-to-video-ms-1.7b'
REVISION='8227dddca75a8561bf858d604cc5dae52b954d01'
TARGET=ROOT/'native/modelscope/weights'/REVISION

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 TARGET.mkdir(parents=True,exist_ok=True)
 with httpx.Client(timeout=60,follow_redirects=True) as client:
  r=client.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true');r.raise_for_status();info=r.json()
 assert info['sha']==REVISION
 files=[f for f in info['siblings'] if f['rfilename'].endswith(('.json','.txt','.md')) or f['rfilename'].endswith('fp16.safetensors')]
 def download(item):
  name=item['rfilename']; path=TARGET/name;path.parent.mkdir(parents=True,exist_ok=True)
  expected=item.get('lfs',{}).get('sha256')
  if path.exists() and path.stat().st_size==item['size'] and (not expected or digest(path)==expected):
   return name,digest(path)
  temp=path.with_suffix(path.suffix+'.part')
  start=temp.stat().st_size if temp.exists() else 0
  headers={'Range':f'bytes={start}-'} if start else {}
  url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true&t={int(time.time())}'
  with httpx.Client(timeout=120,follow_redirects=True) as client:
   with client.stream('GET',url,headers=headers) as response:
    response.raise_for_status()
    if response.status_code!=206:start=0
    with temp.open('ab' if start else 'wb') as stream:
     count=start;last=count
     for chunk in response.iter_bytes(4*1024*1024):
      stream.write(chunk);count+=len(chunk)
      if count-last>=256*1024*1024:
       print(f'{name}: {count/1e6:.0f}/{item["size"]/1e6:.0f} MB',flush=True);last=count
  if temp.stat().st_size!=item['size']:raise ValueError('Download size mismatch: '+name)
  sha=digest(temp)
  if expected and sha!=expected:raise ValueError('Upstream LFS hash mismatch: '+name)
  temp.replace(path);print('verified '+name,flush=True)
  return name,sha
 with ThreadPoolExecutor(max_workers=3) as pool: hashes=dict(pool.map(download,files))
 lock={'repository':REPO,'revision':REVISION,'variant':'fp16','files':hashes}
 (ROOT/'native/modelscope/weights-lock.json').write_text(json.dumps(lock,indent=2),encoding='utf8')
 print(json.dumps({'checkpoint_dir':str(TARGET),'files':len(hashes),'status':'verified'}),flush=True)

if __name__=='__main__':main()
