"""Download immutable official code and recorded data, without installing a simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

COMMIT='e01491fad6434b2245a3b8ca201dd7faedcc458c'
REVISION='523ca21d86ef6844bad6e520cfa364ad132f1ed1'
ARCHIVE_SHA='f704884afb2cb642e0153df8af741e230fe220f52174fcf80245413141541f27'
DATA_SHA='51e085056e55ca7aa727bc9e66a2da2f266d9d3f7d28583f26483289c9143f28'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def download(url,path):
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=120) as response,partial.open('wb') as stream:
        while block:=response.read(1024*1024):stream.write(block)
    partial.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--world-root',type=Path,default=Path(__file__).resolve().parents[4])
    parser.add_argument('--hf-endpoint',default='https://huggingface.co')
    args=parser.parse_args()
    root=args.world_root.resolve(); project=root/'deployments/wm-safetylab'
    source=root/'repos'/('dreamerv3-'+COMMIT)
    archive=root/'downloads'/('dreamerv3-'+COMMIT+'.zip')
    download('https://codeload.github.com/danijar/dreamerv3/zip/'+COMMIT,archive)
    if digest(archive)!=ARCHIVE_SHA:raise ValueError('Upstream archive SHA256 mismatch')
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            destination=(root/'repos'/member.filename).resolve()
            if not destination.is_relative_to(source):raise ValueError('Archive member escaped source root')
            if member.is_dir():continue
            raw=z.read(member)
            if destination.exists():
                if hashlib.sha256(raw).hexdigest()!=digest(destination):raise ValueError('Existing upstream source changed: '+member.filename)
            else:
                destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    data=root/'datasets/cartpole-offline'/REVISION
    for name in ('README.md','teacher.jsonl'):
        download(args.hf_endpoint.rstrip('/')+'/datasets/NathanGavenski/CartPole-v1/resolve/'+REVISION+'/'+name,data/name)
    if digest(data/'teacher.jsonl')!=DATA_SHA:raise ValueError('Recorded dataset SHA256 mismatch')
    lock={'source':{'repo':'https://github.com/danijar/dreamerv3','commit':COMMIT,'source_dir':str(source),
          'archive_sha256':ARCHIVE_SHA,'files':{p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')}},
          'dataset':{'repo':'NathanGavenski/CartPole-v1','revision':REVISION,'path':str(data/'teacher.jsonl'),'sha256':DATA_SHA,'license':'MIT'}}
    folder=project/'native/dreamer';folder.mkdir(parents=True,exist_ok=True)
    (folder/'source-lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    print(json.dumps({'source':str(source),'dataset':str(data),'lock':str(folder/'source-lock.json')},indent=2))

if __name__=='__main__':main()
