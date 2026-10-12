"""Restore the already downloaded pinned Wan source archive on the Linux host."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'native/wan';lock=json.loads((folder/'source-lock.json').read_text(encoding='utf8'))
name='Wan2.1-'+lock['commit'];archive=folder/(name+'.zip');repos=ROOT.parent.parent/'repos';source=repos/name
assert hashlib.sha256(archive.read_bytes()).hexdigest()==lock['archive_sha256']
if not source.exists():
    repos.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target=(repos/member.filename).resolve()
            assert target.is_relative_to(source.resolve()),member.filename
        source.mkdir();z.extractall(repos)
for rel,digest in lock['files'].items():
    assert hashlib.sha256((source/rel).read_bytes()).hexdigest()==digest,rel
assert {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}==set(lock['files'])
linux_lock=dict(lock,source_dir=str(source))
(folder/'source-lock.linux.json').write_text(json.dumps(linux_lock,indent=2),encoding='utf8')
print(json.dumps({'status':'passed','source_dir':str(source),'verified_files':len(lock['files']),'commit':lock['commit']}))
