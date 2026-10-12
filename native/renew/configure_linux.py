"""Create exact locks for an already installed RENEW environment (Linux only).

Run with its native interpreter. This does not install packages or modify upstream.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
commit = '8f6e1ebf66a0200cc2015226cdd7d70900fc0cb7'
source = a.source.resolve()
assert os.name == 'posix', 'Configure this runtime on its Linux execution host'
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip() == commit
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=all'], text=True).strip()
packages = dict(sorted((dist.metadata['Name'],dist.version) for dist in importlib.metadata.distributions()))
output = a.output.resolve()
output.mkdir(parents=True,exist_ok=True)
values = {
    'requirements.lock': '\n'.join(f'{name}=={version}' for name,version in packages.items())+'\n',
    'environment-lock.json': json.dumps({'python':platform.python_version(),'packages':packages},indent=2)+'\n',
    'runtime.json': json.dumps({'python':str(Path(sys.executable).absolute()),'source_dir':str(source),'source_commit':commit,
        'dependency_lock':str(output/'requirements.lock'),'environment_lock':str(output/'environment-lock.json')},indent=2)+'\n',
}
for name,content in values.items():
    target=output/name
    if target.exists() and target.read_text()!=content:
        raise SystemExit('Refusing to silently change existing lock: '+str(target))
for name,content in values.items():
    (output/name).write_text(content,encoding='utf-8')
print(json.dumps({'runtime':str(output/'runtime.json'),'python':platform.python_version(),'packages':len(packages),'source_commit':commit}))