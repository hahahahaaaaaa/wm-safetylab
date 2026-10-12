"""Run with the isolated ManiSkill Python after installation; never overwrite a lock."""
import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


def write_once(path, value):
    if path.exists():
        if path.read_text(encoding='utf-8') != value:
            raise FileExistsError('Existing configuration differs; use a new versioned output directory: ' + str(path))
    else:
        path.write_text(value, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--gpu-device', default='0')
    args = parser.parse_args()
    if platform.system() != 'Linux' or importlib.metadata.version('mani-skill') != '3.0.1':
        raise ValueError('This deployment targets Linux and ManiSkill 3.0.1')
    if not re.fullmatch(r'[0-9]+', args.gpu_device):
        raise ValueError('One numeric GPU device is required')
    output, assets = args.output_dir.resolve(), args.asset_dir.resolve()
    if not assets.is_dir():
        raise ValueError('Create the designated project asset directory first')
    pins = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True)
    packages = {}
    for line in pins.splitlines():
        name, separator, version = line.partition('==')
        if not separator or not name or not version or re.search(r'\s', line):
            raise ValueError('Only exact package pins are supported: ' + line)
        packages[name] = version
    environment = {'python': platform.python_version(), 'packages': packages}
    runtime = {'python': sys.executable, 'dependency_lock': str(output/'requirements-native.lock'),
               'environment_lock': str(output/'environment-lock.json'),
               'gpu_device': args.gpu_device, 'asset_dir': str(assets)}
    output.mkdir(parents=True, exist_ok=True)
    write_once(output/'requirements-native.lock', pins)
    for name, value in [('environment-lock.json', environment), ('runtime.json', runtime)]:
        write_once(output/name, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'runtime': str(output/'runtime.json'), 'packages': len(packages), 'python': sys.executable}))


if __name__ == '__main__':
    main()
