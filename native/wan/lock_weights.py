"""Create a weights manifest from an already downloaded, revision-pinned HF snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): value.update(chunk)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--revision',required=True,help='Actual immutable Hugging Face snapshot commit, 40 hex digits')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',args.revision): parser.error('Supply an actual immutable revision, not main')
    root=args.checkpoint_dir.resolve()
    if not root.is_dir(): parser.error('Checkpoint directory does not exist')
    paths=[p for p in root.rglob('*') if p.is_file() and '.cache' not in p.relative_to(root).parts]
    for path in paths:
        if not path.resolve().is_relative_to(root): parser.error('Use a self-contained local-dir snapshot, not external symlinks')
    required={'diffusion_pytorch_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth','config.json'}
    names={p.relative_to(root).as_posix() for p in paths}
    if not required<=names or not any(n.startswith('google/umt5-xxl/') for n in names):
        parser.error('Incomplete Wan2.1 T2V 1.3B checkpoint/tokenizer files')
    if args.output.resolve().is_relative_to(root): parser.error('Write the manifest outside the checkpoint directory')
    manifest={'repository':'Wan-AI/Wan2.1-T2V-1.3B','revision':args.revision,
              'files':{p.relative_to(root).as_posix():digest(p) for p in sorted(paths)}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as stream: json.dump(manifest,stream,indent=2)
    print(json.dumps({'files':len(paths),'output':str(args.output),'revision':args.revision}))

if __name__=='__main__': main()
