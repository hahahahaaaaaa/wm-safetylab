"""Record this native interpreter and selected installed packages; no downloads or model loading."""
import json,platform,sys
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[2]
names=['torch','diffusers','transformers','accelerate','safetensors','imageio-ffmpeg']
environment={'python':platform.python_version(),'packages':{n:version(n) for n in names}}
folder=ROOT/'native/modelscope'
(folder/'environment-lock.json').write_text(json.dumps(environment,indent=2),encoding='utf8')
config={'python':sys.executable,'checkpoint_dir':str(folder/'weights/8227dddca75a8561bf858d604cc5dae52b954d01'),
        'weights_lock':str(folder/'weights-lock.json'),'environment_lock':str(folder/'environment-lock.json'),
        'cuda_visible_devices':'0'}
(folder/'runtime.json').write_text(json.dumps(config,indent=2),encoding='utf8')
print(json.dumps(config,indent=2))
