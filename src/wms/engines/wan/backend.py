"""Thin bridge to the pinned upstream WanT2V API; no new sampling algorithm."""
import os
import sys
import time
from pathlib import Path


def generate(config, case, seed, output_path, context):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0,config['source_dir'])
    import torch
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    p=case['generation']
    cfg=WAN_CONFIGS['t2v-1.3B']
    context.check_cancelled()
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    context.event('progress', {'phase':'loading_model'})
    pipeline=wan.WanT2V(config=cfg, checkpoint_dir=config['checkpoint_dir'], device_id=0,
                        rank=0,t5_fsdp=False,dit_fsdp=False,use_usp=False,t5_cpu=p['t5_cpu'])
    context.check_cancelled()
    context.event('progress',{'phase':'generating','sampling_steps':p['sampling_steps']})
    video=pipeline.generate(case['prompt'],size=tuple(p['size']),frame_num=p['frame_num'],
            shift=p['shift'],sample_solver=p['sample_solver'],sampling_steps=p['sampling_steps'],
            guide_scale=p['guide_scale'],seed=seed,offload_model=p['offload_model'])
    context.check_cancelled()
    cache_video(tensor=video[None],save_file=str(output_path),fps=cfg.sample_fps,
                nrow=1,normalize=True,value_range=(-1,1))
    if not Path(output_path).is_file():
        raise ValueError('Upstream did not create a video file')
    return {'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'sample_fps':cfg.sample_fps,
            'negative_prompt':cfg.sample_neg_prompt,'parameter_dtype':str(cfg.param_dtype),
            't5_dtype':str(cfg.t5_dtype),'gpu':torch.cuda.get_device_name(0)}
