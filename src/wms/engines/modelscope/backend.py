"""Pretrained low-memory T2V backend. The output contract is checked against decoded MP4 bytes."""
import json
import os
import sys
import time
from pathlib import Path


def generate(config,case,seed,output_path,context):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    import diffusers
    from diffusers import TextToVideoSDPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import export_to_video
    import imageio_ffmpeg
    p=case['generation']
    context.check_cancelled();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    context.event('progress',{'phase':'loading_model'})
    pipe=TextToVideoSDPipeline.from_pretrained(config['checkpoint_dir'],torch_dtype=torch.float16,
             variant='fp16',use_safetensors=True,local_files_only=True)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload(gpu_id=0)
    pipe.enable_vae_slicing()
    pipe.unet.enable_forward_chunking(chunk_size=1,dim=1)
    context.check_cancelled()
    def progress(step,timestep,latents):
        context.check_cancelled()
        context.event('progress',{'phase':'denoising','step':step+1,'total':p['num_inference_steps']})
    result=pipe(prompt=case['prompt'],height=p['height'],width=p['width'],
             num_frames=p['num_frames'],num_inference_steps=p['num_inference_steps'],
             guidance_scale=p['guidance_scale'],generator=torch.Generator(device='cpu').manual_seed(seed),
             callback=progress,callback_steps=1,output_type='np')
    context.check_cancelled();context.event('progress',{'phase':'exporting_video'})
    export_to_video(result.frames[0],str(output_path),fps=p['fps'])
    peak=torch.cuda.max_memory_allocated();pipe.maybe_free_model_hooks()
    return {'model_executed':True,'execution_mode':'native','python':sys.executable,'seed':seed,
       'prompt':case['prompt'],'requested':p,'elapsed_seconds':time.monotonic()-start,
       'peak_allocated_bytes':peak,'torch_version':torch.__version__,'diffusers_version':diffusers.__version__,
       'cuda_version':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'device':'cuda:0',
       'weights_repository':config.get('weights',{}).get('repository'),'weights_revision':config.get('weights',{}).get('revision'),
       'precision':'float16','offload':'model_cpu','safety_evaluation':False}
