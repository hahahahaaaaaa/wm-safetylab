"""Read recorded states/actions and perform official RSSM posterior inference.
This adapter never creates a simulator; inferred actions do not cause input frames.
"""
import json
import math
import os
from pathlib import Path
import sys
import time


def validate_sequence(sequence):
    if sequence.get('schema_version') != 'dreamer.sequence/1.0' or sequence.get('closed_loop') is not False:
        raise ValueError('Expected an explicitly offline Dreamer sequence')
    rows=sequence['steps']
    if not 2 <= len(rows) <= 128: raise ValueError('Sequence needs 2..128 steps')
    for index,row in enumerate(rows):
        obs=row['observation']
        if row['step'] != index or len(obs)!=4 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in obs):
            raise ValueError('Invalid finite four-dimensional observation or step order')
        if type(row['recorded_action']) is not int or row['recorded_action'] not in (0,1):
            raise ValueError('CartPole recorded action must be integer 0 or 1')
        if row['is_first'] != (index==0): raise ValueError('Expected one episode starting at step zero')
    return rows


def infer(config, case, seed, output, worker):
    root=Path(config['checkpoint_dir']); source=Path(config['weights']['source']['source_dir'])
    sequence=json.loads((root/'inference-sequence.json').read_text())
    rows=validate_sequence(sequence)
    rows = rows[:int(case.get('sequence_steps',len(rows)))]
    os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4')
    sys.path.insert(0,str(source))
    import numpy as np
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import elements
    from dreamerv3.agent import Agent
    worker.event('progress',{'phase':'loading_engineering_checkpoint','steps':len(rows)})
    started=time.perf_counter()
    agent_config=json.loads((root/'agent-config.json').read_text())
    agent_config.update(logdir=str(output),seed=seed)
    obs_space={'state':elements.Space(np.float32,(4,)), 'reward':elements.Space(np.float32),
               **{key:elements.Space(bool) for key in ('is_first','is_last','is_terminal')}}
    act_space={'action':elements.Space(np.int32,(),0,2)}
    agent=Agent(obs_space,act_space,elements.Config(agent_config))
    with np.load(root/'checkpoint.npz',allow_pickle=False) as archive:
        params={key:archive[key] for key in archive.files}
    expected=agent.save()['params']
    if params.keys()!=expected.keys(): raise ValueError('Checkpoint parameter keys do not exactly match the configured model')
    for name,value in params.items():
        if value.shape!=expected[name].shape or value.dtype!=expected[name].dtype:
            raise ValueError('Checkpoint parameter shape/dtype mismatch: '+name)
        if not np.isfinite(value).all(): raise ValueError('Nonfinite checkpoint parameter: '+name)
    counters=json.loads((root/'counters.json').read_text())
    agent.load({'params':params,'counters':counters})
    worker.check_cancelled()
    model=agent.model
    def predict():
        batch=1
        reset=jnp.asarray([[r['is_first'] for r in rows]],bool)
        observations={'state':jnp.asarray([[r['observation'] for r in rows]],jnp.float32)}
        previous={'action':jnp.asarray([[0]+[r['recorded_action'] for r in rows[:-1]]],jnp.int32)}
        _,_,tokens=model.enc(model.enc.initial(batch),observations,reset,training=False)
        _,_,features=model.dyn.observe(model.dyn.initial(batch),tokens,previous,reset,training=False)
        _,_,recons=model.dec(model.dec.initial(batch),features,reset,training=False)
        tensor=model.feat2tensor(features)
        policy=model.pol(tensor,bdims=2)['action']
        predicted_actions=policy.sample(nj.seed())
        return {'reconstruction':recons['state'].pred(),'actions':predicted_actions,
                'latent':features['deter']}
    worker.event('progress',{'phase':'offline_posterior_inference','recorded_actions_condition_state':True})
    _, prediction=jax.jit(nj.pure(predict))(agent.params,seed=seed)
    prediction=jax.device_get(prediction)
    worker.check_cancelled()
    outputs=[]
    for index,row in enumerate(rows):
        rec=np.asarray(prediction['reconstruction'][0,index],np.float32)
        latent=np.asarray(prediction['latent'][0,index],np.float32)
        if not np.isfinite(rec).all() or not np.isfinite(latent).all(): raise ValueError('Nonfinite model output')
        action=int(prediction['actions'][0,index])
        if action not in (0,1): raise ValueError('Invalid predicted action')
        outputs.append({'step':index,'observation':row['observation'],
                        'recorded_previous_action':None if index==0 else rows[index-1]['recorded_action'],
                        'recorded_action':row['recorded_action'],'predicted_action':action,
                        'reconstructed_observation':rec.tolist(),'latent_l2':float(np.linalg.norm(latent))})
    trajectory={'schema_version':'dreamer.inference/1.0','closed_loop':False,
                'checkpoint_class':'engineering_checkpoint','feedback':'offline_recorded_observation',
                'action_schema':{'dtype':'int32','values':{'0':'left','1':'right'},'executed':False},
                'samples':outputs}
    report={'implementation':'official_danijar_dreamerv3','source_commit':config['weights']['source']['commit'],
            'loaded_parameter_tensors':len(params),'checkpoint_loaded':True,
            'training_updates':config['weights']['training_updates'],'checkpoint_class':'engineering_checkpoint',
            'converged':False,'closed_loop':False,'inference_episode':sequence['episode_index'],
            'seed':seed,'sequence_steps':len(rows),'elapsed_seconds':time.perf_counter()-started,
            'backend':'cpu','checkpoint_sha256':config['weights']['files']['checkpoint.npz'],
            'data_sha256':config['weights']['files']['inference-sequence.json'],
            'conditioning':'posterior inference from recorded observations and previous recorded actions',
            'claim_limit':'Not a forecast score, task return, success rate, or safety evaluation'}
    return sequence,trajectory,report

