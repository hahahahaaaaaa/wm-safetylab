"""Build a bounded, explicitly non-converged engineering checkpoint from recorded data.
No environment package is imported and no simulator is launched.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def configuration(source, output, seed):
    import elements
    from ruamel.yaml import YAML
    raw = YAML(typ='safe').load((source / 'dreamerv3/configs.yaml').read_text())
    config = elements.Config(raw['defaults']).update(raw['size1m']).update({
        'batch_size': 2, 'batch_length': 16, 'report_length': 16,
        'replay_context': 0, 'seed': seed, 'logdir': str(output),
        'jax.platform': 'cpu', 'jax.compute_dtype': 'float32', 'jax.prealloc': False,
        'agent.opt.warmup': 0, 'agent.imag_length': 5,
    })
    value = dict(config.agent)
    for key in ('logdir','seed','jax','batch_size','batch_length','replay_context','report_length','replica','replicas'):
        value[key] = config[key]
    value['jax'] = {**dict(value['jax']), 'precompile': False, 'profiler': False, 'transfer_guard': False}
    return json.loads(json.dumps(value))


def make_agent(source, config):
    sys.path.insert(0, str(source))
    import numpy as np
    import elements
    from dreamerv3.agent import Agent
    observations = {'state': elements.Space(np.float32, (4,)),
                    'reward': elements.Space(np.float32),
                    **{key: elements.Space(bool) for key in ('is_first','is_last','is_terminal')}}
    actions = {'action': elements.Space(np.int32, (), 0, 2)}
    return Agent(observations, actions, elements.Config(config))


def read_episodes(path, limit=3):
    episodes = []
    current = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['episode_starts']:
                if current:
                    episodes.append(current)
                if len(episodes) == limit:
                    break
                current = []
            current.append(row)
        else:
            if current: episodes.append(current)
    if len(episodes) < limit or any(len(ep) != 500 for ep in episodes[:limit]):
        raise ValueError('Expected three complete, 500-step recorded episodes')
    return episodes[:limit]


def arrays(rows):
    import numpy as np
    length = len(rows)
    return {'state': np.asarray([x['obs'] for x in rows], dtype=np.float32),
            'action': np.asarray([x['actions'] for x in rows], dtype=np.int32),
            'reward': np.asarray([0.0]+[x['rewards'] for x in rows[:-1]], dtype=np.float32),
            'is_first': np.asarray([True]+[False]*(length-1)),
            'is_last': np.asarray([False]*(length-1)+[True]),
            'is_terminal': np.zeros(length, dtype=bool)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-lock', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--updates', type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.updates <= 128: raise ValueError('Engineering training budget is 1..128 updates')
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='4')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output/'checkpoint.npz').exists(): raise FileExistsError('Keep checkpoint immutable; choose a new output directory')
    lock = json.loads(Path(args.asset_lock).read_text())
    source = Path(lock['source']['source_dir'])
    if digest(lock['dataset']['path']) != lock['dataset']['sha256']: raise ValueError('Dataset hash mismatch')
    episodes = read_episodes(lock['dataset']['path'])
    split = {'train_episode_indices':[0,1], 'inference_episode_indices':[2],
             'rows_per_episode':[len(e) for e in episodes], 'episode_sha256':{str(i):hashlib.sha256(json.dumps(e,sort_keys=True,separators=(',',':')).encode()).hexdigest() for i,e in enumerate(episodes)}, 'episode_boundary':'episode_starts == 1',
             'termination_semantics':'500-step time-limit truncation; is_terminal=false, is_last=true at last row',
             'reward_alignment':'reward[0]=0; reward[t]=source.rewards[t-1]',
             'observation_fields':['cart_position_m','cart_velocity_m_s','pole_angle_rad','pole_angular_velocity_rad_s'],
             'action_space':{'dtype':'int32','shape':[],'values':{'0':'left','1':'right'}},
             'source':lock['dataset']}
    dump(output/'split.json', split)
    dump(output/'inference-sequence.json', {'schema_version':'dreamer.sequence/1.0','episode_index':2,
        'steps':[{'step':i,'observation':row['obs'],'recorded_action':row['actions'],
                  'is_first':i==0} for i,row in enumerate(episodes[2][:32])],
        'source':lock['dataset'], 'feedback':'offline_recorded_observation', 'closed_loop':False})
    config = configuration(source, output, 7)
    dump(output/'agent-config.json', config)
    started = time.perf_counter()
    agent = make_agent(source, config)
    import numpy as np
    import jax
    from embodied.jax import internal
    train = [arrays(ep) for ep in episodes[:2]]
    initial = {k:np.array(v) for k,v in agent.save()['params'].items() if k.startswith('dyn/') or k.startswith('enc/')}
    losses=[]
    for update in range(args.updates):
        start = (update * 16) % (500-16)
        data={key:np.stack([ep[key][start:start+16] for ep in train]) for key in train[0]}
        data['is_first'][:,0]=True
        data['reward'][:,0]=0.0
        data['consec']=np.zeros((2,16),np.int32)
        data['stepid']=np.zeros((2,16,20),np.uint8)
        data=internal.device_put(data,agent.train_sharded)
        data['seed']=agent._seeds(update,agent.train_mirrored)
        carry=agent.init_train(2)
        _,_,metrics=agent.train(carry,data)
        jax.block_until_ready(agent.params)
        scalar={key:float(np.asarray(value)) for key,value in metrics.items() if np.asarray(value).size==1}
        if not all(np.isfinite(list(scalar.values()))): raise ValueError('Nonfinite training loss')
        losses.append({'update':update+1,**scalar})
        print(json.dumps({'phase':'training','update':update+1,'elapsed_seconds':time.perf_counter()-started}),flush=True)
    saved=agent.save()
    changed=sum(not np.array_equal(value,saved['params'][key]) for key,value in initial.items())
    if changed==0: raise ValueError('No world-model parameters changed during training')
    np.savez_compressed(output/'checkpoint.npz', **saved['params'])
    dump(output/'counters.json', saved['counters'])
    dump(output/'training-metrics.json', losses)
    receipt={'schema_version':'1.0','checkpoint_class':'engineering_checkpoint','converged':False,
             'closed_loop_evaluation':False,'implementation':'official_danijar_dreamerv3',
             'source':lock['source'],'dataset':lock['dataset'],'seed':7,'training_updates':args.updates,
             'batch_size':2,'batch_length':16,'model_profile':'size1m','imagination_length':5,
             'world_model_changed_tensors':changed,'parameter_count':sum(int(v.size) for k,v in saved['params'].items() if not k.startswith('opt/')),
             'elapsed_seconds':time.perf_counter()-started,
             'files':{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
    dump(output/'weights-lock.json',receipt)
    packages={n:importlib.metadata.version(n) for n in ['jax','jaxlib','numpy','elements','ninjax','optax','chex','portal']}
    dump(output/'environment-lock.json',{'python':platform.python_version(),'packages':packages,'backend':'cpu'})
    print(json.dumps(receipt,indent=2))

if __name__ == '__main__': main()

