"""Exercise both templates through an isolated local HTTP service and real workers.

Run with the platform Python. --sim-python selects the separate Gymnasium Python.
Nothing is submitted to the existing production service.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-python', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=19021)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('Choose an unused local port; do not replace an existing service')
    service_command = [sys.executable, '-m', 'wms', 'serve', '--port', str(args.port),
                       '--storage-root', str(output/'storage'),
                       '--registry', str(ROOT/'templates/model/registry.json'),
                       '--registry', str(ROOT/'templates/environment/registry.json')]
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
    base = f'http://127.0.0.1:{args.port}'
    report = {'api':base, 'service_command':service_command, 'runs':{}, 'checks':{}}
    with (output/'service.log').open('wb') as log:
        process = subprocess.Popen(service_command, cwd=ROOT, stdout=log, stderr=log,
                                   stdin=subprocess.DEVNULL, **options)
        try:
            with httpx.Client(base_url=base+'/api/v1', trust_env=False, timeout=20) as client:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError('Tutorial service exited; inspect service.log')
                    try:
                        client.get('/health').raise_for_status()
                        break
                    except httpx.HTTPError:
                        time.sleep(.1)
                else:
                    raise TimeoutError('Tutorial service did not become ready')

                def execute(label, config):
                    run=client.post('/runs',json=config).raise_for_status().json()
                    print(label+' '+run['id'],flush=True)
                    limit=time.monotonic()+240
                    while time.monotonic()<limit:
                        run=client.get('/runs/'+run['id']).raise_for_status().json()
                        if run['state'] not in {'queued','running','cancelling'}:
                            break
                        time.sleep(.1)
                    if run['state']!='completed':
                        raise RuntimeError(f"{label}: {run.get('error') or run['state']}")
                    evaluation=None
                    if run.get('automatic_evaluation_id'):
                        eid=run['automatic_evaluation_id']
                        while time.monotonic()<limit:
                            evaluation=client.get('/runs/'+eid).raise_for_status().json()
                            if evaluation['state'] not in {'queued','running','cancelling'}:
                                break
                            time.sleep(.1)
                        assert evaluation['state']=='completed' and not evaluation.get('metric_errors'), evaluation
                    report['runs'][label]={'run':run,'evaluation':evaluation}
                    return run,evaluation

                first,score=execute('model_predict',{'recipe':'onboarding-model-predict'})
                assert score['result']['metrics']['template_state_mae_m'] < 1e-12
                trained,_=execute('model_train',{'recipe':'onboarding-model-train'})
                checkpoint=output/'storage/runs'/trained['id']/'output/checkpoint.json'
                after,after_score=execute('checkpoint_predict',{
                    'recipe':'onboarding-model-predict',
                    'model':{'entrypoint':'wms_model_template.adapter:TeachingStateModel',
                             'config':{'checkpoint':str(checkpoint)}},
                    'inputs':{'reference':[[.08],[.24],[.16]]},
                })
                assert after_score['result']['metrics']['template_state_mae_m'] < 1e-12
                runtime={'python':str(Path(args.sim_python).resolve()),
                         'env':{'SDL_VIDEODRIVER':'dummy','SDL_AUDIODRIVER':'dummy'}}
                environment,env_score=execute('environment_rollout',{
                    'recipe':'template-pendulum-rollout','runtime':runtime})
                video,video_score=execute('environment_video',{
                    'recipe':'template-pendulum-video','runtime':runtime})
                assert env_score['result']['metrics']==video_score['result']['metrics']
                assert any(p['media_type']=='video/mp4' for p in video['result']['outputs'])
                response=client.get('/runs/'+video['id']+'/files/simulation.mp4',headers={'Range':'bytes=0-31'})
                assert response.status_code==206 and response.headers['content-type']=='video/mp4'
                report['checks']['model_predict_train_reload']=True
                report['checks']['genuine_simulator_state_and_video_same_metrics']=True
                report['checks']['video_http_range_supported']=True

                # Public Python and CLI use the same HTTP route; no duplicated adapter logic.
                import wms
                prior=os.environ.get('WMS_BASE_URL')
                os.environ['WMS_BASE_URL']=base
                try:
                    assert wms.get_run(first['id'])['state']=='completed'
                    compare=wms.compare([first['id'],after['id']])
                    assert len(compare['rows'])==2
                    cli=subprocess.run([sys.executable,'-m','wms','get-run',video['id']],
                                       cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=30)
                    (output/'cli.json').write_text(cli.stdout,encoding='utf-8')
                    assert cli.returncode==0 and json.loads(cli.stdout)['state']=='completed'
                    report['checks']['python_and_cli']=True
                finally:
                    if prior is None:
                        os.environ.pop('WMS_BASE_URL',None)
                    else:
                        os.environ['WMS_BASE_URL']=prior
                source=output/'storage/runs'/first['id']/'output'
                snapshot={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
                assert 'prediction_errors.json' not in snapshot
                report['checks']['metric_artifacts_in_separate_evaluation']=True
                report['checks']['distinct_worker_interpreters']=(
                    first['process']['command'][0]!=environment['process']['command'][0])
                assert report['checks']['distinct_worker_interpreters']
                report['passed']=all(report['checks'].values())
                (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps({'passed':report['passed'],'output':str(output),'checks':report['checks']}),flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)


if __name__=='__main__':
    main()
