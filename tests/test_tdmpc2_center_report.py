"""The study report summarizes all evidence without model/GPU dependencies."""
import hashlib
import json
from pathlib import Path

import pytest

from wms.engines.tdmpc2.center_report import build


class Context:
    def __init__(self, output):
        self.output = output

    def check_cancelled(self):
        pass

    def write_json(self, name, value):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / name).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def snapshot(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


def fixture(tmp_path, initial=False):
    root = tmp_path / 'runs'
    training = root / 'run_aa' / 'output'
    test = root / 'run_bb' / 'output'
    output = root / 'run_cc' / 'output'
    task = {'env_id': 'WMS-PushCubeCenter-v1', 'reward_version': 'center-dense/1', 'task_sha256': 'test-task-identity'}
    write(training / 'training.json', {'task': 'train', **task, 'seed': 0, 'environment_steps': 204800, 'update_count': 25600,
        'checkpoints': ['initial.pt', 'checkpoint_0051200.pt', 'model.pt']})
    for filename in ('initial.pt', 'checkpoint_0051200.pt', 'model.pt'):
        (training / filename).write_bytes(('fixture weights: ' + filename).encode())
    (training / 'training_history.jsonl').write_text(json.dumps({'environment_steps': 1600, 'collection_return': 1.0, 'collection_success_once': 0.0}) + '\n' + json.dumps({'environment_steps': 204800, 'collection_return': 15., 'collection_success_once': .75}) + '\n')
    recording = {'train_step_start': 0, 'train_step_end': 1600, 'return': .5, 'success_at_end': False,
                 'final_goal_distance_m': .2, 'random_action_steps': 50,
                 'policy_mode': 'exploratory_training_policy_env0', 'video': 'training.mp4',
                 'video_webm': 'training.webm', 'video_poster': 'training_final.png',
                 'video_frames': 51, 'video_fps': 20, 'video_duration_s': 2.55}
    write(training / 'training_recordings.json', {'recordings': [recording], **task})
    for filename in ('training.mp4', 'training.webm', 'training_final.png'):
        (training / filename).touch()
    episodes = []
    for index in range(20):
        success = index != 7
        ep = {'seed': 3000 + index, 'center_success_at_end': success, 'center_stable_last_5': success,
              'region_success_at_end': True, 'final_goal_distance_m': .01 if success else .03, 'steps': 50,
              'video': f'episode_{index}.mp4', 'video_webm': f'episode_{index}.webm',
              'video_poster': f'episode_{index}_final.png', 'video_frames': 51, 'video_fps': 20, 'video_duration_s': 2.55}
        episodes.append(ep)
    report = {'task': 'policy_evaluate', 'env_id': task['env_id'], 'task_sha256': task['task_sha256'],
              'protocol': {'reward_version': task['reward_version'], 'horizon': 50},
              'checkpoint': str(training / 'model.pt'),
              'checkpoint_sha256': hashlib.sha256((training / 'model.pt').read_bytes()).hexdigest(), 'episodes': episodes}
    write(test / 'evaluation.json', report)
    for episode in episodes:
        for field in ('video', 'video_webm', 'video_poster'):
            (test / episode[field]).touch()
    inputs = {'training_run': 'run_aa', 'test_run': 'run_bb'}
    if initial:
        inputs['initial_run'] = 'run_dd'
        initial_report = {**report, 'checkpoint': str(training / 'initial.pt'),
                          'checkpoint_sha256': hashlib.sha256((training / 'initial.pt').read_bytes()).hexdigest(), 'episodes': [{**ep, 'center_success_at_end': False, 'center_stable_last_5': False,
                                                'final_goal_distance_m': .2, 'region_success_at_end': False} for ep in episodes]}
        write(root / 'run_dd' / 'output' / 'evaluation.json', initial_report)
    return training, test, output, inputs


def test_full_cohort_videos_and_comparison_do_not_modify_evidence(tmp_path):
    training, test, output, inputs = fixture(tmp_path, initial=True)
    before = snapshot(training), snapshot(test)
    result = build({'inputs': inputs}, output, Context(output))
    assert (snapshot(training), snapshot(test)) == before
    summary = result['data']
    assert summary['test']['episodes'] == 20
    assert summary['test']['center_successes'] == 19
    assert summary['test']['failed_center_seeds'] == [3007]
    assert summary['test']['final_distance_mean_m'] == pytest.approx(.011)
    assert summary['initial']['center_successes'] == 0
    assert len(summary['test_episodes']) == 20
    page = (output / 'experiment.html').read_text(encoding='utf-8')
    assert '种子 3007 · 中心任务失败' in page
    assert '种子 3019 · 中心任务成功' in page
    assert page.index('/api/v1/runs/run_bb/files/episode_7.webm') < page.index('/api/v1/runs/run_bb/files/episode_7.mp4')
    assert '/api/v1/runs/run_bb/files/episode_7_final.png' in page
    assert '初始与训练后检查点对照' in page
    assert '不是独立测试成功率' in page
    assert 'training_recordings.json' in page
    assert {item['path'] for item in result['outputs']} == {'evidence_summary.json', 'experiment.html'}


def test_report_without_initial_run_omits_initial_comparison(tmp_path):
    _, _, output, inputs = fixture(tmp_path)
    result = build({'inputs': inputs}, output, Context(output))
    assert result['data']['initial'] is None
    assert '初始与训练后检查点对照' not in (output / 'experiment.html').read_text(encoding='utf-8')


def test_report_rejects_wrong_task_or_missing_measured_center_status(tmp_path):
    _, test, output, inputs = fixture(tmp_path)
    path = test / 'evaluation.json'
    report = json.loads(path.read_text())
    report['env_id'] = 'PushCube-v1'
    write(path, report)
    with pytest.raises(ValueError, match='independent center task'):
        build({'inputs': inputs}, output, Context(output))
    report['env_id'] = 'WMS-PushCubeCenter-v1'
    del report['episodes'][3]['center_stable_last_5']
    write(path, report)
    with pytest.raises(ValueError, match='center_stable_last_5'):
        build({'inputs': inputs}, output, Context(output))
    assert not output.exists()


def test_development_seed_overlap_is_disclosed(tmp_path):
    _, _, output, inputs = fixture(tmp_path)
    result = build({'inputs': {**inputs, 'development_runs': ['run_bb']}}, output, Context(output))
    assert result['data']['development_test_seed_overlap'] == list(range(3000, 3020))
    assert '不能声明为完全独立的保留测试集' in (output / 'experiment.html').read_text(encoding='utf-8')


def test_report_rejects_source_output_as_destination_and_path_escape(tmp_path):
    training, test, output, inputs = fixture(tmp_path)
    with pytest.raises(ValueError, match='own output directory'):
        build({'inputs': inputs}, training, Context(training))
    path = test / 'evaluation.json'
    report = json.loads(path.read_text())
    report['episodes'][0]['video_webm'] = '../../secret.webm'
    write(path, report)
    with pytest.raises(ValueError, match='inside the source'):
        build({'inputs': inputs}, output, Context(output))
    assert not output.exists()


@pytest.mark.parametrize('role', ['test', 'development', 'initial'])
def test_report_rejects_checkpoint_from_another_training_even_with_same_task(tmp_path, role):
    training, test, output, inputs = fixture(tmp_path, initial=True)
    if role == 'development':
        path = output.parents[1] / 'run_ee' / 'output' / 'evaluation.json'
        write(path, json.loads((test / 'evaluation.json').read_text()))
        inputs['development_runs'] = ['run_ee']
    elif role == 'initial':
        path = output.parents[1] / 'run_dd' / 'output' / 'evaluation.json'
    else:
        path = test / 'evaluation.json'
    report = json.loads(path.read_text())
    other = output.parents[1] / 'run_ff' / 'output' / Path(report['checkpoint']).name
    other.parent.mkdir(parents=True)
    other.write_bytes(Path(report['checkpoint']).read_bytes())
    report['checkpoint'] = str(other)
    write(path, report)
    with pytest.raises(ValueError, match='selected training_run output'):
        build({'inputs': inputs}, output, Context(output))
    assert not output.exists()


@pytest.mark.parametrize('role', ['test', 'initial'])
def test_report_rejects_wrong_checkpoint_digest(tmp_path, role):
    _, test, output, inputs = fixture(tmp_path, initial=True)
    path = (test / 'evaluation.json') if role == 'test' else output.parents[1] / 'run_dd' / 'output' / 'evaluation.json'
    report = json.loads(path.read_text())
    report['checkpoint_sha256'] = '0' * 64
    write(path, report)
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        build({'inputs': inputs}, output, Context(output))
    assert not output.exists()


def test_initial_comparison_cannot_relabel_trained_weights(tmp_path):
    training, test, output, inputs = fixture(tmp_path, initial=True)
    path = output.parents[1] / 'run_dd' / 'output' / 'evaluation.json'
    report = json.loads(path.read_text())
    report['checkpoint'] = str(training / 'model.pt')
    report['checkpoint_sha256'] = hashlib.sha256((training / 'model.pt').read_bytes()).hexdigest()
    write(path, report)
    with pytest.raises(ValueError, match='initial.pt'):
        build({'inputs': inputs}, output, Context(output))
    assert not output.exists()


def test_unlisted_checkpoint_is_not_accepted_as_selected_training(tmp_path):
    training, test, output, inputs = fixture(tmp_path)
    checkpoint = training / 'unlisted.pt'
    checkpoint.write_bytes(b'unknown weights')
    path = test / 'evaluation.json'
    report = json.loads(path.read_text())
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    write(path, report)
    with pytest.raises(ValueError, match='listed by the selected training'):
        build({'inputs': inputs}, output, Context(output))
    assert not output.exists()


def test_listed_checkpoint_and_hash_cache_are_used_for_development(tmp_path, monkeypatch):
    from wms.engines.tdmpc2 import center_report
    training, test, output, inputs = fixture(tmp_path)
    path = test / 'evaluation.json'
    report = json.loads(path.read_text())
    checkpoint = training / 'checkpoint_0051200.pt'
    report.update(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    write(path, report)
    calls = []
    original = center_report._checkpoint_sha256
    def counted(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(center_report, '_checkpoint_sha256', counted)
    result = build({'inputs': {**inputs, 'development_runs': ['run_bb']}}, output, Context(output))
    assert result['data']['test']['episodes'] == 20
    assert calls == [checkpoint.resolve()]
