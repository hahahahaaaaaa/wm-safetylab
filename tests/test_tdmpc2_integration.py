"""Small CPU tests for platform wiring; GPU behavior is tested on the server."""
import json
from pathlib import Path

from wms.engines.tdmpc2.native import _settings, _source


def test_run_references_resolve_checkpoint_and_evidence(tmp_path):
    root = tmp_path / "runs"
    train = root / "run_aa" / "output"
    evaluated = root / "run_bb" / "output"
    destination = root / "run_cc" / "output"
    train.mkdir(parents=True)
    evaluated.mkdir(parents=True)
    for filename in ("initial.pt", "model.pt"):
        (train / filename).write_bytes(b"fixture path only")
    (evaluated / "evaluation.json").write_text("{}")
    inputs = {"source_run": "run_aa", "evaluation_run": "run_bb"}
    assert _source(inputs, destination) == train / "model.pt"
    assert _source({**inputs, "initial_checkpoint": True}, destination) == train / "initial.pt"
    assert _source(inputs, destination, "evaluation_report") == evaluated / "evaluation.json"


def test_platform_device_override_does_not_rewrite_machine_file(tmp_path):
    runtime = tmp_path / "runtime.json"
    original = {"gpu_device": "0", "python": "/machine/python"}
    runtime.write_text(json.dumps(original))
    value, params, inputs = _settings({"runtime": {"config_file": str(runtime),
                                                   "env": {"CUDA_VISIBLE_DEVICES": "2"}},
                                      "params": {"episodes": 10}})
    assert value["gpu_device"] == "2"
    assert params == {"episodes": 10} and inputs == {}
    assert json.loads(runtime.read_text()) == original


def test_official_success_is_not_center_accuracy(tmp_path):
    from wms.engines.tdmpc2.native import _goal_position_diagnostics, _html_evaluation
    distances, diagnosis = _goal_position_diagnostics([[.2, 0, .02], [.08, 0, .02]], [[0, 0, 0]] * 2)
    assert distances == [.2, .08]
    assert diagnosis['center_within_2cm'] is False
    episode = {'seed': 1000, 'return': 40, 'success_once': True, 'success_at_end': True,
               'action_intervention_steps': 0, 'video': 'episode.mp4', 'video_webm': 'episode.webm',
               'video_frames': 51, 'video_fps': 20, **diagnosis}
    report = {'episodes': [episode], 'checkpoint': 'model.pt', 'intervention': {},
              'protocol': {'official_goal_radius_m': .1, 'official_cube_z_threshold_m': .025}}
    _html_evaluation(tmp_path, report)
    page = (tmp_path / 'visualization.html').read_text(encoding='utf-8')
    assert '官方目标区域成功不等于推到圆心' in page
    assert '<td>是</td><td>是</td><td>8.00 cm</td><td>8.00 cm</td><td>否</td>' in page
    assert '回放 2.55 秒' in page
    assert page.index('episode.webm') < page.index('episode.mp4')


def test_center_diagnostic_uses_final_not_best_position():
    from wms.engines.tdmpc2.native import _goal_position_diagnostics
    _, diagnosis = _goal_position_diagnostics([[.01, 0, .02], [.08, 0, .02]], [[0, 0, 0]] * 2)
    assert diagnosis['min_goal_distance_m'] == .01
    assert diagnosis['center_within_2cm'] is False
    _, boundary = _goal_position_diagnostics([[.02, 0, .02]], [[0, 0, 0]])
    assert boundary['center_within_2cm'] is True


def test_historical_report_does_not_invent_distance(tmp_path):
    from wms.engines.tdmpc2.native import _html_evaluation
    report = {'episodes': [{'seed': 1, 'return': 2, 'success_once': True, 'success_at_end': True,
                           'action_intervention_steps': 0, 'video': None}],
              'checkpoint': 'old.pt', 'intervention': {}}
    _html_evaluation(tmp_path, report)
    page = (tmp_path / 'visualization.html').read_text(encoding='utf-8')
    assert '<td>未记录</td><td>未记录</td><td>未记录</td>' in page
    assert '本回合未录制（不是播放失败）' in page
