"""Read-only TD-MPC2 metrics; no Torch, NumPy, ManiSkill or GPU required."""
import hashlib
import json

import pytest

from wms.engines.tdmpc2.metrics import measure


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def snapshot(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def training():
    return {"task": "train", "environment_steps": 40, "update_count": 12,
            "episodes": 2, "elapsed_seconds": 3.5, "parameter_delta_l2": 0.25,
            "finite_losses": True, "last_losses": {"total_loss": 4.0,
            "consistency_loss": 1.0, "reward_loss": 2.0, "value_loss": 3.0}}


def policy():
    return {"task": "policy_evaluate", "episodes": [
        {"seed": 100, "return": 2.0, "success_once": True, "success_at_end": False,
         "steps": 2, "action_intervention_steps": 1},
        {"seed": 101, "return": -4.0, "success_once": False, "success_at_end": False,
         "steps": 6, "action_intervention_steps": 1},
    ]}


def test_training_metrics_are_known_and_source_remains_unchanged(tmp_path):
    source, output = tmp_path / "source", tmp_path / "evaluation"
    write(source / "training.json", training())
    write(source / "result.json", {"data": {"task": "train"}})
    (source / "checkpoint.pt").write_bytes(b"immutable saved checkpoint")
    before = snapshot(source)
    result = {"source_output_dir": str(source), "output_dir": str(output)}
    values = measure(result, {})
    assert values == {"tdmpc2_environment_steps": 40, "tdmpc2_update_count": 12,
                      "tdmpc2_training_episodes": 2, "tdmpc2_elapsed_seconds": 3.5,
                      "tdmpc2_parameter_delta_l2": 0.25, "tdmpc2_finite_losses": 1,
                      "tdmpc2_last_total_loss": 4.0, "tdmpc2_last_consistency_loss": 1.0,
                      "tdmpc2_last_reward_loss": 2.0, "tdmpc2_last_value_loss": 3.0}
    assert measure(result, {}) == values
    assert snapshot(source) == before
    report = json.loads((output / "tdmpc2_metrics.json").read_text())
    assert report["safety_status"] == "not_measured"
    assert report["metrics"] == values
    assert not any("safe" in key or "collision" in key for key in values)


def test_policy_metrics_include_failures_and_use_actual_step_denominator(tmp_path):
    source = tmp_path / "source"
    write(source / "evaluation.json", policy())
    values = measure({"source_output_dir": str(source), "output_dir": str(tmp_path / "evaluation")}, {})
    assert values == {"tdmpc2_episode_count": 2, "tdmpc2_return_mean": -1.0,
                      "tdmpc2_success_once_rate": 0.5, "tdmpc2_success_at_end_rate": 0.0,
                      "tdmpc2_episode_steps_mean": 4.0, "tdmpc2_action_intervention_steps": 2,
                      "tdmpc2_action_intervention_rate": 0.25}


def test_prediction_metrics_preserve_latent_and_reward_names(tmp_path):
    source = tmp_path / "source"
    metrics = {"latent_mse_h1": 0.2, "latent_mse_h5": 0.5,
               "persistence_latent_mse_h1": 0.4, "persistence_latent_mse_h5": 1.5,
               "reward_mae": 0.3}
    write(source / "prediction_summary.json", {"task": "predict", "metrics": metrics})
    before = snapshot(source)
    assert measure({"source_output_dir": str(source), "output_dir": str(tmp_path / "evaluation")}, {}) == {
        "tdmpc2_" + key: value for key, value in metrics.items()}
    assert snapshot(source) == before


@pytest.mark.parametrize("task", [None, "train", "policy_evaluate", "predict"])
def test_missing_required_source_file_is_an_error(tmp_path, task):
    result = {"source_output_dir": str(tmp_path / "source"),
              "output_dir": str(tmp_path / "evaluation"), "data": {"task": task}}
    with pytest.raises(FileNotFoundError):
        measure(result, {})
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize("child", ["", "nested-evaluation"])
def test_refuses_evaluation_output_inside_source(tmp_path, child):
    source = tmp_path / "source"
    write(source / "training.json", training())
    before = snapshot(source)
    with pytest.raises(ValueError, match="outside the source"):
        measure({"source_output_dir": str(source), "output_dir": str(source / child)}, {})
    assert snapshot(source) == before


@pytest.mark.parametrize("field,value", [("success_once", "false"), ("steps", 0),
                                         ("action_intervention_steps", 3), ("return", float("nan"))])
def test_invalid_policy_evidence_cannot_be_reported_as_valid_scores(tmp_path, field, value):
    source = tmp_path / "source"
    data = policy()
    data["episodes"][0][field] = value
    write(source / "evaluation.json", data)
    with pytest.raises(ValueError):
        measure({"source_output_dir": str(source), "output_dir": str(tmp_path / "evaluation")}, {})


def test_missing_training_loss_is_not_filled_with_zero(tmp_path):
    source = tmp_path / "source"
    data = training()
    del data["last_losses"]["reward_loss"]
    write(source / "training.json", data)
    with pytest.raises(KeyError, match="reward_loss"):
        measure({"source_output_dir": str(source), "output_dir": str(tmp_path / "evaluation")}, {})


def test_explicit_source_task_selects_metadata_and_ambiguous_input_fails(tmp_path):
    source = tmp_path / "source"
    write(source / "training.json", training())
    write(source / "evaluation.json", policy())
    result = {"source_output_dir": str(source), "output_dir": str(tmp_path / "evaluation")}
    with pytest.raises(ValueError, match="Ambiguous"):
        measure(result, {})
    result["data"] = {"task": "policy_evaluate"}
    assert measure(result, {})["tdmpc2_episode_count"] == 2


def test_metadata_filename_and_declared_task_must_agree(tmp_path):
    source = tmp_path / "source"
    write(source / "training.json", policy())
    with pytest.raises(ValueError, match="filename"):
        measure({"source_output_dir": str(source), "output_dir": str(tmp_path / "evaluation")}, {})


def test_center_metrics_use_full_cohort_and_preserve_protocol(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'evaluation'
    data = policy()
    data['env_id'] = 'WMS-PushCubeCenter-v1'
    data['protocol'] = {'reward_version': 'center-dense/1',
                        'success_protocol': {'task_success': 'center_success'}}
    for episode, distance, center, stable in zip(data['episodes'], [.01, .04], [True, False], [False, False]):
        episode.update(final_goal_distance_m=distance, min_goal_distance_m=.005,
                       center_success_at_end=center, center_stable_last_5=stable,
                       region_success_at_end=True)
    write(source / 'evaluation.json', data)
    scores = measure({'source_output_dir': str(source), 'output_dir': str(output)}, {})
    assert scores['tdmpc2_final_goal_distance_m_mean'] == .025
    assert scores['tdmpc2_min_goal_distance_m_mean'] == .005
    assert scores['tdmpc2_center_success_at_end_rate'] == .5
    assert scores['tdmpc2_center_stable_last_5_rate'] == 0
    assert scores['tdmpc2_region_success_at_end_rate'] == 1
    saved = json.loads((output / 'tdmpc2_metrics.json').read_text())
    assert saved['details']['env_id'] == 'WMS-PushCubeCenter-v1'
    assert saved['details']['reward_version'] == 'center-dense/1'
    assert saved['details']['success_protocol']['task_success'] == 'center_success'


@pytest.mark.parametrize('field', ['final_goal_distance_m', 'min_goal_distance_m',
                                  'center_success_at_end', 'center_stable_last_5',
                                  'region_success_at_end'])
def test_partial_geometry_is_never_filled_with_zero_or_compared_as_full_cohort(tmp_path, field):
    source = tmp_path / 'source'
    data = policy()
    data['episodes'][0][field] = .01 if field.endswith('_m') else True
    write(source / 'evaluation.json', data)
    scores = measure({'source_output_dir': str(source), 'output_dir': str(tmp_path / 'evaluation')}, {})
    assert not any(field in name for name in scores)
