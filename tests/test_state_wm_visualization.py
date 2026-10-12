"""Numerical alignment checks; no simulator, GPU, renderer or training required.

NumPy belongs to the native experiment environment, not the platform core.
The CPU-only core test suite therefore skips these checks when it is absent.
"""
import pytest

np = pytest.importorskip("numpy")

from wms.engines.state_wm_maniskill.visualize import align_episode


@pytest.fixture
def episode_arrays():
    """A held-out episode plus a training episode, with distinguishable seeds."""
    length, horizon, dims = 7, 3, 24
    path = np.arange(length + 1, dtype=float)[:, None] * np.ones((1, dims))
    unrelated = path + 100
    data = {
        "states": np.concatenate([unrelated[:-1], path[:-1]]),
        "next_states": np.concatenate([unrelated[1:], path[1:]]),
        "episode_ids": np.repeat([2, 7], length),
        "steps": np.tile(np.arange(length), 2),
    }
    starts = np.arange(length - horizon + 1)
    truth = np.stack([path[start + 1:start + horizon + 1] for start in starts])
    predictions = {
        "truth": path[1:].copy(),
        "initial": path[:-1].copy(),
        "episode_ids": np.full(length, 7),
        "steps": np.arange(length),
        "rollout_truth": truth,
        "rollout_initial": path[starts].copy(),
        "rollout_episode_ids": np.full(len(starts), 7),
        "rollout_steps": starts,
    }
    for name, delta in [("action_conditioned", .001), ("no_action", .002)]:
        predictions[name + "_one_step"] = np.stack([
            predictions["truth"] + delta, predictions["truth"] + 10 * delta,
        ])
        predictions[name + "_rollout"] = np.stack([
            truth + delta, truth + 10 * delta,
        ])
    return data, predictions, path


def test_one_step_predictions_align_to_post_action_video_frame(episode_arrays):
    data, predictions, path = episode_arrays
    aligned = align_episode(data, predictions, 7, seed_index=0, horizon=1, fps=20)
    np.testing.assert_array_equal(aligned["truth"], path)
    np.testing.assert_allclose(aligned["time_s"], np.arange(8) / 20)
    np.testing.assert_array_equal(aligned["valid_frames"], np.arange(1, 8))
    np.testing.assert_array_equal(aligned["start_frames"], np.arange(7))
    for name, delta in [("action_conditioned", .001), ("no_action", .002)]:
        assert np.isnan(aligned["forecasts"][name][0]).all()
        np.testing.assert_allclose(aligned["forecasts"][name][1:], path[1:] + delta)
        for body in ("cube", "tcp"):
            assert np.isnan(aligned["errors_mm"][name][body][0])
            np.testing.assert_allclose(aligned["errors_mm"][name][body][1:], np.sqrt(3) * delta * 1000)


def test_rollout_targets_t_plus_h_and_persistence_holds_start_state(episode_arrays):
    data, predictions, path = episode_arrays
    aligned = align_episode(data, predictions, 7, seed_index=0, horizon=3, fps=20)
    np.testing.assert_array_equal(aligned["valid_frames"], np.arange(3, 8))
    np.testing.assert_array_equal(aligned["start_frames"], np.arange(5))
    assert np.isnan(aligned["forecasts"]["persistence"][:3]).all()
    # Three-step persistence is x[t], not x[t+2] or the contemporaneous truth.
    np.testing.assert_array_equal(aligned["forecasts"]["persistence"][3:], path[:-3])
    np.testing.assert_allclose(aligned["forecasts"]["action_conditioned"][3:], path[3:] + .001)
    np.testing.assert_allclose(aligned["errors_mm"]["persistence"]["cube"][3:], 3 * np.sqrt(3) * 1000)


def test_explicit_training_seed_is_not_ensemble_mean(episode_arrays):
    data, predictions, path = episode_arrays
    aligned = align_episode(data, predictions, 7, seed_index=1, horizon=3, fps=20)
    np.testing.assert_allclose(aligned["forecasts"]["action_conditioned"][3:], path[3:] + .01)
    assert not np.allclose(aligned["forecasts"]["action_conditioned"][3:], path[3:] + .0055)


@pytest.mark.parametrize("key", ["truth", "initial", "rollout_truth", "rollout_initial"])
def test_rejects_predictions_from_a_different_simulation(episode_arrays, key):
    data, predictions, _ = episode_arrays
    predictions[key] = predictions[key].copy() + 1
    with pytest.raises(ValueError):
        align_episode(data, predictions, 7, seed_index=0,
                      horizon=3 if key.startswith("rollout") else 1, fps=20)


def test_training_episode_is_not_silently_presented_as_held_out(episode_arrays):
    data, predictions, _ = episode_arrays
    with pytest.raises(ValueError):
        align_episode(data, predictions, 2, seed_index=0, horizon=1, fps=20)


def test_rejects_missing_control_step_in_video_alignment(episode_arrays):
    data, predictions, _ = episode_arrays
    data["steps"][-1] += 1
    with pytest.raises(ValueError):
        align_episode(data, predictions, 7, seed_index=0, horizon=1, fps=20)
