"""Task identity and training recording boundaries without native dependencies."""
import pytest
from wms.engines.tdmpc2.backend import _check_warm_start_config, _recording_starts


def _config(env='PushCube-v1'):
    return dict(env_id=env, control_mode='pd_joint_delta_pos', obs_shape={'state':[35]}, action_dim=8, model_size=5)


def test_legacy_warm_start_stays_with_its_original_reward():
    old=_config()
    _check_warm_start_config(old,{**old,'reward_version':'maniskill-normalized-dense'})
    with pytest.raises(ValueError,match='env_id'):
        _check_warm_start_config(old,{**_config('WMS-PushCubeCenter-v1'),'reward_version':'center-dense/1'})
    with pytest.raises(ValueError,match='reward_version'):
        _check_warm_start_config({**_config('WMS-PushCubeCenter-v1'),'reward_version':'center-dense/0'},
                                {**_config('WMS-PushCubeCenter-v1'),'reward_version':'center-dense/1'})


def test_recordings_are_actual_nonoverlapping_existing_episode_batches():
    starts=_recording_starts(204800,1600)
    assert starts==[0,51200,102400,153600,203200]
    assert all(x%1600==0 and x+1600<=204800 for x in starts)
    assert _recording_starts(1600,1600)==[0]
    with pytest.raises(ValueError):
        _recording_starts(204801,1600)
