"""Official PushCube deployment rollout. No perturbations or safety experiment.

The simulator is imported only inside this native process. The official planner
uses privileged simulator state; this is not a learned WM or a visual policy.
"""
import importlib.metadata
import os
import time
from pathlib import Path


def rollout(config, case, seed, output, worker):
    from wms.engines.maniskill.runtime import native_environment
    if (case.get('task'), case.get('robot'), case.get('controller'),
            case.get('physics_backend'), case.get('executor')) != (
            'PushCube-v1', 'panda', 'pd_joint_pos', 'physx_cpu', 'official_motion_planner'):
        raise ValueError('Unsupported ManiSkill deployment recipe')
    os.environ.update(native_environment(config))
    started = time.perf_counter()
    import numpy as np
    import torch
    import gymnasium as gym
    import imageio.v2 as imageio
    import mani_skill.envs
    from mani_skill.examples.motionplanning.panda.solutions import push_cube
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output)
    worker.event('progress', {'phase': 'creating_official_environment', 'task': 'PushCube-v1'})
    default_limit = gym.spec('PushCube-v1').max_episode_steps
    env = gym.make('PushCube-v1', robot_uids='panda', num_envs=1,
                   obs_mode='rgb', control_mode='pd_joint_pos',
                   render_mode='rgb_array', sim_backend='physx_cpu', render_backend='gpu',
                   max_episode_steps=int(case['max_steps']))
    frames, rows = [], []
    initial = {}
    native = env.unwrapped
    control_freq = float(native.control_freq)
    limit = env.get_wrapper_attr('_max_episode_steps')
    if type(limit) is not int or limit < 1:
        env.close()
        raise RuntimeError('The native episode limit could not be verified')
    stop_reason = None

    def array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if not np.isfinite(value).all():
            raise ValueError('Non-finite simulator signal')
        return value

    def scalar_bool(value):
        value = array(value).reshape(-1)
        if value.size != 1:
            raise ValueError('Deployment recipe requires exactly one environment')
        return bool(value[0])

    def state():
        return {'qpos': array(native.agent.robot.get_qpos()).reshape(-1).tolist(),
                'qvel': array(native.agent.robot.get_qvel()).reshape(-1).tolist(),
                'cube_pose': array(native.obj.pose.raw_pose).reshape(-1).tolist()}

    def capture(observation):
        rgb = array(observation['sensor_data']['base_camera']['rgb'])
        if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected one uint8 RGB observation from base_camera')
        frames.append(rgb[0].copy())

    class EpisodeFinished(Exception):
        pass

    class Recorder(gym.Wrapper):
        def reset(self, **kwargs):
            if frames or rows:
                raise ValueError('Deployment does not retry or filter unsuccessful episodes')
            worker.check_cancelled()
            observation, info = self.env.reset(**kwargs)
            initial.update(state())
            initial['goal_position'] = array(native.goal_region.pose.p).reshape(-1).tolist()
            capture(observation)
            return observation, info

        def step(self, action):
            nonlocal stop_reason
            worker.check_cancelled()
            if rows and (rows[-1]['terminated'] or rows[-1]['truncated']):
                stop_reason = 'terminated' if rows[-1]['terminated'] else 'truncated'
                raise EpisodeFinished()
            if len(rows) >= case['max_steps']:
                stop_reason = 'deployment_step_budget'
                raise EpisodeFinished()
            command = array(action).reshape(-1).tolist()
            if len(command) != 8:
                raise ValueError('Panda pd_joint_pos requires seven arm targets and one gripper command')
            observation, reward, terminated, truncated, info = self.env.step(action)
            rows.append({'step': len(rows), 't': (len(rows) + 1) / control_freq,
                         'command': command, **state(),
                         'success': scalar_bool(info['success']),
                         'terminated': scalar_bool(terminated), 'truncated': scalar_bool(truncated)})
            capture(observation)
            if len(rows) % 10 == 0:
                worker.event('progress', {'phase': 'rollout', 'steps': len(rows),
                                          'task_success': rows[-1]['success']})
            return observation, reward, terminated, truncated, info

    recorded = Recorder(env)
    planner_status = 'completed'
    try:
        worker.event('progress', {'phase': 'official_motion_planner', 'seed': seed})
        try:
            result = push_cube.solve(recorded, seed=seed, debug=False, vis=False)
            if isinstance(result, (int, np.integer)) and result == -1:
                planner_status = 'no_plan'
        except EpisodeFinished:
            planner_status = 'episode_ended'
        if not rows or len(frames) != len(rows) + 1:
            raise ValueError('No complete simulator rollout was captured')
        worker.check_cancelled()
        video = output / 'video.mp4'
        with imageio.get_writer(video, fps=control_freq, codec='libx264',
                                quality=8, macro_block_size=1) as writer:
            for frame in frames:
                worker.check_cancelled()
                writer.append_data(frame)
        trajectory = {'schema_version': 'maniskill.trajectory/1.0',
                      'task': 'PushCube-v1', 'robot': 'panda', 'seed': seed,
                      'initial_state': initial, 'control_freq': control_freq,
                      'command_schema': {'arm': {'size': 7, 'unit': 'rad', 'type': 'absolute_joint_target'},
                                         'gripper': {'size': 1, 'type': 'normalized_controller_command'}},
                      'measurement_schema': {'qpos': '7 arm angles + 2 finger positions',
                                             'cube_pose': 'world position xyz and quaternion wxyz'},
                      'video_alignment': 'frame 0 = reset; frame step+1 = post-action observation',
                      'steps': rows}
        report = {'schema_version': 'maniskill.runtime/1.0',
                  'task': 'PushCube-v1', 'robot': 'panda', 'controller': 'pd_joint_pos',
                  'physics_backend': native.backend.sim_backend,
                  'render_backend': native.backend.render_backend,
                  'render_device': str(native.backend.render_device),
                  'executor': 'official_motion_planner', 'planner_status': planner_status,
                  'step_count': len(rows), 'task_success': rows[-1]['success'],
                  'success_once': any(row['success'] for row in rows),
                  'end_reason': stop_reason or ('terminated' if rows[-1]['terminated'] else
                                 'truncated' if rows[-1]['truncated'] else 'planner_completed'),
                  'default_episode_limit': default_limit, 'effective_episode_limit': limit,
                  'requested_step_budget': case['max_steps'], 'control_freq': control_freq,
                  'vulkan_icd': os.environ.get('VK_ICD_FILENAMES'),
                  'simulation_freq': float(native.sim_freq),
                  'observation_camera': 'base_camera', 'video_frame_count': len(frames),
                  'video_width': int(frames[0].shape[1]), 'video_height': int(frames[0].shape[0]),
                  'seed': seed, 'elapsed_seconds': time.perf_counter() - started,
                  'versions': {name: importlib.metadata.version(name) for name in
                               ('mani-skill', 'sapien', 'torch', 'numpy', 'gymnasium', 'mplib')},
                  'claim_limit': 'Deployment and official task execution only; no safety evaluation, '
                                 'perturbation, learned world model, probability study or policy training.'}
        inputs = {'schema_version': 'maniskill.input/1.0', 'case': case, 'seed': seed,
                  'initial_state': initial, 'observation_camera': 'base_camera',
                  'controller_uses_privileged_state': True, 'interventions': [],
                  'episode_end_policy': 'Stop before the next action after native terminated/truncated'}
        return inputs, trajectory, report, video
    finally:
        env.close()
