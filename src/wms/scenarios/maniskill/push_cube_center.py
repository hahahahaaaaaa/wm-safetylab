"""Independent center-precision task; original PushCube-v1 stays unchanged."""
import torch
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env

TASK_ID = "WMS-PushCubeCenter-v1"
REWARD_VERSION = "center-dense/1"


@register_env(TASK_ID, max_episode_steps=50)
class PushCubeCenterEnv(PushCubeEnv):
    """Same 20 cm initial displacement, robot, state and controller as PushCube.

    goal_radius remains .1 for the original scene and its initialization.
    Center precision is a separate criterion, never a change to that property.
    Rewards remain distance-sensitive inside both the 10 cm and 2 cm regions.
    """
    center_tolerance_m = 0.02
    table_height_tolerance_m = 0.005

    def evaluate(self):
        original = super().evaluate()["success"]
        distance = torch.linalg.vector_norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], dim=-1)
        on_table = (self.obj.pose.p[..., 2] - self.cube_half_size).abs() <= self.table_height_tolerance_m
        center = (distance <= self.center_tolerance_m) & on_table
        return {"success": center, "center_success": center,
                "original_region_success": original, "goal_distance_m": distance,
                "on_table": on_table}

    def compute_dense_reward(self, obs, action, info):
        push_point = self.obj.pose.p + torch.tensor(
            [-self.cube_half_size - 0.005, 0, 0], device=self.device)
        reach_distance = torch.linalg.vector_norm(push_point - self.agent.tcp.pose.p, dim=-1)
        reach = 1 - torch.tanh(5 * reach_distance)
        distance = info["goal_distance_m"]
        coarse = 1 - torch.tanh(5 * distance)
        precision = 1 - torch.tanh(40 * distance)
        # Never set reward to a constant when success=True: approaching the
        # actual center must remain better than resting at the tolerance edge.
        normalized = 0.2 * reach + info["on_table"].float() * (0.3 * coarse + 0.5 * precision)
        return 4 * normalized

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4
