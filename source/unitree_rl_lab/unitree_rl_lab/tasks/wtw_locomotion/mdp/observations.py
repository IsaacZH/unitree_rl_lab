from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase

def clock_inputs(
    env: ManagerBasedRLEnv, frequency: float, phase: float, offset: float, bound: float, duration: float
) -> torch.Tensor:
    """
    Generates clock signals for each foot based on a central gait generator.

    This function replicates a common method for generating periodic signals for legged locomotion,
    often used to drive gait patterns in the policy.

    Args:
        env: The environment instance.
        frequency: The frequency of the gait cycle (in Hz).
        phase: The phase offset between pairs of legs (e.g., 0.5 for a trot).
        offset: A global phase offset for all legs.
        bound: An additional phase offset, often for specific leg pairs.
        duration: The duty factor for the stance phase, i.e., the proportion of the cycle
            a foot is on the ground.

    Returns:
        A tensor of shape (num_envs, 4) containing the sine-encoded clock signal for each of the
        four feet [FL, FR, RL, RR].
    """
    # 1. 初始化或获取步态索引
    # gait_index 是一个在 [0, 1) 范围内循环的全局相位
    if not hasattr(env, "gait_index"):
        env.gait_index = torch.zeros(env.num_envs, device=env.device)

    # 2. 更新步态索引
    env.gait_index += env.step_dt * frequency
    env.gait_index = torch.fmod(env.gait_index, 1.0)

    # 3. 计算每条腿的相位索引 (FL, FR, RL, RR)
    foot_indices = torch.zeros(env.num_envs, 4, device=env.device)
    foot_indices[:, 0] = torch.fmod(env.gait_index + offset, 1.0)
    foot_indices[:, 1] = torch.fmod(env.gait_index + phase + offset + bound, 1.0)
    foot_indices[:, 2] = torch.fmod(env.gait_index + phase, 1.0)
    foot_indices[:, 3] = torch.fmod(env.gait_index + bound, 1.0)

    # 4. 根据支撑相占空比(duration)重新映射相位
    # 这个过程将相位分为支撑相和摆动相，并将它们分别映射到 [0, 0.5) 和 [0.5, 1)
    stance_mask = foot_indices < duration
    swing_mask = ~stance_mask

    # 支撑相: [0, duration) -> [0, 0.5)
    foot_indices[stance_mask] = foot_indices[stance_mask] * (0.5 / duration)
    # 摆动相: [duration, 1.0) -> [0.5, 1.0)
    foot_indices[swing_mask] = 0.5 + (foot_indices[swing_mask] - duration) * (0.5 / (1.0 - duration))

    # 5. 生成时钟信号 (sin 编码)
    return torch.sin(2.0 * torch.pi * foot_indices)