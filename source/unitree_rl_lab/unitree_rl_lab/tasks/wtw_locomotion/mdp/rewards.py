from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from unitree_rl_lab.tasks.wtw_locomotion.sensors import GaitSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)

def action_smoothness(
    env: ManagerBasedRLEnv,
    action_name: str = "JointPositionAction",
) -> torch.Tensor:
    """Penalize first-order joint position target changes.

    Returns sum over joint dimensions of (target_t - target_{t-1})^2.

    This computes smoothness on the actual joint position targets sent to the robot.
    For JointPositionAction: target = offset + scale * action
    
    This version directly computes prev_processed_actions from env.action_manager.prev_action
    without needing to cache, which is more elegant and avoids state management.
    
    Args:
        env: The environment.
        action_name: The name of the action term to use (default: "JointPositionAction").
    """
    # Get the action term using public API
    action_term = env.action_manager.get_term(action_name)
    
    # Current joint position target = processed_action
    # For JointPositionAction: processed_action = offset + scale * raw_action
    cur_target = action_term.processed_actions
    
    # Compute previous joint position target from prev_action
    # Access protected attributes (they exist at runtime, type checker just doesn't see them)
    prev_raw_action = env.action_manager.prev_action
    scale = getattr(action_term, "_scale", 1.0)  # type: ignore
    offset = getattr(action_term, "_offset", 0.0)  # type: ignore
    prev_target = offset + scale * prev_raw_action
    diff = torch.square(cur_target - prev_target)
    # ignore first step where prev is zeroW
    diff = diff * (prev_raw_action != 0.0)
    # Compute smoothness penalty
    reward = torch.sum(diff, dim=1)
    
    return reward

"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, 
    velocity_threshold: float, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero and robot is not moving.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    mask = (command_norm < 0.1) & (body_vel < velocity_threshold)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * mask.float()


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def compute_gait_parameters(
    env: ManagerBasedRLEnv,
    frequency_command_name: str = "gait_frequency",
    phase_command_name: str = "gait_phase",
    offset_command_name: str = "gait_offset",
    bound_command_name: str = "gait_bound",
    duration_command_name: str = "gait_duration",
) -> torch.Tensor:
    """
    Compute all gait parameters including gait indices, foot indices, clock inputs, and desired contact states.
    
    This function creates and manages gait state directly in the environment, without relying on GaitSensor.
    The gait state is stored in env attributes and persists across calls.
    
    Args:
        env: The environment
        frequency_command_name: Name of frequency command (default: "gait_frequency")
        phase_command_name: Name of phase command (default: "gait_phase")
        offset_command_name: Name of offset command (default: "gait_offset")
        bound_command_name: Name of bound command (default: "gait_bound")
        duration_command_name: Name of duration command (default: "gait_duration")
    
    Returns:
        Dictionary containing:
            - gait_indices: [num_envs] - Main gait phase [0, 1)
            - foot_indices: [num_envs, 4] - Per-foot phase indices
            - clock_inputs: [num_envs, 4] - sin(2π * foot_indices)
            - doubletime_clock_inputs: [num_envs, 4] - sin(4π * foot_indices)
            - halftime_clock_inputs: [num_envs, 4] - sin(π * foot_indices)
            - desired_contact_states: [num_envs, 4] - Smoothed contact states [0, 1]
    """
    
    n = env.num_envs
    device = env.device
    dt = env.step_dt
    
    # === Initialize gait_indices if not exists ===
    if not hasattr(env, "_gait_indices"):
        env._gait_indices = torch.zeros(n, device=device)  # type: ignore
    if not hasattr(env, "_foot_indices"):
        env._foot_indices = torch.zeros(n, 4, device=device)  # type: ignore
    if not hasattr(env, "_desired_contact_states"):
        env._desired_contact_states = torch.zeros(n, 4, device=device)  # type: ignore

    # === 1. Extract gait command parameters ===
    # Use default values if commands don't exist
    try:
        frequencies = env.command_manager.get_command(frequency_command_name).squeeze(-1)
    except:
        frequencies = torch.full((n,), 3.0, device=device)
    
    try:
        phases = env.command_manager.get_command(phase_command_name).squeeze(-1)
    except:
        phases = torch.full((n,), 0.5, device=device)
    
    try:
        offsets = env.command_manager.get_command(offset_command_name).squeeze(-1)
    except:
        offsets = torch.full((n,), 0.0, device=device)
    
    try:
        bounds = env.command_manager.get_command(bound_command_name).squeeze(-1)
    except:
        bounds = torch.full((n,), 0.0, device=device)
    
    try:
        durations = env.command_manager.get_command(duration_command_name).squeeze(-1)
    except:
        durations = torch.full((n,), 0.5, device=device)
    
    # === 2. Update gait indices ===
    env._gait_indices = torch.remainder(  # type: ignore
        env._gait_indices + dt * frequencies,  # type: ignore
        1.0
    )
    gait_indices = env._gait_indices  # type: ignore
    
    # === 3. Calculate foot indices (4 legs) [FL, FR, RL, RR] ===
    fi_list = [
        gait_indices + offsets,                                    # FL
        gait_indices + phases + offsets + bounds,                  # FR
        gait_indices + phases,                                     # RL
        gait_indices + bounds                                      # RR
    ]
    foot_indices = torch.remainder(torch.stack(fi_list, dim=1), 1.0)
    env._foot_indices = foot_indices  # type: ignore
    
    # === 4. Remap stance/swing phases using durations ===
    foot_indices_warped = foot_indices.clone()
    for j in range(4):
        phase_j = foot_indices[:, j]
        stance_mask = phase_j < durations
        swing_mask = phase_j > durations
        # inverse
        # stance: [0, durations) -> [0, 0.5)
        foot_indices_warped[stance_mask, j] = phase_j[stance_mask] * (0.5 / durations[stance_mask])
        # swing: (durations, 1) -> [0.5, 1)
        foot_indices_warped[swing_mask, j] = 0.5 + (phase_j[swing_mask] - durations[swing_mask]) * (
            0.5 / (1 - durations[swing_mask])
        )
    
    # === 5. Generate clock signals (sin inputs) ===
    clock_inputs = torch.sin(2 * torch.pi * foot_indices_warped)
    doubletime_clock_inputs = torch.sin(4 * torch.pi * foot_indices_warped)
    halftime_clock_inputs = torch.sin(torch.pi * foot_indices_warped)
    
    # === 6. Smooth desired contact states ===
    kappa = 0.07
    smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf
    
    def smoothing_multiplier(col_tensor: torch.Tensor) -> torch.Tensor:
        base = torch.remainder(col_tensor, 1.0)
        term1 = smoothing_cdf_start(base) * (1 - smoothing_cdf_start(base - 0.5))
        term2 = smoothing_cdf_start(base - 1.0) * (1 - smoothing_cdf_start(base - 0.5 - 1.0))
        return term1 + term2
    
    smoothing_FL = smoothing_multiplier(foot_indices_warped[:, 1])
    smoothing_FR = smoothing_multiplier(foot_indices_warped[:, 0])
    smoothing_RL = smoothing_multiplier(foot_indices_warped[:, 3])
    smoothing_RR = smoothing_multiplier(foot_indices_warped[:, 2])
    env._desired_contact_states = torch.stack([smoothing_FL, smoothing_FR, smoothing_RL, smoothing_RR], dim=1) # type: ignore

    return torch.tensor(0.0, device=env.device)


def tracking_contacts_shaped_force(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Reward that penalizes foot forces on swing legs and shapes contact forces.

    Args:
        env: the environment
        gait_sensor_cfg: gait sensor config
        contact_sensor_cfg: contact sensor config
    """
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    # 1. 获得每条腿的总力（norm）
    foot_forces = torch.norm(
        contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :], dim=-1
    )  # [num_envs, num_feet]

    # 2. 获得期望接触状态
    desired_contact = env._desired_contact_states  # type: ignore 

    # 3. 惩罚 swing leg 的接触力，reward 越大越好（非支撑腿力越小）
    gait_force_sigma = 100.0
    swing_mask = 1.0 - desired_contact  # 1: swing leg, 0: stance leg
    reward_per_leg = -swing_mask * (1.0 - torch.exp(-foot_forces**2 / gait_force_sigma))  # [num_envs, 4]

    # 4. 每条腿平均
    reward = torch.mean(reward_per_leg, dim=1)  # [num_envs]
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1

    return reward

def tracking_contacts_shaped_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Penalize foot velocities on stance legs. Swing legs are not penalized.

    Args:
        env: environment
        gait_sensor_cfg: gait sensor config
        asset_cfg: robot asset config

    Returns:
        reward: Tensor of shape [num_envs]
    """

    asset: RigidObject = env.scene[asset_cfg.name]

    # 1. 脚速度 L2 norm
    foot_velocities = torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1
    )  # [num_envs, num_feet]

    # 2. 支撑腿 mask
    desired_contact = env._desired_contact_states  # [num_envs, 4]
    stance_mask = desired_contact  # 1: stance leg, 0: swing leg

    # 3. 奖励/惩罚公式（速度越大惩罚越大）
    gait_vel_sigma = 10.0
    reward_per_leg = -stance_mask * (1 - torch.exp(-foot_velocities**2 / gait_vel_sigma))  # [num_envs, 4]

    # 4. 对四条腿平均
    reward = torch.mean(reward_per_leg, dim=1)  # [num_envs]
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1

    return reward



def feet_clearance_cmd_linear(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ray_sensor_cfg: SceneEntityCfg | None = None,
    foot_height_command_name: str = "foot_height",
    velocity_command_name: str = "base_velocity",
) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]
    
    if ray_sensor_cfg is not None:
        sensor: RayCaster = env.scene[ray_sensor_cfg.name]
        # Adjust the target height using the sensor data
        ground_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)
    else:
        # Use the provided target height directly for flat terrain
        ground_height = 0

    threshold = 0.7
    phases = torch.clamp(env._desired_contact_states - threshold, min=0) / (1 - threshold) # type: ignore
    foot_height = (asset.data.body_pos_w[:, asset_cfg.body_ids, 2]).view(env.num_envs, -1)
    
    target_height = env.command_manager.get_command(foot_height_command_name).squeeze(-1)  # [num_envs]
    target_height = target_height.unsqueeze(-1)  # [num_envs, 1] for broadcasting with phases [num_envs, 4]
    target_foot_height = target_height * phases + ground_height + 0.023
    # print(f"foot_height for env 0: {foot_height[0]}")
    # print(f"target_foot_height for env 0: {target_foot_height[0]}")
    rew_foot_clearance = torch.square(target_foot_height - foot_height)
    reward = torch.sum(rew_foot_clearance, dim=1)
    # 当没有速度指令时不抬腿
    reward *= torch.linalg.norm(env.command_manager.get_command(velocity_command_name), dim=1) > 0.1
    return reward

def raibert_heuristic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    velocity_command_name: str = "base_velocity",
    frequency_command_name: str = "gait_frequency",
    stance_width_command_name: str = "stand_width",
    stance_length_command_name: str = "stand_length"
) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    
    root_yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            root_yaw_quat, cur_footpos_translated[:, i, :]
        )
        
    # nominal positions: [FL, FR, RL, RR]
    # desired_stance_width = 0.35
    desired_stance_width = env.command_manager.get_command(stance_width_command_name).squeeze(-1)
    # Create nominal y positions for each environment: [num_envs, 4]
    desired_ys_nom = torch.stack([
        desired_stance_width / 2,
        -desired_stance_width / 2,
        desired_stance_width / 2,
        -desired_stance_width / 2
    ], dim=-1)  # [num_envs, 4]

    # desired_stance_length = 0.45
    desired_stance_length = env.command_manager.get_command(stance_length_command_name).squeeze(-1)
    # Create nominal x positions for each environment: [num_envs, 4]
    desired_xs_nom = torch.stack([
        desired_stance_length / 2,
        desired_stance_length / 2,
        -desired_stance_length / 2,
        -desired_stance_length / 2
    ], dim=-1)  # [num_envs, 4]

    # raibert offsets
    cmd_frequencies = env.command_manager.get_command(frequency_command_name).squeeze(-1)  # [batch_size]
    
    # Calculate phases for all legs
    phases = (torch.abs(1.0 - (env._foot_indices * 2.0)) * 1.0 - 0.5).unsqueeze(-1)  # [batch_size, 4, 1]
    
    x_vel_des = env.command_manager.get_command(velocity_command_name)[:, 0:1]  # [batch_size, 1]
    yaw_vel_des = env.command_manager.get_command(velocity_command_name)[:, 2:3]  # [batch_size, 1]
    y_vel_des = yaw_vel_des * desired_stance_length.unsqueeze(-1) / 2  # [batch_size, 1]
    
    # Reshape velocities and frequencies for broadcasting
    x_vel_des = x_vel_des.unsqueeze(1)  # [batch_size, 1, 1]
    y_vel_des = y_vel_des.unsqueeze(1)  # [batch_size, 1, 1]
    cmd_frequencies = cmd_frequencies.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
    
    # Calculate offsets (phases is already [batch_size, 4, 1])
    desired_ys_offset = (phases * y_vel_des * (0.5 / cmd_frequencies)).squeeze(-1)  # [batch_size, 4]
    desired_ys_offset[:, 2:4] *= -1
    desired_xs_offset = (phases * x_vel_des * (0.5 / cmd_frequencies)).squeeze(-1)  # [batch_size, 4]

    desired_ys_nom = desired_ys_nom + desired_ys_offset
    desired_xs_nom = desired_xs_nom + desired_xs_offset

    desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

    err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footpos_in_body_frame[:, :, 0:2])

    # print(f"desired_footsteps_body_frame: {desired_footsteps_body_frame[0, :, :]}")
    # print(f"footpos_in_body_frame: {footpos_in_body_frame[0, :, :]}")

    reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
    return reward

"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
