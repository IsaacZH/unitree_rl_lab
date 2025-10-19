from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from weakref import WeakKeyDictionary
from typing import cast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv





def feet_contact_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ground_height: float = 0.0,
    near_thresh: float = 0.03,
) -> torch.Tensor:
    """Penalize feet velocity when feet are near the ground (contact vicinity).
    """
    # Use asset body ids (feet) for indexing
    asset: Articulation = env.scene[asset_cfg.name]
    # world positions / velocities of feet
    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_lin_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    near_ground = (feet_pos_z - ground_height) < near_thresh
    vel_sq = torch.sum(torch.square(feet_lin_vel), dim=-1)  # (N, n_feet)
    reward = torch.sum(near_ground * vel_sq, dim=1)
    return reward


def feet_impact_vel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize vertical impact velocity at contact.

    Approximates WTW's `_reward_feet_impact_vel` by using previous-step vertical velocity
    and current contact detection from the contact sensor.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = cast(list[int], asset_cfg.body_ids)
    if body_ids is None or len(body_ids) == 0:
        raise RuntimeError("feet_impact_vel: asset_cfg.body_ids is not resolved. Provide body_names for feet.")

    # Current contact state (per env, per foot): use max over force-history at current step window
    net_forces_hist = contact_sensor.data.net_forces_w_history
    contacts = (
        net_forces_hist[:, :, body_ids, :].norm(dim=-1).max(dim=1)[0] > contact_threshold
    )  # (N, n_feet) boolean

    # Cache previous vertical foot velocity (world frame)
    vz_now = asset.data.body_lin_vel_w[:, body_ids, 2]
    state = _get_state(env)
    prev_vz = state.get("prev_feet_vz")
    if prev_vz is None or prev_vz.shape != vz_now.shape:
        prev_vz = torch.zeros_like(vz_now)
        state["prev_feet_vz"] = prev_vz.clone()

    vz_prev = prev_vz
    # Penalize downward (negative) velocities at the moment of contact
    neg_vz_prev = torch.clamp(vz_prev, max=0.0)
    per_foot_penalty = contacts * torch.square(neg_vz_prev)
    reward = torch.sum(per_foot_penalty, dim=1)

    # Update cache and clear on terminated envs
    state["prev_feet_vz"] = vz_now.clone()
    if hasattr(env, "termination_manager") and hasattr(env.termination_manager, "terminated"):
        term_mask = env.termination_manager.terminated
        if term_mask is not None:
            buf = state.get("prev_feet_vz")
            if buf is not None:
                buf[term_mask] = 0.0
                state["prev_feet_vz"] = buf

    return reward


