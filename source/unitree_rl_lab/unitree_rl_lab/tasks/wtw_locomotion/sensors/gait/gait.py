# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
import numpy as np

from isaaclab.sensors.sensor_base import SensorBase
from .gait_data import GaitSensorData

if TYPE_CHECKING:
    from .gait_cfg import GaitSensorCfg


class GaitSensor(SensorBase):
    """The Gait sensor.

    This sensor provides information about the gait of the robot, including foot positions, velocities, and other gait-related data.
    """

    cfg: GaitSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: GaitSensorCfg):
        """Initializes the Gait sensor.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = GaitSensorData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Gait sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self._view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._view.count}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> GaitSensorData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    # @property
    # def num_instances(self) -> int:
    #     return self._view.count

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        idx = slice(None) if env_ids is None else env_ids
        # reset accumulative data buffers
        self._data.gait_indices[idx] = 0.0

    def update(self, dt: float, force_recompute: bool = False):
        # save timestamp
        self._dt = dt
        # execute updating
        super().update(dt, force_recompute)

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        This function creates handles and registers the provided data types with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the imu prim is not a RigidBodyPrim
        """
        # Initialize parent class
        super()._initialize_impl()
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # check if the prim at path is a rigid prim
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # check if it is a RigidBody Prim
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
        else:
            raise RuntimeError(f"Failed to find a RigidBodyAPI for the prim paths: {self.cfg.prim_path}")

        # Create internal buffers
        self._initialize_buffers_impl()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # 规范化 env_ids：当传入的是完整集合时，转换为 slice(None)；当为空时直接返回
        idx = env_ids  # 默认使用原始 env_ids
        if isinstance(env_ids, slice):
            idx = env_ids
        else:
            # 传入的 env_ids 可能是 list/tuple/tensor
            if len(env_ids) == self._num_envs:
                idx = slice(None)
            elif len(env_ids) == 0:
                return
        
        # === 1. 提取 gait command 参数 ===
        n = self._num_envs
        device = self.device
        frequencies = torch.full((n,), 3.0, device=device)   # 频率（Hz）
        phases = torch.full((n,), 0.5, device=device)
        offsets = torch.full((n,), 0.0, device=device)
        bounds = torch.full((n,), 0.0, device=device)
        durations = torch.full((n,), 0.5, device=device)

        # === 2. 更新 gait index ===
        self._data.gait_indices[idx] = torch.remainder(
            self._data.gait_indices[idx] + self._sim_physics_dt * frequencies[idx],
            1.0
        )

        # === 3. 计算 foot indices（4 条腿）[FL, FR, RL, RR]===
        gait_idx_sel = self._data.gait_indices[idx]
        phases_sel = phases[idx]
        offsets_sel = offsets[idx]
        bounds_sel = bounds[idx]
        durations_sel = durations[idx]

        fi_list = [
            gait_idx_sel + offsets_sel,                                        # FL
            gait_idx_sel + phases_sel + offsets_sel + bounds_sel,              # FR
            gait_idx_sel + phases_sel,                                         # RL
            gait_idx_sel + bounds_sel                                          # RR
        ]
        # 形状 (m, 4)
        foot_indices_sel = torch.remainder(torch.stack(fi_list, dim=1), 1.0)
        self._data.foot_indices[idx] = foot_indices_sel

        # === 4. 按 durations warp 重新映射 stance/swing ===
        # foot_indices_sel ∈ [0, 1)，可直接使用
        for j in range(4):
            phase_j = foot_indices_sel[:, j]
            stance_mask = phase_j < durations_sel
            swing_mask = phase_j > durations_sel
            # stance: [0, durations) -> [0, 0.5)
            foot_indices_sel[stance_mask, j] = phase_j[stance_mask] * (0.5 / durations_sel[stance_mask])
            # swing: (durations, 1) -> [0.5, 1)
            foot_indices_sel[swing_mask, j] = 0.5 + (phase_j[swing_mask] - durations_sel[swing_mask]) * (
                0.5 / (1 - durations_sel[swing_mask])
            )
            

        # === 6. 生成时钟信号（sin 输入）===
        clock_inputs_sel = torch.sin(2 * np.pi * foot_indices_sel)
        doubletime_clock_inputs_sel = torch.sin(4 * np.pi * foot_indices_sel)
        halftime_clock_inputs_sel = torch.sin(1 * np.pi * foot_indices_sel)

        # === 7. 平滑化 desired contact states ===
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        def smoothing_multiplier(col_tensor: torch.Tensor) -> torch.Tensor:
            # 输入: (m,)
            base = torch.remainder(col_tensor, 1.0)
            term1 = smoothing_cdf_start(base) * (1 - smoothing_cdf_start(base - 0.5))
            term2 = smoothing_cdf_start(base - 1.0) * (1 - smoothing_cdf_start(base - 0.5 - 1.0))
            return term1 + term2

        smoothing_FL = smoothing_multiplier(foot_indices_sel[:, 0])
        smoothing_FR = smoothing_multiplier(foot_indices_sel[:, 1])
        smoothing_RL = smoothing_multiplier(foot_indices_sel[:, 2])
        smoothing_RR = smoothing_multiplier(foot_indices_sel[:, 3])
        desired_contact_states_sel = torch.stack([smoothing_FL, smoothing_FR, smoothing_RL, smoothing_RR], dim=1)

        # === 写回仅选择的 env ===
        self._data.clock_inputs[idx] = clock_inputs_sel
        self._data.doubletime_clock_inputs[idx] = doubletime_clock_inputs_sel
        self._data.halftime_clock_inputs[idx] = halftime_clock_inputs_sel
        self._data.desired_contact_states[idx] = desired_contact_states_sel


    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        n = self._num_envs
        device = self.device
        self._data.gait_indices = torch.zeros(n, device=device)
        self._data.foot_indices = torch.zeros(n, 4, device=device)
        self._data.clock_inputs = torch.zeros(n, 4, device=device)
        self._data.doubletime_clock_inputs = torch.zeros(n, 4, device=device)
        self._data.halftime_clock_inputs = torch.zeros(n, 4, device=device)
        self._data.desired_contact_states = torch.zeros(n, 4, device=device)


    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     # set visibility of markers
    #     # note: parent only deals with callbacks. not their visibility
    #     if debug_vis:
    #         # create markers if necessary for the first time
    #         if not hasattr(self, "acceleration_visualizer"):
    #             self.acceleration_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
    #         # set their visibility to true
    #         self.acceleration_visualizer.set_visibility(True)
    #     else:
    #         if hasattr(self, "acceleration_visualizer"):
    #             self.acceleration_visualizer.set_visibility(False)

    # def _debug_vis_callback(self, event):
    #     # safely return if view becomes invalid
    #     # note: this invalidity happens because of isaac sim view callbacks
    #     if self._view is None:
    #         return
    #     # get marker location
    #     # -- base state
    #     base_pos_w = self._data.pos_w.clone()
    #     base_pos_w[:, 2] += 0.5
    #     # -- resolve the scales
    #     default_scale = self.acceleration_visualizer.cfg.markers["arrow"].scale
    #     arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self._data.lin_acc_b.shape[0], 1)
    #     # get up axis of current stage
    #     up_axis = stage_utils.get_stage_up_axis()
    #     # arrow-direction
    #     quat_opengl = math_utils.quat_from_matrix(
    #         math_utils.create_rotation_matrix_from_view(
    #             self._data.pos_w,
    #             self._data.pos_w + math_utils.quat_apply(self._data.quat_w, self._data.lin_acc_b),
    #             up_axis=up_axis,
    #             device=self._device,
    #         )
    #     )
    #     quat_w = math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "world")
    #     # display markers
    #     self.acceleration_visualizer.visualize(base_pos_w, quat_w, arrow_scale)
