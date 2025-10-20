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
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.gait_indices[env_ids] = 0.0

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

        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
            
        cmd_frequency = 3.0
        
        self._data.gait_indices[env_ids] = torch.remainder(
            self._data.gait_indices[env_ids] + self._sim_physics_dt * cmd_frequency,
            1.0
        )


    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        # data buffers
        self._data.gait_indices = torch.zeros(self._num_envs, device=self.device)

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
