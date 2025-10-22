"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.managers import CommandTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import UniformCommandCfg


class UniformCommand(CommandTerm):


    cfg: UniformCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.scalar_command = torch.zeros(self.num_envs, 1, device=self.device)
        # # -- metrics
        # self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base command in the base frame. Shape is (num_envs,)."""
        return self.scalar_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # # logs data
        # self.metrics["error_vel_xy"] += (
        #     torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        # )
        # self.metrics["error_vel_yaw"] += (
        #     torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        # )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), 1, device=self.device)
        self.scalar_command[env_ids, :] = r.uniform_(*self.cfg.ranges)

    def _update_command(self):
        """Update the command (no-op for uniform command as it doesn't change between resampling)."""
        pass
