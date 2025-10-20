# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
import torch

@dataclass
class GaitSensorData:
    """Simplified data container for gait-related sensor outputs."""

    # 基础 gait 相位
    gait_indices: torch.Tensor = None                # (num_envs,)

    # 每条腿的 gait 相位（包含 phase + offset + bound）
    foot_indices: torch.Tensor = None                # (num_envs, 4)

    # 各腿目标接触概率（连续平滑）
    desired_contact_states: torch.Tensor = None      # (num_envs, 4)

    # 时钟信号（sin waveform）
    clock_inputs: torch.Tensor = None                # (num_envs, 4)
    doubletime_clock_inputs: torch.Tensor = None     # (num_envs, 4)
    halftime_clock_inputs: torch.Tensor = None       # (num_envs, 4)

