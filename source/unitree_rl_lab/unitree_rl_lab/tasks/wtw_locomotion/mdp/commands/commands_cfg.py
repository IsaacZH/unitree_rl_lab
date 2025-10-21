import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from .uniform_command import UniformCommand
    
    
@configclass
class UniformCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    ranges: tuple[float, float] = MISSING
    """Distribution ranges for the commands."""