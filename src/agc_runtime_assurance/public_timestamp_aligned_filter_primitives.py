"""Stable public facade for the V8 timestamp-aligned backup primitives.

Independent environment adapters import this facade rather than any scientific
runner.  It contains no simulator bridge, seed logic, output writer, or result
analysis.
"""

from .gazebo_robust_backup_filter import GazeboPlanarPlant, RobustBackupConfig
from .gazebo_timestamp_aligned_set_filter import (
    AlignedStateSet,
    SetBackupDecision,
    TimestampAlignedSetBackupFilter,
    TimestampAlignmentConfig,
    TimestampAlignmentError,
    align_async_state_history,
)

__all__ = [
    "AlignedStateSet",
    "GazeboPlanarPlant",
    "RobustBackupConfig",
    "SetBackupDecision",
    "TimestampAlignedSetBackupFilter",
    "TimestampAlignmentConfig",
    "TimestampAlignmentError",
    "align_async_state_history",
]
