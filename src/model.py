"""Compatibility module for loading legacy checkpoints.

This module provides backward compatibility for checkpoints saved with
the old module path `src.model`. It re-exports the model classes from
their current location.
"""

# Re-export all model classes for backward compatibility with old checkpoints
from src.stage2.models.model import (
    Upsample,
    FontDiffuserModel,
    FontDiffuserModelDPM,
)

__all__ = ['Upsample', 'FontDiffuserModel', 'FontDiffuserModelDPM']

