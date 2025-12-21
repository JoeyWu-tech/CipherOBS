"""Compatibility module for loading legacy checkpoints.

This module provides backward compatibility for checkpoints saved with
the old module path `src.modules`. It re-exports modules from their
current location in src/stage2/models/modules/.
"""

from src.stage2.models.modules.scr_modules import (
    StyleExtractor,
    Projector,
    make_layers,
    vgg,
)

__all__ = ['StyleExtractor', 'Projector', 'make_layers', 'vgg']


