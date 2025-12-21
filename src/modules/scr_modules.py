"""Compatibility module for loading legacy checkpoints.

Re-exports from src.stage2.models.modules.scr_modules for backward compatibility.
"""

from src.stage2.models.modules.scr_modules import (
    StyleExtractor,
    Projector,
    make_layers,
    vgg,
)

__all__ = ['StyleExtractor', 'Projector', 'make_layers', 'vgg']


