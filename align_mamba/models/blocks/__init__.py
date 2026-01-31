"""Decoder block implementations.

Importing this module registers all blocks with the BlockRegistry.
"""

from .polarized import PolarizedMamba2Block
from .state_expanded import StateExpandedBlock, compute_forget_lower_bound
from .memmamba import MemMambaBlock

__all__ = [
    "PolarizedMamba2Block",
    "StateExpandedBlock",
    "compute_forget_lower_bound",
    "MemMambaBlock",
]
