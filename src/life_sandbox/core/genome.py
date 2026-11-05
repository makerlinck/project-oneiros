"""Genome layout shared by every candidate brain (SANDBOX.md §6.3).

Block order is a code contract: founders write these blocks, mutation perturbs
them, and every brain decodes from the same offsets.  Brain-specific blocks
start at BRAIN_BASE and are owned by the selected brain module.
"""

from __future__ import annotations


def diet_logits_slice() -> slice:
    return slice(0, 3)


def color_slice() -> slice:
    return slice(3, 6)


def odor_slice(odor_channels: int) -> slice:
    return slice(6, 6 + odor_channels)


def theta_repro_index(odor_channels: int) -> int:
    return 6 + odor_channels


def brain_base(odor_channels: int) -> int:
    """First index available to brain-specific blocks."""
    return 7 + odor_channels
