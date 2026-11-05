"""Shared fixtures.  Tiny worlds keep the core tests fast."""

from __future__ import annotations

import pytest

from life_sandbox.config import Capacities, Config, RunConfig, WorldConfig


@pytest.fixture
def tiny_cfg() -> Config:
    return Config.default().replaced(
        capacities=Capacities(creature=8, bush=6, carcass=2, neutral=1),
        world=WorldConfig(founders=4),
        run=RunConfig(seed=3, ticks=5),
    )
