"""Configuration loading, validation and hashing (SANDBOX.md §10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from life_sandbox.config import Config, Sensors

ROOT = Path(__file__).resolve().parents[1]


def test_default_toml_matches_builtin_defaults():
    cfg = Config.load(ROOT / "configs" / "world" / "default.toml")
    assert cfg.to_dict() == Config.default().to_dict()


def test_slice_dimension_is_70_before_sound_phase():
    assert Config.default().sensors.slice_dim == 70
    assert Config.default().replaced(sensors=Sensors(sound_channels=2)).sensors.slice_dim == 74  # two ears


def test_config_hash_is_stable_and_seed_sensitive():
    a = Config.default()
    b = Config.load(ROOT / "configs" / "world" / "default.toml")
    assert a.config_hash() == b.config_hash()
    assert a.replaced(run=a.run.__class__(seed=1)).config_hash() != a.config_hash()


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown config key"):
        Config.from_dict({"world": {"nope": 1}})
    with pytest.raises(ValueError, match="unknown config key"):
        Config.from_dict({"nope": 1})


def test_readable_leak_is_rejected_until_m3():
    cfg = Config.default().replaced(
        learning=Config.default().learning.__class__(readable_leak=True)
    )
    with pytest.raises(ValueError, match="readable_leak"):
        cfg.validate()


def test_brain_widths_are_not_configurable():
    assert not hasattr(Config.default().brain, "genome_dim")
    assert not hasattr(Config.default().brain, "plastic_dim")


def test_validation_catches_bad_slice_and_granularity():
    bad = Config.default().replaced(sensors=Sensors(sound_channels=1))
    with pytest.raises(ValueError, match="slice dimension"):
        bad.validate()
    bad_eta = Config.default().replaced(
        learning=Config.default().learning.__class__(eta_granularity="per_synapse")
    )
    with pytest.raises(ValueError, match="eta_granularity"):
        bad_eta.validate()
