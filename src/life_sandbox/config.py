"""Experiment configuration (SANDBOX.md §10).

Arms differ only in configuration (SANDBOX.md §0, §12.1): the code exposes
capability switches, never arm names.  Configuration is static Python data,
never part of the JAX pytree, so it cannot leak into the agent or the random
stream.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, get_type_hints

ETA_GRANULARITIES = ("per_unit", "per_group", "shared")
BRAIN_KINDS = ("bootstrap", "plastic_rnn", "slice_transformer", "queue_rnn", "queue_segment_rnn", "mtrnn", "four_layer")


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    ticks: int = 200


@dataclass(frozen=True)
class WorldConfig:
    """World geometry and global constants (SANDBOX.md §4)."""

    width: float = 64.0
    height: float = 64.0
    cell: float = 1.0
    day_ticks: float = 200.0
    odor_diffusion: float = 0.2
    odor_decay: float = 0.02
    odor_scale: int = 1_000_000
    founders: int = 64
    ramp_ticks: int = 0
    event_capacity: int = 2048
    # Calibration placeholders: L is measured in M1 (PRINCIPLES section 7 stage 0),
    # c_switch is an experiment factor {0.25, 0.5, 1, 2, 4} (SANDBOX section 4.4).
    l_ticks: float = 500.0
    c_switch: float = 1.0

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Odor grid resolution (rows, cols), one cell per world unit."""
        return (round(self.height / self.cell), round(self.width / self.cell))


@dataclass(frozen=True)
class Capacities:
    """Prototype slot limits (SANDBOX.md §3.1)."""

    creature: int = 256
    bush: int = 60
    carcass: int = 64
    neutral: int = 3


@dataclass(frozen=True)
class Sensors:
    """Slice layout (SANDBOX.md §5.6).  Segments are assembled in this order."""

    vision_pixels: int = 15
    fov_deg: float = 60.0
    odor_channels: int = 4
    touch_channels: int = 4
    proprio_channels: int = 2
    interoception_channels: int = 6
    skin_channels: int = 1
    light_channels: int = 1
    load_channels: int = 1
    efference_channels: int = 6
    sound_channels: int = 0  # channels per ear; phase 6 adds 2 x S dims (§5.6)

    @property
    def vision_channels(self) -> int:
        return 3

    @property
    def slice_dim(self) -> int:
        """Total slice width D: 70 before the sound phase, 74 after (§5.6)."""
        return (
            self.vision_pixels * self.vision_channels
            + self.odor_channels
            + self.touch_channels
            + self.load_channels
            + self.proprio_channels
            + self.interoception_channels
            + self.skin_channels
            + self.light_channels
            + self.efference_channels
            + 2 * self.sound_channels
        )


@dataclass(frozen=True)
class BrainConfig:
    # kind selects the architecture by configuration (SANDBOX §6.4).  M0 drives
    # bodies with the bootstrap random policy; plastic_rnn (B0) is M2.
    kind: str = "bootstrap"
    h: int = 32
    modulators: int = 4
    # §6.5 estimates ~8.3k; the exact B0 block (weights, G per post unit, biases,
    # per-unit eta/lambda, A-D, eta_p) needs 8354 including the shared header.
    genome_dim: int = 8400
    plastic_dim: int = 5696
    motion_noise: float = 0.05


@dataclass(frozen=True)
class LearningConfig:
    """Capability switches for the default brain (SANDBOX.md §6.2).

    Arms are built by flipping these switches; no arm name appears in code.
    """

    plasticity: bool = True
    prediction: bool = True
    eta_granularity: str = "per_unit"
    body_loop: bool = True
    readable_leak: bool = False


@dataclass(frozen=True)
class MotionConfig:
    noise_scale: float = 1.0


@dataclass(frozen=True)
class Config:
    run: RunConfig = RunConfig()
    world: WorldConfig = WorldConfig()
    capacities: Capacities = Capacities()
    sensors: Sensors = Sensors()
    brain: BrainConfig = BrainConfig()
    learning: LearningConfig = LearningConfig()
    motion: MotionConfig = MotionConfig()

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], source: str = "<dict>") -> "Config":
        return _build(cls, raw, source)

    @classmethod
    def default(cls) -> "Config":
        return cls()

    def replaced(self, **sections: Any) -> "Config":
        """Return a copy with whole sections replaced (arm construction)."""
        return dataclasses.replace(self, **sections)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """Hash recorded in the runs table (SANDBOX.md §7.2)."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def validate(self) -> None:
        s = self.sensors
        if s.slice_dim not in (70, 74):
            raise ValueError(
                "slice dimension must be 70 before the sound phase or 74 after; got "
                + str(s.slice_dim)
                + " (SANDBOX.md §5.6)"
            )
        if self.capacities.creature <= 0 or self.world.founders <= 0:
            raise ValueError("capacities.creature and world.founders must be positive")
        if self.world.founders > self.capacities.creature:
            raise ValueError("world.founders cannot exceed capacities.creature")
        if self.world.cell <= 0 or self.world.odor_scale <= 0:
            raise ValueError("world.cell and world.odor_scale must be positive")
        if self.world.event_capacity <= 0:
            raise ValueError("world.event_capacity must be positive")
        if self.world.l_ticks <= 0 or self.world.c_switch <= 0:
            raise ValueError("world.l_ticks and world.c_switch must be positive")
        if self.brain.kind not in BRAIN_KINDS:
            raise ValueError(
                "brain.kind must be one of " + repr(BRAIN_KINDS) + ", got " + repr(self.brain.kind)
            )
        if self.learning.eta_granularity not in ETA_GRANULARITIES:
            raise ValueError(
                "learning.eta_granularity must be one of "
                + repr(ETA_GRANULARITIES)
                + ", got "
                + repr(self.learning.eta_granularity)
            )


def _build(cls: type, raw: Any, source: str) -> Any:
    """Fill a frozen dataclass tree from nested TOML data, rejecting typos."""
    hints = get_type_hints(cls)
    if not isinstance(raw, Mapping):
        raise ValueError("expected a table for " + cls.__name__ + " in " + source)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError("unknown config key(s) " + repr(sorted(unknown)) + " in " + source)
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, source + "." + name)
        elif hint is bool:
            kwargs[name] = bool(value)
        elif hint is int:
            kwargs[name] = int(value)
        elif hint is float:
            kwargs[name] = float(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
