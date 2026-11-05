"""sense system (§3.3 step 3, §5.6, §4.3).

Owner: world teammate.  Contract: apply(state, cfg) -> (WorldState, EventBuffer).

Every tick this system builds the slice s_t of every creature slot in the fixed
segment order of SANDBOX.md §5.6::

    vision 1x15x3 = 45 | odour 4 | touch 4 | load 1 | proprio 2 |
    interoception 6 | skin delta-T 1 | light 1 | efference copy 6   -> D = 70

with interoception = stomach fill |G|_1, pain bundle 4 channels, heartbeat 1.
The slice is the only thing the brain sees; N (nutrient stores), I (integrity)
and X (toxin load) never enter it directly.  Core temperature b enters only
through the sanctioned physical sensor "skin delta-T = T(position) - b"
(§5.6, PRINCIPLES.md §2.2), never as a raw readout.  The pain bundle carries
body state only as noise amplitude: zero mean, no scalar valence.

Pain bundle (§5.6): nine sources, each with its own strength a_k(t), waveform and
period/damping.  Eight are resonant AR(2) carriers

    x_k(t) = phi1*x_k(t-1) + phi2*x_k(t-2) + c_k*xi_k(t)
    phi1 = 2*r_k*cos(2*pi/P_k), phi2 = -r_k^2
    c_k  = sqrt((1+phi2)*((1-phi2)^2 - phi1^2)/(1-phi2))   (unit stationary var)

and the first source (acute pain) is a sparse +-1 impulse train with probability
min(1, a_inj) per tick, smoothed by a first-order filter with phi = 0.5.  They are
mixed into the shared four channels through the per-run unit vectors
state.pain_weights[9, 4] plus a 0.02 base noise.  The AR(2) carrier state lives
in Pain.x and is updated by this system (the body system only updates a_inj).

The leak arm (cfg.learning.readable_leak, PRINCIPLES.md §2.4 exception) appends
the five designer summary dimensions of §5.6 after the physical sensors; the
state slice is allocated with the extra columns by state.empty.  The fifth leak
dimension needs a per-creature memory of who was bitten in the last 50 ticks;
no such component exists in M0, so it is written as 0.0 and left for the Damage
component to grow later.

Determinism (SANDBOX.md §2): all noise comes from the sense system key; the
segment order and the random draw order are fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from life_sandbox.config import Config
from life_sandbox.core import events, genome, masks, rng
from life_sandbox.core.components import N_PAIN_SOURCES
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update
from life_sandbox.world import fields, vision

# Stomach capacity G_cap (§5.1, §5.2); the distension source starts at 80% full.
G_CAP: float = 0.3
DISTENSION_ONSET: float = 0.8

# The pain bundle always shares four interoceptive channels (§5.6).
PAIN_CHANNELS: int = 4

# Nine sources: acute, toxin, distension, heat, cold, hunger(plant/animal/toxin),
# sleepiness.  Acute has no period; it is handled separately (§5.6).
PAIN_PERIODS: tuple[float | None, ...] = (None, 5.0, 8.0, 12.0, 20.0, 40.0, 70.0, 110.0, 30.0)
PAIN_DAMPING: tuple[float | None, ...] = (None, 0.90, 0.90, 0.95, 0.95, 0.995, 0.995, 0.995, 0.97)
ACUTE_SMOOTHING: float = 0.5

# Width of the readable_leak block (§5.6).
LEAK_WIDTH: int = 5


@dataclass(frozen=True)
class NoiseScales:
    """Standard deviations of the independent sensor noises (§5.6).

    Values fixed by the specification are the pixel noise formula, the 0.02
    pain base noise and the 0.05 heartbeat noise; the rest are local M0 choices
    kept here as one configurable bundle instead of scattering literals.
    """

    vision_base: float = vision.VISION_BASE_NOISE
    vision_dark: float = vision.VISION_DARK_NOISE
    odour: float = 0.01
    touch: float = 0.005
    load: float = 0.005
    proprio: float = 0.005
    stomach: float = 0.005
    pain: float = 0.02
    heartbeat: float = 0.05
    skin: float = 0.02
    light: float = 0.01


DEFAULT_NOISE = NoiseScales()


@dataclass(frozen=True)
class Segment:
    """One slice segment: name, width and which body reads it is allowed.

    reads_body is a static audit tag for tests/test_slice_layout.py:
    none = pure exteroception, amplitude = body state only sets noise amplitude
    (pain bundle), difference = the sanctioned T - b skin sensor.
    """

    name: str
    width: int
    reads_body: str


def segments(cfg: Config) -> tuple[Segment, ...]:
    """The slice layout in its fixed order (§5.6)."""
    sensors = cfg.sensors
    if sensors.sound_channels:
        raise NotImplementedError("sound channels arrive in phase 6 (§5.6)")
    interoception = 1 + PAIN_CHANNELS + 1
    if interoception != sensors.interoception_channels:
        raise ValueError(
            "interoception must be stomach(1) + pain(4) + heartbeat(1) (§5.6); got "
            + str(sensors.interoception_channels)
        )
    if sensors.odor_channels != PAIN_CHANNELS:
        raise ValueError("the pain bundle shares four channels, so odour_channels is 4 (§5.6)")
    return (
        Segment("vision", sensors.vision_pixels * sensors.vision_channels, "none"),
        Segment("odour", sensors.odor_channels, "none"),
        Segment("touch", sensors.touch_channels, "none"),
        Segment("load", sensors.load_channels, "none"),
        Segment("proprio", sensors.proprio_channels, "none"),
        Segment("interoception", sensors.interoception_channels, "amplitude"),
        Segment("skin", sensors.skin_channels, "difference"),
        Segment("light", sensors.light_channels, "none"),
        Segment("efference", sensors.efference_channels, "none"),
    )


def offsets(cfg: Config) -> dict[str, int]:
    """Start index of every segment in the physical part of the slice."""
    starts: dict[str, int] = {}
    position = 0
    for segment in segments(cfg):
        starts[segment.name] = position
        position += segment.width
    return starts


def physical_dim(cfg: Config) -> int:
    """Width of the physical sensors (70 before the sound phase, §5.6)."""
    return sum(segment.width for segment in segments(cfg))


def slice_dim(cfg: Config) -> int:
    """Width the brain sees: physical sensors plus the readable_leak block."""
    return physical_dim(cfg) + (LEAK_WIDTH if cfg.learning.readable_leak else 0)


def diet_rho(state: WorldState) -> jax.Array:
    """Diet need proportions rho = softmax(genome logits) (§5.1)."""
    logits = state.entities.creature.genome.g[:, genome.diet_logits_slice()]
    return jax.nn.softmax(logits, axis=-1).astype(jnp.float32)


def pain_strengths(state: WorldState, cfg: Config) -> jax.Array:
    """Nine noise-source strengths a_k(t), computed from body state (§5.6).

    The body quantities only set how loud each source is; none of them is
    written to the slice.  Order matches components.PAIN_SOURCES.
    """
    creature = state.entities.creature
    body = creature.body
    rho = diet_rho(state)
    stomach = jnp.sum(body.G, axis=-1)
    distension = jnp.maximum(0.0, stomach / G_CAP - DISTENSION_ONSET) / (1.0 - DISTENSION_ONSET)
    fill = body.N / jnp.maximum(rho, 1e-6)
    hunger = jnp.minimum(1.0, 3.0 * rho) * (1.0 - fill) ** 2
    sources = jnp.stack(
        [
            creature.pain.a_inj,
            2.0 * body.X,
            distension,
            jnp.maximum(0.0, body.b - 0.5),
            jnp.maximum(0.0, -body.b - 0.5),
            hunger[:, 0],
            hunger[:, 1],
            hunger[:, 2],
            body.S,
        ],
        axis=-1,
    )
    return sources.astype(jnp.float32)


def ar2_coefficients(period: float, damping: float) -> tuple[float, float, float]:
    """AR(2) coefficients of a resonant carrier with unit stationary variance."""
    phi1 = 2.0 * damping * math.cos(2.0 * math.pi / period)
    phi2 = -(damping * damping)
    gain = math.sqrt(((1.0 + phi2) * ((1.0 - phi2) ** 2 - phi1 * phi1)) / (1.0 - phi2))
    return phi1, phi2, gain


def _carrier_constants() -> tuple[jax.Array, jax.Array, jax.Array]:
    phi1 = []
    phi2 = []
    gain = []
    for period, damping in zip(PAIN_PERIODS, PAIN_DAMPING, strict=True):
        if period is None:
            phi1.append(0.0)
            phi2.append(0.0)
            gain.append(0.0)
            continue
        a, b, c = ar2_coefficients(float(period), float(damping))
        phi1.append(a)
        phi2.append(b)
        gain.append(c)
    # Drop the acute slot: sources 1..8 are the resonant carriers.
    return (
        jnp.asarray(phi1[1:], jnp.float32),
        jnp.asarray(phi2[1:], jnp.float32),
        jnp.asarray(gain[1:], jnp.float32),
    )


_PHI1, _PHI2, _GAIN = _carrier_constants()


def pain_step(
    state: WorldState,
    cfg: Config,
    key: jax.Array,
    noise: NoiseScales = DEFAULT_NOISE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Advance the nine pain carriers one tick (§5.6).

    Returns (bundle signal [N, 4], new carrier state [N, 9, 2], strengths [N, 9]).
    The sense system owns the carrier state; the body system owns a_inj.
    """
    n = cfg.capacities.creature
    pain = state.entities.creature.pain
    weights = state.pain_weights
    channels = weights.shape[1]
    strengths = pain_strengths(state, cfg)

    acute_key, ar_key, base_key = rng.subsplit(key, 3)
    probability_key, sign_key = rng.subsplit(acute_key, 2)

    # Acute pain: sparse +-1 impulses, then a first-order smoother (phi = 0.5).
    probability = jnp.minimum(1.0, strengths[:, 0])
    draw = jax.random.uniform(probability_key, (n,), dtype=jnp.float32)
    sign = jnp.where(jax.random.uniform(sign_key, (n,), dtype=jnp.float32) < 0.5, -1.0, 1.0)
    impulse = jnp.where(draw < probability, sign, 0.0)
    acute = ACUTE_SMOOTHING * pain.x[:, 0, 0] + (1.0 - ACUTE_SMOOTHING) * impulse

    # Resonant AR(2) carriers for sources 1..8.
    previous_one = pain.x[:, 1:, 0]
    previous_two = pain.x[:, 1:, 1]
    drive = jax.random.normal(ar_key, (n, N_PAIN_SOURCES - 1), dtype=jnp.float32)
    resonant = _PHI1 * previous_one + _PHI2 * previous_two + _GAIN * drive

    carriers = jnp.concatenate([acute[:, None], resonant], axis=1)  # [N, 9]
    history = jnp.concatenate([jnp.zeros((n, 1), jnp.float32), previous_one], axis=1)
    new_x = jnp.stack([carriers, history], axis=-1)

    mixed = jnp.einsum("nk,kc->nc", carriers * strengths, weights)
    base = jax.random.normal(base_key, (n, channels), dtype=jnp.float32)
    signal = mixed + noise.pain * base
    return signal.astype(jnp.float32), new_x, strengths


def leak_dims(state: WorldState, cfg: Config) -> jax.Array:
    """The five designer summary dimensions of the leak arm (§5.6).

    Only used when cfg.learning.readable_leak is true (PRINCIPLES.md §2.4
    exception).  Dimension 5 needs a per-creature memory of biting a living
    target in the last 50 ticks; M0 has no such component, so it stays 0.
    """
    creature = state.entities.creature
    body = creature.body
    n = cfg.capacities.creature
    total_nutrients = jnp.sum(body.N, axis=-1) - 0.5
    recent_biter = jnp.zeros((n,), jnp.float32)
    return jnp.stack([total_nutrients, body.I, body.X, body.b, recent_biter], axis=-1).astype(
        jnp.float32
    )


def _noise(key: jax.Array, shape: tuple[int, ...], scale: float) -> jax.Array:
    return jax.random.normal(key, shape, dtype=jnp.float32) * jnp.asarray(scale, jnp.float32)


def observe(
    state: WorldState,
    cfg: Config,
    key: jax.Array,
    noise: NoiseScales = DEFAULT_NOISE,
) -> tuple[jax.Array, jax.Array]:
    """Assemble s_t for every creature slot; also return the new pain state.

    Pure with respect to the returned state (the caller stores new_x).  The
    physical part is 70 dims; the leak block is appended only in the leak arm.
    """
    n = cfg.capacities.creature
    creature = state.entities.creature
    body = creature.body
    sensors = cfg.sensors

    if physical_dim(cfg) != sensors.slice_dim:
        raise ValueError(
            "slice layout mismatch: segments sum to "
            + str(physical_dim(cfg))
            + " but sensors.slice_dim is "
            + str(sensors.slice_dim)
            + " (§5.6)"
        )

    vision_key, pain_key, misc_key = rng.subsplit(key, 3)
    pixels = vision.render(
        state,
        cfg,
        vision_key,
        noise_base=noise.vision_base,
        noise_dark=noise.vision_dark,
    )
    pain, new_x, _ = pain_step(state, cfg, pain_key, noise)

    keys = rng.subsplit(misc_key, 8)
    (
        odour_key,
        touch_key,
        load_key,
        proprio_key,
        stomach_key,
        heartbeat_key,
        skin_key,
        light_key,
    ) = (keys[0], keys[1], keys[2], keys[3], keys[4], keys[5], keys[6], keys[7])

    position = creature.transform.pos
    heading = creature.transform.theta
    nose = position + (creature.shape.radius + fields.NOSE_OFFSET)[:, None] * jnp.stack(
        [jnp.cos(heading), jnp.sin(heading)], axis=-1
    )
    odour = jnp.log1p(jnp.maximum(fields.sample_odor(cfg, state.fields.odor, nose), 0.0))
    odour = odour + _noise(odour_key, (n, sensors.odor_channels), noise.odour)

    touch = creature.contact.touch + _noise(touch_key, (n, sensors.touch_channels), noise.touch)
    load = masks.held_mass(state.entities)[:, None] + _noise(load_key, (n, 1), noise.load)
    proprio = jnp.stack([creature.transform.v, creature.transform.omega], axis=-1)
    proprio = proprio + _noise(proprio_key, (n, sensors.proprio_channels), noise.proprio)

    stomach = jnp.sum(body.G, axis=-1, keepdims=True)
    stomach = stomach + _noise(stomach_key, (n, 1), noise.stomach)
    heartbeat = jnp.sin(body.phi_hb)[:, None]
    heartbeat = heartbeat + _noise(heartbeat_key, (n, 1), noise.heartbeat)
    interoception = jnp.concatenate([stomach, pain, heartbeat], axis=-1)

    temperature, luminance = fields.environment_at(state, cfg)
    skin = (temperature - body.b)[:, None]
    skin = skin + _noise(skin_key, (n, 1), noise.skin)
    light = luminance[:, None] + _noise(light_key, (n, 1), noise.light)

    # Efference copy is not a sensor: it is an exact copy of last tick's outputs.
    efference = creature.motor.last_out

    pieces = [
        pixels.reshape(n, -1),
        odour,
        touch,
        load,
        proprio,
        interoception,
        skin,
        light,
        efference,
    ]
    if cfg.learning.readable_leak:
        pieces.append(leak_dims(state, cfg))
    assembled = jnp.concatenate(pieces, axis=-1)
    assembled = jnp.where(creature.alive[:, None], assembled, 0.0)
    return assembled.astype(jnp.float32), new_x


def apply(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    """Step 3: advance the pain carriers and write the slice for every slot."""
    key = rng.system_key(state.key, state.tick, "sense")
    assembled, new_x = observe(state, cfg, key)
    creature = update(state.entities.creature, pain=update(state.entities.creature.pain, x=new_x))
    state = update(
        state,
        entities=update(state.entities, creature=creature),
        slice=assembled,
    )
    return state, events.empty(cfg)
