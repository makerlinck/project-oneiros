"""Initial world placement (SANDBOX.md §4, §5.1, §5.5).

Builds the first world state: bushes (3 species, temperature-preference
weighted positions), the founder cohort as adults, neutral objects, berry
composition, per-run pain-bundle weights and the uid allocation.  Everything is
drawn from the init key; there is no Python randomness.

Founder body values and bush stocks are calibration values (M1); they live in
world/constants.py so a calibration record can cite one place.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.brain.genome import initial_sigmas, layout as brain_layout
from life_sandbox.config import Config
from life_sandbox.core import components as C
from life_sandbox.core import genome as G
from life_sandbox.core import rng
from life_sandbox.core.masks import allocate_uids
from life_sandbox.core.state import WorldState, empty
from life_sandbox.core.struct import update
from life_sandbox.world import constants as K


def initial_state(cfg: Config, seed: int | None = None) -> WorldState:
    """Create the first world state for one world seed."""
    cfg.validate()
    key = rng.world_key(cfg.run.seed if seed is None else seed)
    state = empty(cfg, key)
    init_key = rng.system_key(key, jnp.asarray(0, dtype=jnp.int32), "init")
    k_bush, k_founder, k_neutral, k_pain = jax.random.split(init_key, 4)
    state = _place_bushes(state, cfg, k_bush)
    state = _place_founders(state, cfg, k_founder)
    state = _place_neutrals(state, cfg, k_neutral)
    state = _sample_pain_weights(state, cfg, k_pain)
    return state


def random_genome(key: jax.Array, n: int, cfg: Config) -> jax.Array:
    """Random genome with the shared header blocks filled (SANDBOX §5.1, §6.3).

    Founder diet logits are plant/animal N(0, 1) and toxin N(-3, 1); colour,
    odour and theta_repro are drawn from their own keys so a reseed (death.py)
    reuses exactly the same convention as the founder cohort.
    """
    keys = jax.random.split(key, 7)
    channels = cfg.sensors.odor_channels
    layout = brain_layout(cfg)
    # C4 initialization table: per-block standard deviations follow the fan-in
    # (SANDBOX.md §6.3), so founders differ in their default action tendencies.
    genome = jax.random.normal(keys[0], (n, layout.total), dtype=jnp.float32) * initial_sigmas(cfg)[
        None, :
    ]
    diet_logits = jnp.stack(
        [
            jax.random.normal(keys[1], (n,), dtype=jnp.float32) * K.FOUNDER_LOGIT_STD,
            jax.random.normal(keys[2], (n,), dtype=jnp.float32) * K.FOUNDER_LOGIT_STD,
            jax.random.normal(keys[3], (n,), dtype=jnp.float32) * K.FOUNDER_LOGIT_STD
            + K.FOUNDER_TOXIN_LOGIT_MEAN,
        ],
        axis=1,
    )
    color = jax.random.normal(keys[4], (n, 3), dtype=jnp.float32)
    odor = jax.random.normal(keys[5], (n, channels), dtype=jnp.float32)
    theta_repro = jax.random.uniform(keys[6], (n,), minval=0.5, maxval=1.0)
    genome = genome.at[:, G.diet_logits_slice()].set(diet_logits)
    genome = genome.at[:, G.color_slice()].set(color)
    genome = genome.at[:, G.odor_slice(channels)].set(odor)
    genome = genome.at[:, G.theta_repro_index(channels)].set(theta_repro)
    return genome


def diet_from_genome(genome: jax.Array, cfg: Config) -> jax.Array:
    """Diet weights rho = softmax(diet logits), which also give N_cap (§5.1)."""
    return jax.nn.softmax(genome[:, G.diet_logits_slice()], axis=1)


def look_from_genome(genome: jax.Array, cfg: Config) -> C.Look:
    """Hereditary colour (sigmoid) and odour spectrum (raw genome values)."""
    channels = cfg.sensors.odor_channels
    return C.Look(
        rgb=jax.nn.sigmoid(genome[:, G.color_slice()]),
        odor=genome[:, G.odor_slice(channels)],
    )


def body_from_genome(genome: jax.Array, cfg: Config) -> C.Body:
    """Adult body defaults from a genome (founders and reseeded individuals)."""
    n = genome.shape[0]
    diet = diet_from_genome(genome, cfg)
    reserves = cfg.body.founder_reserves * diet
    # Adult tissue follows from the growth rule: newborn 0.03 + 0.1 per grown
    # mass unit over (1.0 - 0.3) => 0.10 * c_flesh (SANDBOX.md §5.2, §5.5).
    tissue = K.ADULT_TISSUE_FRACTION * jnp.asarray(K.FLESH_COMP, dtype=jnp.float32)[None, :]
    tissue = jnp.broadcast_to(tissue, (n, C.N_NUTRIENTS))
    return C.Body(
        G=jnp.zeros((n, C.N_NUTRIENTS), dtype=jnp.float32),
        N=reserves,
        B=tissue,
        I=jnp.full((n,), K.FOUNDER_INTEGRITY, dtype=jnp.float32),
        X=jnp.zeros((n,), dtype=jnp.float32),
        b=jnp.zeros((n,), dtype=jnp.float32),
        k=jnp.zeros((n,), dtype=jnp.float32),
        p=jnp.zeros((n,), dtype=jnp.float32),
        phi_hb=jnp.zeros((n,), dtype=jnp.float32),
        S=jnp.zeros((n,), dtype=jnp.float32),
        asleep=jnp.zeros((n,), dtype=jnp.bool_),
        age=jnp.zeros((n,), dtype=jnp.int32),
        mu=jnp.ones((n,), dtype=jnp.float32),
    )


def _pad(rows: jax.Array, total: int) -> jax.Array:
    """Pad a leading-slot array with zeros up to the arena capacity."""
    n = rows.shape[0]
    if n == total:
        return rows
    pad = jnp.zeros((total - n,) + rows.shape[1:], dtype=rows.dtype)
    return jnp.concatenate([rows, pad], axis=0)


def _weighted_sample(key: jax.Array, probs: jax.Array, count: int) -> jax.Array:
    """Sample the given number of distinct indices without replacement."""
    logits = jnp.log(jnp.maximum(probs, 1e-12))
    uniform = jax.random.uniform(key, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
    gumbel = -jnp.log(-jnp.log(uniform))
    return jnp.argsort(-(logits + gumbel))[:count].astype(jnp.int32)


def _spectrum(kind: jax.Array, channels: int) -> jax.Array:
    """Species odour spectra padded or truncated to the configured channel count."""
    base = jnp.asarray(K.BUSH_ODOR, dtype=jnp.float32)
    wide = jnp.zeros((base.shape[0], channels), dtype=jnp.float32)
    take = min(base.shape[1], channels)
    wide = wide.at[:, :take].set(base[:, :take])
    return wide[kind]


def _place_bushes(state: WorldState, cfg: Config, key: jax.Array) -> WorldState:
    capacity = cfg.capacities.bush
    per_kind = capacity // K.NUM_BUSH_KINDS
    if per_kind * K.NUM_BUSH_KINDS != capacity:
        raise ValueError("capacities.bush must divide evenly across the three bush species")
    rows, cols = cfg.world.grid_shape
    cell = cfg.world.cell
    y = (jnp.arange(rows, dtype=jnp.float32) + 0.5) * cell
    t_geo = -2.0 + 4.0 * y / cfg.world.height

    keys = jax.random.split(key, 2 * K.NUM_BUSH_KINDS)
    pos = jnp.zeros((capacity, 2), dtype=jnp.float32)
    kind = jnp.zeros(capacity, dtype=jnp.int32)
    stock = jnp.zeros(capacity, dtype=jnp.float32)
    for species in range(K.NUM_BUSH_KINDS):
        p_y = jnp.exp(-0.5 * ((t_geo - K.BUSH_T_PREF[species]) / K.BUSH_T_SIGMA) ** 2)
        probs = jnp.broadcast_to(p_y[:, None], (rows, cols)).reshape(-1)
        probs = probs / jnp.sum(probs)
        flat = _weighted_sample(keys[2 * species], probs, per_kind)
        iy, ix = flat // cols, flat % cols
        slots = species * per_kind + jnp.arange(per_kind, dtype=jnp.int32)
        positions = jnp.stack(
            [(ix.astype(jnp.float32) + 0.5) * cell, (iy.astype(jnp.float32) + 0.5) * cell], axis=1
        )
        pos = pos.at[slots].set(positions)
        kind = kind.at[slots].set(species)
        stock = stock.at[slots].set(
            jax.random.uniform(keys[2 * species + 1], (per_kind,), minval=0.3, maxval=0.9)
        )

    uids, counter = allocate_uids(state.uid_counter, capacity)
    colors = jnp.asarray(K.BERRY_RGB, dtype=jnp.float32)[kind]
    green = jnp.asarray(K.BUSH_GREEN, dtype=jnp.float32)
    rgb = green[None, :] * (1.0 - stock)[:, None] + colors * stock[:, None]
    arena = C.BushArena(
        alive=jnp.ones(capacity, dtype=jnp.bool_),
        uid=uids,
        transform=C.Transform(
            pos=pos,
            theta=jnp.zeros(capacity, dtype=jnp.float32),
            v=jnp.zeros(capacity, dtype=jnp.float32),
            omega=jnp.zeros(capacity, dtype=jnp.float32),
        ),
        shape=C.Shape(
            radius=jnp.full(capacity, K.BUSH_RADIUS, dtype=jnp.float32),
            mass=jnp.full(capacity, jnp.inf, dtype=jnp.float32),
        ),
        look=C.Look(rgb=rgb, odor=_spectrum(kind, cfg.sensors.odor_channels)),
        edible=C.BushEdible(stock=stock, kind=kind),
    )
    entities = update(state.entities, bush=arena)
    return update(state, entities=entities, uid_counter=counter)


def _place_founders(state: WorldState, cfg: Config, key: jax.Array) -> WorldState:
    capacity = cfg.capacities.creature
    n = cfg.world.founders
    margin = 2.0
    keys = jax.random.split(key, 4)
    pos = jax.random.uniform(
        keys[0],
        (n, 2),
        dtype=jnp.float32,
        minval=margin,
        maxval=jnp.asarray([cfg.world.width - margin, cfg.world.height - margin], jnp.float32),
    )
    theta = jax.random.uniform(keys[1], (n,), minval=-jnp.pi, maxval=jnp.pi)
    genome = random_genome(keys[2], n, cfg)
    heartbeat_phase = jax.random.uniform(keys[3], (n,), minval=0.0, maxval=2.0 * jnp.pi)

    uids, counter = allocate_uids(state.uid_counter, n)
    identity = C.Identity(
        uid=_pad(uids, capacity),
        parent_uid=_pad(jnp.full(n, -1, dtype=jnp.int32), capacity),
        lineage_id=_pad(uids, capacity),
        generation=_pad(jnp.zeros(n, dtype=jnp.int32), capacity),
        birth_tick=_pad(jnp.zeros(n, dtype=jnp.int32), capacity),
    )
    transform = C.Transform(
        pos=_pad(pos, capacity),
        theta=_pad(theta, capacity),
        v=_pad(jnp.zeros(n, dtype=jnp.float32), capacity),
        omega=_pad(jnp.zeros(n, dtype=jnp.float32), capacity),
    )
    shape = C.Shape(
        radius=_pad(jnp.full(n, K.FOUNDER_RADIUS, dtype=jnp.float32), capacity),
        mass=_pad(jnp.full(n, K.ADULT_MASS, dtype=jnp.float32), capacity),
    )
    look_values = look_from_genome(genome, cfg)
    look = C.Look(rgb=_pad(look_values.rgb, capacity), odor=_pad(look_values.odor, capacity))
    adult = body_from_genome(genome, cfg)
    body = C.Body(
        G=_pad(adult.G, capacity),
        N=_pad(adult.N, capacity),
        B=_pad(adult.B, capacity),
        I=_pad(adult.I, capacity),
        X=_pad(adult.X, capacity),
        b=_pad(adult.b, capacity),
        k=_pad(adult.k, capacity),
        p=_pad(adult.p, capacity),
        phi_hb=_pad(heartbeat_phase, capacity),
        S=_pad(adult.S, capacity),
        asleep=_pad(adult.asleep, capacity),
        age=_pad(adult.age, capacity),
        mu=_pad(adult.mu, capacity),
    )
    lay = brain_layout(cfg)
    arena = C.CreatureArena(
        alive=_pad(jnp.ones(n, dtype=jnp.bool_), capacity),
        identity=identity,
        transform=transform,
        shape=shape,
        look=look,
        body=body,
        pain=C.Pain.zeros(capacity),
        damage=C.Damage.zeros(capacity),
        grip=C.Grip.zeros(capacity),
        contact=C.Contact.zeros(capacity),
        genome=C.Genome(g=_pad(genome, capacity)),
        brain=C.BrainState.zeros(capacity, cfg.brain.h, lay.d, lay.plastic_dim),
        motor=C.Motor.zeros(capacity),
        focal=C.Focal.zeros(capacity),
    )
    entities = update(state.entities, creature=arena)
    return update(state, entities=entities, uid_counter=counter)


def _place_neutrals(state: WorldState, cfg: Config, key: jax.Array) -> WorldState:
    capacity = cfg.capacities.neutral
    n = capacity
    margin = 3.0
    keys = jax.random.split(key, 6)
    pos = jax.random.uniform(
        keys[0],
        (n, 2),
        dtype=jnp.float32,
        minval=margin,
        maxval=jnp.asarray([cfg.world.width - margin, cfg.world.height - margin], jnp.float32),
    )
    theta = jax.random.uniform(keys[1], (n,), minval=-jnp.pi, maxval=jnp.pi)
    rgb = jax.random.uniform(keys[2], (n, 3), minval=K.NEUTRAL_POSITIONS_RGB_MIN, maxval=1.0)
    odor = jax.random.normal(keys[3], (n, cfg.sensors.odor_channels), dtype=jnp.float32)
    odor = odor / jnp.maximum(jnp.linalg.norm(odor, axis=1, keepdims=True), 1e-6)
    heading = jax.random.uniform(keys[4], (n,), minval=-jnp.pi, maxval=jnp.pi)
    timer = jax.random.randint(keys[5], (n,), minval=0, maxval=20).astype(jnp.int32)

    uids, counter = allocate_uids(state.uid_counter, n)
    arena = C.NeutralArena(
        alive=jnp.ones(capacity, dtype=jnp.bool_),
        uid=uids,
        transform=C.Transform(
            pos=pos,
            theta=theta,
            v=jnp.zeros(n, dtype=jnp.float32),
            omega=jnp.zeros(n, dtype=jnp.float32),
        ),
        shape=C.Shape(
            radius=jnp.full(n, K.NEUTRAL_RADIUS, dtype=jnp.float32),
            mass=jnp.full(n, K.NEUTRAL_MASS, dtype=jnp.float32),
        ),
        look=C.Look(rgb=rgb, odor=odor),
        held_by=C.HeldBy.zeros(n),
        scripted=C.Scripted(heading=heading, timer=timer),
    )
    entities = update(state.entities, neutral=arena)
    return update(state, entities=entities, uid_counter=counter)


def _sample_pain_weights(state: WorldState, cfg: Config, key: jax.Array) -> WorldState:
    """Per-run unit mixing vectors for the pain bundle (SANDBOX.md §5.6)."""
    weights = jax.random.normal(key, (C.N_PAIN_SOURCES, C.N_PAIN_CHANNELS), dtype=jnp.float32)
    weights = weights / jnp.maximum(jnp.linalg.norm(weights, axis=1, keepdims=True), 1e-6)
    return update(state, pain_weights=weights)
