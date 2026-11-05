"""ECS components, stored as structure-of-arrays pytrees (SANDBOX.md §3.1-3.2).

A component is a frozen dataclass whose fields are JAX arrays with a leading
slot axis of length capacity.  Entities are slot indices into a per-prototype
arena; the alive mask marks occupied slots.  Slot indices are reused, uids are
not.

Dtypes are explicit everywhere: float32 for continuous quantities, int32 for
indices, counters and uids, bool for masks.  x64 is deliberately off, so uids
are int32: one world's counter stays far below 2**31 (see README status note).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from life_sandbox.core.struct import struct

# Prototype ids (§3.1).  Stored in references; never used as array indices.
CREATURE: int = 0
BUSH: int = 1
CARCASS: int = 2
NEUTRAL: int = 3
PROTOTYPE_NAMES: tuple[str, ...] = ("creature", "bush", "carcass", "neutral")

# Body output layout (§5.3).  The brain emits u_acc, u_turn, k, u_bite, u_grab,
# u_drop; Motor.last_out keeps the same order.
OUT_ACC, OUT_TURN, OUT_K, OUT_BITE, OUT_GRAB, OUT_DROP = range(6)
N_OUTPUTS: int = 6

# Nutrient slots (§5.1).
PLANT, ANIMAL, TOXIN = range(3)
N_NUTRIENTS: int = 3
FLESH_COMP: tuple[float, float, float] = (0.3, 0.7, 0.0)

# Pain bundle sources (§5.6).  These are observer-side names only: the agent
# receives four shared channels and never sees which source is which.
PAIN_SOURCES: tuple[str, ...] = (
    "acute",
    "toxin",
    "distension",
    "heat",
    "cold",
    "hunger_plant",
    "hunger_animal",
    "hunger_toxin",
    "sleepiness",
)
N_PAIN_SOURCES: int = len(PAIN_SOURCES)

# Empty reference sentinel.
NO_REF: int = -1


def f32(*shape: int) -> jax.Array:
    return jnp.zeros(shape, dtype=jnp.float32)


def i32(*shape: int) -> jax.Array:
    return jnp.zeros(shape, dtype=jnp.int32)


def i8(*shape: int) -> jax.Array:
    return jnp.zeros(shape, dtype=jnp.int8)


def boolean(*shape: int) -> jax.Array:
    return jnp.zeros(shape, dtype=jnp.bool_)


@struct
@dataclass(frozen=True)
class Identity:
    """Per-creature identity (§3.2).  uid is unique per world and never reused."""

    uid: jax.Array
    parent_uid: jax.Array
    lineage_id: jax.Array
    generation: jax.Array
    birth_tick: jax.Array

    @staticmethod
    def zeros(n: int) -> "Identity":
        return Identity(
            uid=i32(n),
            parent_uid=i32(n),
            lineage_id=i32(n),
            generation=i32(n),
            birth_tick=i32(n),
        )


@struct
@dataclass(frozen=True)
class Transform:
    """Pose and velocity (§5.3).  pos is continuous, theta is wrapped to [-pi, pi)."""

    pos: jax.Array
    theta: jax.Array
    v: jax.Array
    omega: jax.Array

    @staticmethod
    def zeros(n: int) -> "Transform":
        return Transform(pos=f32(n, 2), theta=f32(n), v=f32(n), omega=f32(n))


@struct
@dataclass(frozen=True)
class Shape:
    radius: jax.Array
    mass: jax.Array

    @staticmethod
    def zeros(n: int) -> "Shape":
        return Shape(radius=f32(n), mass=f32(n))


@struct
@dataclass(frozen=True)
class Look:
    """Visible colour and odour spectrum.  Both are hereditary (§5.1, §4.2)."""

    rgb: jax.Array
    odor: jax.Array

    @staticmethod
    def zeros(n: int, odor_channels: int) -> "Look":
        return Look(rgb=f32(n, 3), odor=f32(n, odor_channels))


@struct
@dataclass(frozen=True)
class Body:
    """Body state (§5.2).  Only stomach fill is readable by the agent; every other
    quantity reaches it through the pain bundle, the heartbeat, or control error."""

    G: jax.Array
    N: jax.Array
    B: jax.Array
    I: jax.Array
    X: jax.Array
    b: jax.Array
    k: jax.Array
    p: jax.Array
    phi_hb: jax.Array
    S: jax.Array
    asleep: jax.Array
    age: jax.Array
    mu: jax.Array

    @staticmethod
    def zeros(n: int) -> "Body":
        return Body(
            G=f32(n, N_NUTRIENTS),
            N=f32(n, N_NUTRIENTS),
            B=f32(n, N_NUTRIENTS),
            I=f32(n),
            X=f32(n),
            b=f32(n),
            k=f32(n),
            p=f32(n),
            phi_hb=f32(n),
            S=f32(n),
            asleep=boolean(n),
            age=i32(n),
            mu=f32(n),
        )


@struct
@dataclass(frozen=True)
class Pain:
    """Pain bundle carrier states (§5.6).  x is the AR(2) state of each source."""

    a_inj: jax.Array
    x: jax.Array

    @staticmethod
    def zeros(n: int) -> "Pain":
        return Pain(a_inj=f32(n), x=f32(n, N_PAIN_SOURCES, 2))


@struct
@dataclass(frozen=True)
class Damage:
    """Last damage source and this tick's injury (SANDBOX.md §5.5, §5.6).

    The persistent fields (cause, source_uid, same_lineage, nutrient) survive
    until overwritten, so the die system can report the cause.  injury is the
    integrity damage accumulated during the current tick; metabolism turns it
    into acute pain and then clears it.
    """

    cause: jax.Array
    source_uid: jax.Array
    same_lineage: jax.Array
    nutrient: jax.Array
    injury: jax.Array

    @staticmethod
    def zeros(n: int) -> "Damage":
        return Damage(
            cause=i8(n),
            source_uid=jnp.full((n,), NO_REF, dtype=jnp.int32),
            same_lineage=boolean(n),
            nutrient=jnp.full((n,), NO_REF, dtype=jnp.int8),
            injury=f32(n),
        )


@struct
@dataclass(frozen=True)
class Holding:
    """What this creature holds; validated against the target's uid on read."""

    kind: jax.Array
    slot: jax.Array
    uid: jax.Array

    @staticmethod
    def zeros(n: int) -> "Holding":
        return Holding(kind=jnp.full((n,), NO_REF, dtype=jnp.int32), slot=i32(n), uid=i32(n))


@struct
@dataclass(frozen=True)
class HeldBy:
    """Who holds this entity; validated against the holder's uid on read."""

    kind: jax.Array
    slot: jax.Array
    uid: jax.Array

    @staticmethod
    def zeros(n: int) -> "HeldBy":
        return HeldBy(kind=jnp.full((n,), NO_REF, dtype=jnp.int32), slot=i32(n), uid=i32(n))


@struct
@dataclass(frozen=True)
class Grip:
    holding: Holding
    held_by: HeldBy

    @staticmethod
    def zeros(n: int) -> "Grip":
        return Grip(holding=Holding.zeros(n), held_by=HeldBy.zeros(n))


@struct
@dataclass(frozen=True)
class Contact:
    """Previous-tick contact force per quadrant: front, right, back, left.

    Magnitude is accumulated penetration depth after collision resolution
    (documented decision: docs/SANDBOX.md §3.2 leaves the unit open).
    """

    touch: jax.Array

    @staticmethod
    def zeros(n: int) -> "Contact":
        return Contact(touch=f32(n, 4))


@struct
@dataclass(frozen=True)
class BushEdible:
    stock: jax.Array
    kind: jax.Array

    @staticmethod
    def zeros(n: int) -> "BushEdible":
        return BushEdible(stock=f32(n), kind=i32(n))


@struct
@dataclass(frozen=True)
class CarcassEdible:
    n: jax.Array
    age_dead: jax.Array

    @staticmethod
    def zeros(n: int) -> "CarcassEdible":
        return CarcassEdible(n=f32(n, N_NUTRIENTS), age_dead=i32(n))


@struct
@dataclass(frozen=True)
class Genome:
    g: jax.Array

    @staticmethod
    def zeros(n: int, genome_dim: int) -> "Genome":
        return Genome(g=f32(n, genome_dim))


@struct
@dataclass(frozen=True)
class BrainState:
    """Default brain state (SANDBOX.md §6.1).

    h: recurrent state; s_hat: previous prediction; w_fast: plastic weights;
    last_pre/last_post: pre and post activities of the previous step, used by
    the Hebbian rule.
    """

    h: jax.Array
    s_hat: jax.Array
    w_fast: jax.Array
    last_pre: jax.Array
    last_post: jax.Array

    @staticmethod
    def zeros(n: int, h: int, slice_dim: int, plastic_dim: int) -> "BrainState":
        return BrainState(
            h=f32(n, h),
            s_hat=f32(n, slice_dim),
            w_fast=f32(n, plastic_dim),
            last_pre=f32(n, slice_dim),
            last_post=f32(n, h),
        )


@struct
@dataclass(frozen=True)
class Motor:
    last_out: jax.Array

    @staticmethod
    def zeros(n: int) -> "Motor":
        return Motor(last_out=f32(n, N_OUTPUTS))


@struct
@dataclass(frozen=True)
class Scripted:
    """State of the neutral-object walker (§4.5)."""

    heading: jax.Array
    timer: jax.Array

    @staticmethod
    def zeros(n: int) -> "Scripted":
        return Scripted(heading=f32(n), timer=i32(n))


@struct
@dataclass(frozen=True)
class Focal:
    """Observer-only trace slot (§7.3).  Never visible to the agent."""

    trace_slot: jax.Array

    @staticmethod
    def zeros(n: int) -> "Focal":
        return Focal(trace_slot=jnp.full((n,), -1, dtype=jnp.int32))


@struct
@dataclass(frozen=True)
class CreatureArena:
    alive: jax.Array
    identity: Identity
    transform: Transform
    shape: Shape
    look: Look
    body: Body
    pain: Pain
    damage: Damage
    grip: Grip
    contact: Contact
    genome: Genome
    brain: BrainState
    motor: Motor
    focal: Focal


@struct
@dataclass(frozen=True)
class BushArena:
    """Bushes are static, but they still need a position for vision, odour and
    collision.  SANDBOX.md §3.2 lists Transform only for creature/carcass/neutral;
    giving bushes a transform with zero velocity is the documented deviation."""

    alive: jax.Array
    uid: jax.Array
    transform: Transform
    shape: Shape
    look: Look
    edible: BushEdible


@struct
@dataclass(frozen=True)
class CarcassArena:
    alive: jax.Array
    uid: jax.Array
    transform: Transform
    shape: Shape
    look: Look
    held_by: HeldBy
    edible: CarcassEdible


@struct
@dataclass(frozen=True)
class NeutralArena:
    alive: jax.Array
    uid: jax.Array
    transform: Transform
    shape: Shape
    look: Look
    held_by: HeldBy
    scripted: Scripted


@struct
@dataclass(frozen=True)
class Entities:
    """All prototypes of one world, each in its own fixed-capacity arena (§3.1)."""

    creature: CreatureArena
    bush: BushArena
    carcass: CarcassArena
    neutral: NeutralArena

    @staticmethod
    def empty(cfg) -> "Entities":
        n_creature = cfg.capacities.creature
        n_bush = cfg.capacities.bush
        n_carcass = cfg.capacities.carcass
        n_neutral = cfg.capacities.neutral
        odor_channels = cfg.sensors.odor_channels
        return Entities(
            creature=CreatureArena(
                alive=boolean(n_creature),
                identity=Identity.zeros(n_creature),
                transform=Transform.zeros(n_creature),
                shape=Shape.zeros(n_creature),
                look=Look.zeros(n_creature, odor_channels),
                body=Body.zeros(n_creature),
                pain=Pain.zeros(n_creature),
                damage=Damage.zeros(n_creature),
                grip=Grip.zeros(n_creature),
                contact=Contact.zeros(n_creature),
                genome=Genome.zeros(n_creature, cfg.brain.genome_dim),
                brain=BrainState.zeros(
                    n_creature,
                    cfg.brain.h,
                    cfg.sensors.slice_dim,
                    cfg.brain.plastic_dim,
                ),
                motor=Motor.zeros(n_creature),
                focal=Focal.zeros(n_creature),
            ),
            bush=BushArena(
                alive=boolean(n_bush),
                uid=i32(n_bush),
                transform=Transform.zeros(n_bush),
                shape=Shape.zeros(n_bush),
                look=Look.zeros(n_bush, odor_channels),
                edible=BushEdible.zeros(n_bush),
            ),
            carcass=CarcassArena(
                alive=boolean(n_carcass),
                uid=i32(n_carcass),
                transform=Transform.zeros(n_carcass),
                shape=Shape.zeros(n_carcass),
                look=Look.zeros(n_carcass, odor_channels),
                held_by=HeldBy.zeros(n_carcass),
                edible=CarcassEdible.zeros(n_carcass),
            ),
            neutral=NeutralArena(
                alive=boolean(n_neutral),
                uid=i32(n_neutral),
                transform=Transform.zeros(n_neutral),
                shape=Shape.zeros(n_neutral),
                look=Look.zeros(n_neutral, odor_channels),
                held_by=HeldBy.zeros(n_neutral),
                scripted=Scripted.zeros(n_neutral),
            ),
        )
