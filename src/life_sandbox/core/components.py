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

# Decaying damage sources (SANDBOX.md §5.5).  loss[source] accumulates integrity
# damage with an exponential decay (tau = 50 ticks); the largest decayed source
# at death is the reported cause.
LOSS_NUTRITION: int = 0
LOSS_TOXIN: int = 1
LOSS_HEAT: int = 2
LOSS_COLD: int = 3
LOSS_BITE: int = 4
N_DAMAGE_SOURCES: int = 5
LOSS_CAUSES: tuple[str, ...] = ("nutrition", "toxin", "heat", "cold", "bite")
DAMAGE_DECAY_TAU: float = 50.0

# Pain bundle channels (§5.6): nine sources mixed into four shared channels.
# The width is a physical property of the afferent bundle, independent of the
# odour channel count even though both are four today.
N_PAIN_CHANNELS: int = 4

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

# Nutrient ledger categories (SANDBOX.md §3.5).  Inflow and outflow are split by
# reason so a segment report can attribute where mass crossed the book boundary.
INFLOW_BITES: int = 0
INFLOW_RESEED: int = 1
N_INFLOW_CATEGORIES: int = 2
INFLOW_NAMES: tuple[str, ...] = ("bites", "reseed")

OUTFLOW_METABOLISM: int = 0
OUTFLOW_EXCRETION: int = 1
OUTFLOW_OVERFLOW: int = 2
OUTFLOW_TOXIN_LOAD: int = 3
OUTFLOW_BIRTH_LOSS: int = 4
OUTFLOW_DEATH_LOSS: int = 5
OUTFLOW_DECOMPOSITION: int = 6
N_OUTFLOW_CATEGORIES: int = 7
OUTFLOW_NAMES: tuple[str, ...] = (
    "metabolism",
    "excretion",
    "overflow",
    "toxin_load",
    "birth_loss",
    "death_loss",
    "decomposition",
)

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
    """Decaying per-source damage ledger (SANDBOX.md §5.5).

    loss[source] accumulates integrity damage with an exponential decay of
    DAMAGE_DECAY_TAU ticks; the die system reports the largest decayed source.
    It is observation-only and invisible to the agent.  injury is this tick's
    bite damage (acute pain); the nutrient index records the last missing
    nutrient for nutrition deaths.
    """

    source_uid: jax.Array
    same_lineage: jax.Array
    nutrient: jax.Array
    injury: jax.Array
    loss: jax.Array

    @staticmethod
    def zeros(n: int) -> "Damage":
        return Damage(
            source_uid=jnp.full((n,), NO_REF, dtype=jnp.int32),
            same_lineage=boolean(n),
            nutrient=jnp.full((n,), NO_REF, dtype=jnp.int8),
            injury=f32(n),
            loss=f32(n, N_DAMAGE_SOURCES),
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

    h: recurrent state; s_hat: prediction made for the next slice; w_fast:
    plastic weights; e: prediction error e_t = s_t - s_hat_t computed once by
    think and reused bit-exactly by learn; h_prev: h_(t-1), the pre activity of
    W_rec and of the prediction delta rule.
    """

    h: jax.Array
    s_hat: jax.Array
    w_fast: jax.Array
    e: jax.Array
    h_prev: jax.Array

    @staticmethod
    def zeros(n: int, h: int, slice_dim: int, plastic_dim: int) -> "BrainState":
        return BrainState(
            h=f32(n, h),
            s_hat=f32(n, slice_dim),
            w_fast=f32(n, plastic_dim),
            e=f32(n, slice_dim),
            h_prev=f32(n, h),
        )


@struct
@dataclass(frozen=True)
class Motor:
    """Intended outputs and the effective controls after motion noise (§5.3).

    last_out keeps the raw brain outputs; u_eff = [u_acc', u_turn'] is what
    motion actually applied (noise added, clipped).  Other systems (metabolism,
    interaction) read u_eff instead of re-deriving the noise from the motion key.
    """

    last_out: jax.Array
    u_eff: jax.Array

    @staticmethod
    def zeros(n: int) -> "Motor":
        return Motor(last_out=f32(n, N_OUTPUTS), u_eff=f32(n, 2))


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
class Ledger:
    """Per-tick nutrient flow draft area (SANDBOX.md §3.5).

    step clears it at the start of every tick and systems accumulate the flows
    where they happen; step returns it as a per-tick scan output, so the host
    sums in float64 and nothing accumulates across ticks inside the state.  It
    is a global resource, never a sensor: the agent cannot see it.
    """

    inflow: jax.Array
    outflow: jax.Array
    transfer_from: jax.Array
    transfer_to: jax.Array
    digested: jax.Array

    @staticmethod
    def zeros() -> "Ledger":
        return Ledger(
            inflow=f32(N_INFLOW_CATEGORIES, N_NUTRIENTS),
            outflow=f32(N_OUTFLOW_CATEGORIES, N_NUTRIENTS),
            transfer_from=f32(N_NUTRIENTS),
            transfer_to=f32(N_NUTRIENTS),
            digested=f32(N_NUTRIENTS),
        )


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
        # Local import: brain.genome imports this module, so a module-level
        # import here would be a cycle.  Genome and fast-weight widths are
        # derived from the selected brain layout (SANDBOX.md §6.3), never from
        # hand-written config numbers.
        from life_sandbox.brain.genome import layout as brain_layout

        lay = brain_layout(cfg)
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
                genome=Genome.zeros(n_creature, lay.total),
                brain=BrainState.zeros(n_creature, cfg.brain.h, lay.d, lay.plastic_dim),
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
