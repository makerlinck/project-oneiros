"""motion system (§3.3 step 6, §5.3).

Contract: `apply(state, cfg) -> (WorldState, EventBuffer)` — a pure function.
Randomness comes only from the per-tick motion key, so opening or closing any
observer cannot change the motion noise stream.

Implements SANDBOX.md §5.3:

    u'    = clip(u + N(0, sigma_m^2), -1, 1)   sigma_m = noise_scale*0.05*(1+2*(1-I))
    c_I   = 0.3 + 0.7*I
    g     = (0.3 + 0.7*mu)*c_I
    a_max = 0.05*(1+k)*g / (1 + m_held/m)
    v     = clip((v + a_max*u_acc')*(1 - 0.1), +-(0.5*(1 + 0.5k)))
    theta = theta + 0.3*c_I*u_turn'            (wrapped to [-pi, pi))
    pos   = pos + v*(cos theta, sin theta)

Only the two motor outputs (u_acc, u_turn) are perturbed here; the interaction
outputs are read by the interact system as intended values.  The body system
derives k (the metabolic knob) from the same intended outputs.

Documented decisions (SANDBOX.md is silent):
- Sleep: body outputs are invalid, so u_acc' = u_turn' = 0, k = 0 for the speed
  cap and a_max, and the velocity still decays under drag while the position
  integrates the decaying velocity.
- A held entity sits at the holder's contact distance: pos = pos_h + (r_h+r_t)
  * (cos theta_h, sin theta_h), theta = theta_h, v = v_h.  Following is applied
  up to a depth of 3 in one tick so a short hold chain settles; longer chains
  converge over successive ticks.  Carcasses and neutral objects follow the
  same rule; bushes can never be held (§5.4).
- omega stores the angular velocity actually applied this tick (proprioception,
  §5.6), not the previous raw output.
- Motor.u_eff = [u_acc', u_turn'] is written for the systems that charge or
  gate on the applied controls (§5.3): metabolism's motion cost and the
  interaction escape rule read it instead of re-deriving the noise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.config import Config
from life_sandbox.core import components as C
from life_sandbox.core import events, masks, rng
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update

# SANDBOX.md §5.3 initial values.  Kept here as module constants because the
# config has no body section yet; M1 calibration lifts them into the record.
_ACC_SCALE: float = 0.05
_TURN_SCALE: float = 0.3
_DRAG: float = 0.1
_SPEED_CAP: float = 0.5
_NOISE_SCALE: float = 0.05

_HOLD_CHAIN_PASSES: int = 3


def intended_knob(state: WorldState) -> jax.Array:
    """Metabolic knob k for mechanics: 0 while asleep, else this tick's output.

    Motion runs before the body system (SANDBOX.md §3.3), so it must not read
    the previous tick's Body.k; the intended output is the same quantity the
    body system will store.
    """
    body = state.entities.creature.body
    last = state.entities.creature.motor.last_out
    knob = jnp.clip(last[:, C.OUT_K], 0.0, 1.0)
    return jnp.where(body.asleep, jnp.asarray(0.0, jnp.float32), knob)


def integrity_scale(body: C.Body) -> jax.Array:
    """c_I = 0.3 + 0.7*I: injury weakens push, turn, bite and grip (§5.3)."""
    return 0.3 + 0.7 * body.I


def body_efficiency(body: C.Body) -> jax.Array:
    """g = (0.3 + 0.7*mu)*c_I (§5.3, altricial form arrives in phase 5)."""
    return (0.3 + 0.7 * body.mu) * integrity_scale(body)


def max_acceleration(state: WorldState) -> jax.Array:
    """a_max = 0.05*(1+k)*g / (1 + m_held/m) (§5.3)."""
    ents = state.entities
    mass = ents.creature.shape.mass
    held = masks.held_mass(ents)
    ratio = held / jnp.maximum(mass, jnp.asarray(1e-6, jnp.float32))
    return _ACC_SCALE * (1.0 + intended_knob(state)) * body_efficiency(
        ents.creature.body
    ) / (1.0 + ratio)


def motion_noise(state: WorldState, cfg: Config) -> jax.Array:
    """Gaussian motor noise of shape (capacity, 2) from the motion key (§5.3).

    The interact system re-derives the same array for the escape rule (§5.4):
    keys are stateless, so reading it twice consumes nothing.
    """
    ents = state.entities
    count = ents.creature.alive.shape[0]
    key = rng.system_key(state.key, state.tick, "motion")
    draws = jax.random.normal(key, (count, 2), dtype=jnp.float32)
    sigma = cfg.motion.noise_scale * _NOISE_SCALE * (1.0 + 2.0 * (1.0 - ents.creature.body.I))
    return draws * sigma[:, None]


def perturbed_controls(state: WorldState, cfg: Config) -> tuple[jax.Array, jax.Array]:
    """u_acc' and u_turn' after noise and clipping; zero while asleep."""
    ents = state.entities
    noise = motion_noise(state, cfg)
    last = ents.creature.motor.last_out
    acc = jnp.clip(last[:, C.OUT_ACC] + noise[:, 0], -1.0, 1.0)
    turn = jnp.clip(last[:, C.OUT_TURN] + noise[:, 1], -1.0, 1.0)
    asleep = ents.creature.body.asleep
    return jnp.where(asleep, jnp.zeros_like(acc), acc), jnp.where(
        asleep, jnp.zeros_like(turn), turn
    )


def _wrap_angle(angle: jax.Array) -> jax.Array:
    """Wrap to [-pi, pi)."""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _follow_held(state: WorldState) -> WorldState:
    """Place every held entity at its holder's contact distance (§5.4)."""
    ents = state.entities
    creature = ents.creature

    def _place(arena_transform, arena_shape, arena_held_by):
        held_by = arena_held_by
        valid = masks.ref_valid(
            ents, held_by.kind, held_by.slot, held_by.uid
        ) & (held_by.kind == C.CREATURE)
        slot = jnp.clip(held_by.slot, 0, creature.alive.shape[0] - 1)
        radius_target = arena_shape.radius
        radius_holder = creature.shape.radius[slot]
        theta = creature.transform.theta[slot]
        pos = creature.transform.pos[slot] + (radius_holder + radius_target)[:, None] * jnp.stack(
            [jnp.cos(theta), jnp.sin(theta)], axis=1
        )
        transform = arena_transform
        transform = update(transform, pos=jnp.where(valid[:, None], pos, transform.pos))
        transform = update(
            transform, theta=jnp.where(valid, theta, transform.theta)
        )
        transform = update(
            transform, v=jnp.where(valid, creature.transform.v[slot], transform.v)
        )
        transform = update(
            transform, omega=jnp.where(valid, creature.transform.omega[slot], transform.omega)
        )
        return transform

    for _ in range(_HOLD_CHAIN_PASSES):
        # Held creatures first, so a carcass held by a held creature sees the
        # updated (carried) holder pose within the same tick.
        creature = ents.creature
        new_transform = _place(creature.transform, creature.shape, creature.grip.held_by)
        creature = update(creature, transform=new_transform)
        ents = update(ents, creature=creature)

        carcass = ents.carcass
        new_transform = _place(carcass.transform, carcass.shape, carcass.held_by)
        carcass = update(carcass, transform=new_transform)
        ents = update(ents, carcass=carcass)

        neutral = ents.neutral
        new_transform = _place(neutral.transform, neutral.shape, neutral.held_by)
        neutral = update(neutral, transform=new_transform)
        ents = update(ents, neutral=neutral)

    return update(state, entities=ents)


def apply(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    """Integrate one tick of motion (SANDBOX.md §3.3 step 6, §5.3)."""
    ents = state.entities
    creature = ents.creature
    body = creature.body
    transform = creature.transform
    alive = creature.alive

    acc, turn = perturbed_controls(state, cfg)
    knob = intended_knob(state)

    speed = transform.v + max_acceleration(state) * acc
    speed = speed * (1.0 - _DRAG)
    cap = _SPEED_CAP * (1.0 + 0.5 * knob)
    speed = jnp.clip(speed, -cap, cap)

    omega = _TURN_SCALE * integrity_scale(body) * turn
    theta = _wrap_angle(transform.theta + omega)
    pos = transform.pos + speed[:, None] * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)

    new_transform = C.Transform(
        pos=jnp.where(alive[:, None], pos, transform.pos),
        theta=jnp.where(alive, theta, transform.theta),
        v=jnp.where(alive, speed, transform.v),
        omega=jnp.where(alive, omega, transform.omega),
    )
    # Effective controls u' ([u_acc', u_turn']) for the systems that charge or
    # gate on what motion actually applied (§5.3): store on Motor so body and
    # interaction never re-derive the motion noise from the key.
    u_eff = jnp.where(alive[:, None], jnp.stack([acc, turn], axis=1), 0.0)
    creature = update(
        creature,
        transform=new_transform,
        motor=update(creature.motor, u_eff=u_eff.astype(jnp.float32)),
    )
    ents = update(ents, creature=creature)
    state = update(state, entities=ents)
    state = _follow_held(state)
    return state, events.empty(cfg)
