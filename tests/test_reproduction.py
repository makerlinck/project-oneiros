"""Reproduction: C5 birth budget, deferred parents and keyed mutation noise.

SANDBOX.md §5.5 / §6.3.  The per-tick birth budget is B_MAX; eligible parents
beyond it are deferred without paying and emit `birth_defer`.  Mutation noise
is drawn per actual birth from fold_in(reproduce_key, child_uid), so a birth's
genome depends only on (reproduce_key, child_uid) and never on the arena
capacity or on the other births of the same tick.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from life_sandbox.brain import genome as brain_genome
from life_sandbox.config import Capacities, Config, MotionConfig
from life_sandbox.core import components as C, events, rng
from life_sandbox.core.hashing import state_hash
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import reproduction

FLESH = jnp.asarray([0.3, 0.7, 0.0], jnp.float32)
START_UID = 1000
PARENT_RESERVES = (0.17, 0.17, 0.17)  # sum 0.51: just above theta_repro = 0.5


def make_cfg(capacity: int = 40) -> Config:
    return Config.default().replaced(
        capacities=Capacities(creature=capacity, bush=6, carcass=2, neutral=1),
        motion=MotionConfig(noise_scale=0.0),
    )


def base_state(cfg: Config):
    state = empty(cfg)
    lay = brain_genome.layout(cfg)
    cre = state.entities.creature
    cre = update(
        cre,
        genome=C.Genome(
            g=jnp.zeros((cfg.capacities.creature, lay.total), jnp.float32)
        ),
    )
    return update(
        state,
        entities=update(state.entities, creature=cre),
        uid_counter=jnp.asarray(START_UID, jnp.int32),
    )


def add_parent(
    state,
    cfg: Config,
    slot: int,
    *,
    pos,
    uid: int,
    reserves=PARENT_RESERVES,
    tissue=(0.15, 0.35, 0.0),
    maturity: float = 1.0,
):
    cre = state.entities.creature
    cre = update(
        cre,
        alive=cre.alive.at[slot].set(True),
        identity=update(
            cre.identity,
            uid=cre.identity.uid.at[slot].set(uid),
            lineage_id=cre.identity.lineage_id.at[slot].set(uid),
            generation=cre.identity.generation.at[slot].set(1),
        ),
        transform=update(
            cre.transform,
            pos=cre.transform.pos.at[slot].set(jnp.asarray(pos, jnp.float32)),
            theta=cre.transform.theta.at[slot].set(0.0),
        ),
        shape=update(
            cre.shape,
            radius=cre.shape.radius.at[slot].set(0.5),
            mass=cre.shape.mass.at[slot].set(1.0),
        ),
        body=update(
            cre.body,
            B=cre.body.B.at[slot].set(jnp.asarray(tissue, jnp.float32)),
            N=cre.body.N.at[slot].set(jnp.asarray(reserves, jnp.float32)),
            I=cre.body.I.at[slot].set(1.0),
            mu=cre.body.mu.at[slot].set(maturity),
        ),
    )
    return update(state, entities=update(state.entities, creature=cre))


def grid_position(index: int, columns: int = 10, spacing: float = 3.0):
    return (3.0 + spacing * (index % columns), 3.0 + spacing * (index // columns))


def many_parents(cfg: Config, count: int):
    state = base_state(cfg)
    for index in range(count):
        state = add_parent(
            state, cfg, index, pos=grid_position(index), uid=1 + index
        )
    return state


def records(buffer) -> list[dict]:
    result = []
    for i in range(int(buffer.count)):
        result.append(
            {
                "kind": int(buffer.kind[i]),
                "uid": int(buffer.uid[i]),
                "target_kind": int(buffer.target_kind[i]),
                "target_uid": int(buffer.target_uid[i]),
                "success": bool(buffer.success[i]),
                "reason": int(buffer.reason[i]),
            }
        )
    return result


def of_kind(buffer, name: str) -> list[dict]:
    code = events.kind_code(name)
    return [r for r in records(buffer) if r["kind"] == code]


def test_budget_caps_births_and_defers_the_rest():
    cfg = make_cfg(40)
    parents = 20
    state = many_parents(cfg, parents)
    out, buffer = reproduction.apply(state, cfg)

    births = of_kind(buffer, "birth")
    deferred = of_kind(buffer, "birth_defer")
    failures = of_kind(buffer, "birth_fail")
    assert len(births) == reproduction.B_MAX == 16
    assert len(deferred) == parents - reproduction.B_MAX
    assert failures == []
    assert int(jnp.sum(out.entities.creature.alive)) == parents + reproduction.B_MAX
    assert int(out.uid_counter) == START_UID + reproduction.B_MAX

    # Every deferral event names the deferred parent and is a deferral, not a
    # failure; deferred parents paid nothing this tick.
    cre = out.entities.creature
    deferred_uids = {row["uid"] for row in deferred}
    for row in deferred:
        assert row["success"]
        assert row["target_kind"] == C.CREATURE
        assert row["target_uid"] == row["uid"]
        assert row["reason"] == events.reason_code("none")
    for slot in range(parents):
        parent_uid = int(cre.identity.uid[slot])
        if parent_uid in deferred_uids:
            assert jnp.allclose(cre.body.N[slot], jnp.asarray(PARENT_RESERVES), atol=1e-7)
        else:
            assert jnp.sum(cre.body.N[slot]) < 0.5


def test_deferred_parents_birth_next_tick():
    cfg = make_cfg(40)
    parents = 20
    state = many_parents(cfg, parents)
    first, first_buffer = reproduction.apply(state, cfg)
    deferred_uids = {row["uid"] for row in of_kind(first_buffer, "birth_defer")}

    second_state = update(first, tick=(first.tick + 1).astype(jnp.int32))
    out, buffer = reproduction.apply(second_state, cfg)
    births = of_kind(buffer, "birth")
    deferred = of_kind(buffer, "birth_defer")
    assert len(births) == parents - reproduction.B_MAX  # exactly the deferred ones
    assert deferred == []
    assert int(jnp.sum(out.entities.creature.alive)) == parents + parents
    assert int(out.uid_counter) == START_UID + parents
    born_uids = {row["target_uid"] for row in births}
    assert born_uids == deferred_uids


def test_mutation_noise_is_keyed_by_child_uid():
    """Same (reproduce_key, child_uid) -> same child genome, whatever the arena."""
    small = make_cfg(8)
    base = base_state(small)
    base = add_parent(base, small, 0, pos=(10.0, 10.0), uid=1)
    out, buffer = reproduction.apply(base, small)
    births = of_kind(buffer, "birth")
    assert len(births) == 1
    child_uid = births[0]["uid"]
    assert child_uid == START_UID
    child_genome = out.entities.creature.genome.g[1]
    parent_genome = base.entities.creature.genome.g[0]

    key = rng.system_key(base.key, base.tick, "reproduce")
    lay = brain_genome.layout(small)
    sigma = brain_genome.mutation_sigmas(small)
    expected = parent_genome + (
        jax.random.normal(jax.random.fold_in(key, child_uid), (lay.total,), dtype=jnp.float32)
        * sigma
    )
    assert jnp.allclose(child_genome, expected, atol=1e-7)

    # A larger arena (capacity changes) yields exactly the same child genome.
    large = make_cfg(64)
    base_large = base_state(large)
    base_large = add_parent(base_large, large, 0, pos=(10.0, 10.0), uid=1)
    out_large, _ = reproduction.apply(base_large, large)
    assert int(out_large.entities.creature.identity.uid[1]) == child_uid
    assert jnp.array_equal(out_large.entities.creature.genome.g[1], child_genome)


def test_birth_is_independent_of_other_births():
    """A child's genome does not change when a second parent also gives birth."""
    cfg = make_cfg(40)
    alone = base_state(cfg)
    alone = add_parent(alone, cfg, 0, pos=grid_position(0), uid=1)
    alone_out, _ = reproduction.apply(alone, cfg)
    child_alone = alone_out.entities.creature.genome.g[1]

    crowded = many_parents(cfg, 2)
    crowded_out, buffer = reproduction.apply(crowded, cfg)
    births = of_kind(buffer, "birth")
    assert len(births) == 2
    first_child_uid = min(row["uid"] for row in births)
    assert first_child_uid == START_UID
    child_slot = int(
        jnp.where(crowded_out.entities.creature.identity.uid == first_child_uid)[0][0]
    )
    assert jnp.array_equal(
        crowded_out.entities.creature.genome.g[child_slot], child_alone
    )


def test_reproduction_compiled_and_eager_agree():
    """Same input state: compiled and eager agree; same-mode runs are bit-equal."""
    cfg = make_cfg(40)
    state = many_parents(cfg, 20)
    eager_out, eager_buffer = reproduction.apply(state, cfg)
    jitted = jax.jit(lambda s: reproduction.apply(s, cfg))
    jit_out, jit_buffer = jitted(state)

    eager_events = records(eager_buffer)
    jit_events = records(jit_buffer)
    assert eager_events == jit_events
    # Float32 ULP differences from XLA fusion are allowed in the ledger only;
    # everything else is compared bitwise.
    assert state_hash(update(eager_out, ledger=C.Ledger.zeros())) == state_hash(
        update(jit_out, ledger=C.Ledger.zeros())
    )
    assert np.allclose(
        np.asarray(eager_out.ledger.outflow, np.float64),
        np.asarray(jit_out.ledger.outflow, np.float64),
        rtol=1e-6,
        atol=1e-9,
    )
    # The compiled execution mode is bit-reproducible with itself (SANDBOX §2).
    again_out, _ = jitted(state)
    assert state_hash(again_out) == state_hash(jit_out)
