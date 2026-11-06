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
from life_sandbox.config import Capacities, Config, MotionConfig, ReproductionConfig
from life_sandbox.core import components as C, events, genome as core_genome, rng
from life_sandbox.core.hashing import state_hash
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import reproduction
from life_sandbox.world import init as world_init

FLESH = jnp.asarray([0.3, 0.7, 0.0], jnp.float32)
START_UID = 1000
PARENT_RESERVES = (0.17, 0.17, 0.17)  # sum 0.51: just above theta_repro = 0.5


def make_cfg(
    capacity: int = 40, *, randomize: bool = False, from_reserves: bool = True
) -> Config:
    return Config.default().replaced(
        capacities=Capacities(creature=capacity, bush=6, carcass=2, neutral=1),
        motion=MotionConfig(noise_scale=0.0),
        reproduction=ReproductionConfig(
            randomize_genome=randomize, tissue_from_reserves=from_reserves
        ),
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
                "parent_uid": int(buffer.parent_uid[i]),
                "target_kind": int(buffer.target_kind[i]),
                "target_uid": int(buffer.target_uid[i]),
                "success": bool(buffer.success[i]),
                "reason": int(buffer.reason[i]),
                "nutrient": int(buffer.nutrient[i]),
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
        assert row["parent_uid"] == C.NO_REF  # birth_defer: uid is the parent
        assert row["reason"] == events.reason_code("none")
    for row in births:
        assert row["parent_uid"] == row["target_uid"]  # birth: parent uid
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


def _with_modified_parent_genome(state, cfg, slot: int, value: float = 3.0):
    """Change the parent's weight block only (eligibility header untouched)."""
    lay = brain_genome.layout(cfg)
    cre = state.entities.creature
    genome = cre.genome.g.at[slot, lay.w_in].set(value)
    return update(
        state, entities=update(state.entities, creature=update(cre, genome=C.Genome(g=genome)))
    )


def test_randomize_genome_arm_is_parent_independent():
    """cfg.reproduction.randomize_genome: child genome ignores the parent (§9 M1)."""
    cfg = make_cfg(8, randomize=True)
    base = add_parent(base_state(cfg), cfg, 0, pos=(10.0, 10.0), uid=1)
    out_a, buffer_a = reproduction.apply(base, cfg)
    assert len(of_kind(buffer_a, "birth")) == 1
    child_a = out_a.entities.creature.genome.g[1]

    modified = _with_modified_parent_genome(base, cfg, 0)
    out_b, buffer_b = reproduction.apply(modified, cfg)
    assert len(of_kind(buffer_b, "birth")) == 1
    child_b = out_b.entities.creature.genome.g[1]
    assert jnp.array_equal(child_a, child_b)  # parent genome is ignored
    assert not jnp.array_equal(child_a, base.entities.creature.genome.g[0])
    # Founder convention from world/init.random_genome: theta_repro ~ U[0.5, 1].
    theta_index = core_genome.theta_repro_index(cfg.sensors.odor_channels)
    assert 0.5 <= float(child_a[theta_index]) <= 1.0
    assert float(jnp.max(jnp.abs(child_a))) > 0.0


def test_randomize_genome_is_keyed_by_child_uid():
    """Fresh random genomes are reproducible per child_uid and distinct per child."""
    cfg = make_cfg(8, randomize=True)
    base = add_parent(base_state(cfg), cfg, 0, pos=(10.0, 10.0), uid=1)
    first, _ = reproduction.apply(base, cfg)
    again, _ = reproduction.apply(base, cfg)
    assert jnp.array_equal(
        first.entities.creature.genome.g[1], again.entities.creature.genome.g[1]
    )

    crowded = many_parents(cfg, 2)
    crowded_out, buffer = reproduction.apply(crowded, cfg)
    births = of_kind(buffer, "birth")
    assert len(births) == 2
    rows = []
    for row in births:
        slot = int(
            jnp.where(crowded_out.entities.creature.identity.uid == row["uid"])[0][0]
        )
        rows.append(crowded_out.entities.creature.genome.g[slot])
    assert not jnp.array_equal(rows[0], rows[1])


def test_default_arm_still_uses_parent_plus_mutation():
    """The default arm (randomize_genome=False) keeps parent + sigma * N(0,1)."""
    cfg = make_cfg(8)
    assert cfg.reproduction.randomize_genome is False
    state = add_parent(base_state(cfg), cfg, 0, pos=(10.0, 10.0), uid=1)
    out, _ = reproduction.apply(state, cfg)
    parent = state.entities.creature.genome.g[0]
    child = out.entities.creature.genome.g[1]
    key = rng.system_key(state.key, state.tick, "reproduce")
    lay = brain_genome.layout(cfg)
    sigma = brain_genome.mutation_sigmas(cfg)
    expected = parent + jax.random.normal(
        jax.random.fold_in(key, START_UID), (lay.total,), dtype=jnp.float32
    ) * sigma
    assert jnp.allclose(child, expected, atol=1e-7)


def test_randomize_genome_arm_compiles_and_matches_eager():
    """The control arm is jit-safe and reproducible on the same input state."""
    cfg = make_cfg(8, randomize=True)
    state = add_parent(base_state(cfg), cfg, 0, pos=(10.0, 10.0), uid=1)
    eager_out, eager_buffer = reproduction.apply(state, cfg)
    jitted = jax.jit(lambda s: reproduction.apply(s, cfg))
    jit_out, jit_buffer = jitted(state)
    assert records(eager_buffer) == records(jit_buffer)
    assert state_hash(update(eager_out, ledger=C.Ledger.zeros())) == state_hash(
        update(jit_out, ledger=C.Ledger.zeros())
    )
    assert jnp.array_equal(
        eager_out.entities.creature.genome.g[1], jit_out.entities.creature.genome.g[1]
    )


def test_reserve_payment_keeps_adult_tissue():
    """Default arm: B_child = 0.03*c_flesh comes from N; the adult B stays."""
    cfg = make_cfg(8)
    state = base_state(cfg)
    parent_n = jnp.asarray([0.5, 0.5, 0.5])
    parent_b = jnp.asarray([0.15, 0.35, 0.0])
    state = add_parent(
        state, cfg, 0, pos=(10.0, 10.0), uid=1,
        reserves=tuple(parent_n), tissue=tuple(parent_b),
    )
    out, buffer = reproduction.apply(state, cfg)
    births = of_kind(buffer, "birth")
    assert len(births) == 1
    assert births[0]["uid"] == START_UID
    assert births[0]["parent_uid"] == 1
    assert births[0]["nutrient"] == -1  # success keeps the neutral value
    cre = out.entities.creature
    tissue = reproduction.CHILD_TISSUE * FLESH
    child_rho = world_init.diet_from_genome(cre.genome.g[1:2], cfg)[0]
    expected_child_n = jnp.minimum(0.3 * child_rho, parent_n - tissue)
    remaining = parent_n - tissue - expected_child_n
    expected_loss = jnp.minimum(0.02 * jnp.ones(3) / 3.0, remaining)
    assert jnp.array_equal(cre.body.B[0], parent_b)  # adult B untouched
    assert jnp.allclose(cre.body.B[1], tissue, atol=1e-7)
    assert jnp.allclose(cre.body.N[1], expected_child_n, atol=1e-6)
    assert jnp.allclose(
        cre.body.N[0],
        parent_n - tissue - expected_child_n - expected_loss,
        atol=1e-6,
    )


def test_reserve_payment_fails_without_reserve():
    """One N component below B_child fails the birth with no_reserve (§5.5)."""
    cfg = make_cfg(8)
    state = base_state(cfg)
    reserves = jnp.asarray([0.001, 0.5, 0.5])  # first < 0.03*0.3
    tissue = jnp.asarray([0.15, 0.35, 0.0])
    state = add_parent(
        state, cfg, 0, pos=(10.0, 10.0), uid=9,
        reserves=tuple(reserves), tissue=tuple(tissue),
    )
    out, buffer = reproduction.apply(state, cfg)
    fails = of_kind(buffer, "birth_fail")
    assert len(fails) == 1
    assert fails[0]["reason"] == events.reason_code("no_reserve")
    assert fails[0]["uid"] == 9
    assert fails[0]["parent_uid"] == C.NO_REF
    assert fails[0]["nutrient"] == C.PLANT  # P is the blocked reserve
    cre = out.entities.creature
    assert not bool(cre.alive[1])
    assert jnp.allclose(cre.body.N[0], reserves, atol=1e-7)
    assert jnp.allclose(cre.body.B[0], tissue, atol=1e-7)


def test_reserve_payment_allows_many_births():
    """Without the B cap a fed parent can split many times in a row (§5.5)."""
    cfg = make_cfg(8)
    state = base_state(cfg)
    parent_b = jnp.asarray([0.15, 0.35, 0.0])
    state = add_parent(
        state, cfg, 0, pos=(10.0, 10.0), uid=1,
        reserves=(0.33, 0.33, 0.33), tissue=tuple(parent_b),
    )
    births = 0
    for cycle in range(5):
        cre = state.entities.creature
        # Simulate feeding between births.
        cre = update(
            cre,
            body=update(
                cre.body, N=cre.body.N.at[0].set(jnp.asarray([0.33, 0.33, 0.33], jnp.float32))
            ),
        )
        state = update(
            state,
            entities=update(state.entities, creature=cre),
            tick=(state.tick + jnp.asarray(1, jnp.int32)),
        )
        out, buffer = reproduction.apply(state, cfg)
        new_births = of_kind(buffer, "birth")
        assert len(new_births) == 1, (
            cycle,
            [row["reason"] for row in of_kind(buffer, "birth_fail")],
        )
        assert jnp.array_equal(out.entities.creature.body.B[0], parent_b)
        state = out
        births += 1
    assert births == 5
    assert int(jnp.sum(state.entities.creature.alive)) == 6
    assert int(state.uid_counter) == START_UID + 5


def test_control_arm_keeps_old_tissue_payment():
    """tissue_from_reserves=false reproduces the old B payment (A/B control)."""
    cfg = make_cfg(8, from_reserves=False)
    state = base_state(cfg)
    parent_n = jnp.asarray([0.5, 0.5, 0.5])
    parent_b = jnp.asarray([0.15, 0.35, 0.0])
    state = add_parent(
        state, cfg, 0, pos=(10.0, 10.0), uid=1,
        reserves=tuple(parent_n), tissue=tuple(parent_b),
    )
    out, buffer = reproduction.apply(state, cfg)
    assert len(of_kind(buffer, "birth")) == 1
    cre = out.entities.creature
    tissue = reproduction.CHILD_TISSUE * FLESH
    child_rho = world_init.diet_from_genome(cre.genome.g[1:2], cfg)[0]
    expected_child_n = jnp.minimum(0.3 * child_rho, parent_n)
    remaining = parent_n - expected_child_n
    expected_loss = jnp.minimum(0.02 * jnp.ones(3) / 3.0, remaining)
    assert jnp.allclose(cre.body.N[1], expected_child_n, atol=1e-6)
    assert jnp.allclose(
        cre.body.N[0], parent_n - expected_child_n - expected_loss, atol=1e-6
    )
    assert jnp.allclose(cre.body.B[0], parent_b - tissue, atol=1e-7)

    # The control arm still fails with no_tissue when B is too small.
    fail_state = base_state(cfg)
    fail_state = add_parent(
        fail_state, cfg, 0, pos=(10.0, 10.0), uid=5,
        reserves=(0.5, 0.5, 0.5), tissue=(0.001, 0.001, 0.0),
    )
    _, fail_buffer = reproduction.apply(fail_state, cfg)
    fails = of_kind(fail_buffer, "birth_fail")
    assert len(fails) == 1
    assert fails[0]["reason"] == events.reason_code("no_tissue")
    assert fails[0]["nutrient"] == C.NO_REF  # control arm has no reserve diagnostic


def _no_reserve_case(state, cfg, reserves, tissue=(0.15, 0.35, 0.0), uid=31):
    state = add_parent(
        state, cfg, 0, pos=(10.0, 10.0), uid=uid,
        reserves=reserves, tissue=tissue,
    )
    out, buffer = reproduction.apply(state, cfg)
    fails = of_kind(buffer, "birth_fail")
    assert len(fails) == 1
    assert fails[0]["reason"] == events.reason_code("no_reserve")
    return fails[0]


def test_no_reserve_reports_the_blocked_nutrient():
    """birth_fail(no_reserve).nutrient names P/A with the smallest N_j/B_child_j."""
    cfg = make_cfg(8)
    # Only plant reserves are below B_child_P = 0.009.
    only_plant = _no_reserve_case(base_state(cfg), cfg, (0.001, 0.5, 0.5))
    assert only_plant["nutrient"] == C.PLANT
    # Only animal reserves are below B_child_A = 0.021.
    only_animal = _no_reserve_case(base_state(cfg), cfg, (0.5, 0.001, 0.5), uid=32)
    assert only_animal["nutrient"] == C.ANIMAL

    # Toxin never participates in B_child: N_T = 0 does not block the birth.
    state = base_state(cfg)
    state = add_parent(
        state, cfg, 0, pos=(10.0, 10.0), uid=33,
        reserves=(0.5, 0.5, 0.0), tissue=(0.15, 0.35, 0.0),
    )
    out, buffer = reproduction.apply(state, cfg)
    births = of_kind(buffer, "birth")
    assert len(births) == 1
    assert births[0]["nutrient"] == -1
    assert bool(out.entities.creature.alive[1])
