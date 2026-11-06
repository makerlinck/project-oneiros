"""Conservation accounting from the per-tick ledger (SANDBOX.md §3.5, §13).

The ledger is written where flows happen (interaction: bush-bite inflow; body:
metabolism/excretion/overflow/toxin load; reproduce: birth loss; die: death
loss, decomposition and reseed inflow; environment: rot transfers and carcass
decomposition), so the closed-book check is no longer a tautology:

    inflow - outflow + transfer_to - transfer_from = delta_stock   (per type)
    sum(transfer_from) = sum(transfer_to)                          (conversions)

Long runs use the production path (`run(compile=True)`); the eager path is
checked against it on a short segment because eager execution is far slower
per tick but must produce the same float64 ledger totals.
"""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.config import Capacities, Config, MotionConfig, RunConfig, WorldConfig
from life_sandbox.core import accounting, components as C, events
from life_sandbox.core.state import empty
from life_sandbox.core.step import run
from life_sandbox.core.struct import update
from life_sandbox.core.systems import death, environment, interaction, metabolism, reproduction
from life_sandbox.world.init import initial_state

N = 4
FLESH = jnp.asarray([0.3, 0.7, 0.0], jnp.float32)


def small_cfg(*, founders=2, seed=3) -> Config:
    return Config.default().replaced(
        capacities=Capacities(creature=N, bush=3, carcass=1, neutral=1),
        world=WorldConfig(founders=founders),
        run=RunConfig(seed=seed, ticks=5),
        motion=MotionConfig(noise_scale=0.0),
    )


def fake_result(*, inflow=None, outflow=None, transfer_from=None, transfer_to=None):
    """RunResult-shaped object for mini segments assembled by hand."""
    zero_in = np.zeros((C.N_INFLOW_CATEGORIES, C.N_NUTRIENTS), dtype=np.float64)
    zero_out = np.zeros((C.N_OUTFLOW_CATEGORIES, C.N_NUTRIENTS), dtype=np.float64)
    zero = np.zeros(C.N_NUTRIENTS, dtype=np.float64)
    return types.SimpleNamespace(
        inflow=zero_in if inflow is None else np.asarray(inflow, dtype=np.float64),
        outflow=zero_out if outflow is None else np.asarray(outflow, dtype=np.float64),
        transfer_from=zero if transfer_from is None else np.asarray(transfer_from, np.float64),
        transfer_to=zero if transfer_to is None else np.asarray(transfer_to, np.float64),
        dropped_events=0,
    )


def result_from_ledger(ledger, *, dropped=0):
    return types.SimpleNamespace(
        inflow=np.asarray(ledger.inflow, dtype=np.float64),
        outflow=np.asarray(ledger.outflow, dtype=np.float64),
        transfer_from=np.asarray(ledger.transfer_from, dtype=np.float64),
        transfer_to=np.asarray(ledger.transfer_to, dtype=np.float64),
        dropped_events=int(dropped),
    )


def stock_state(*, reserves=(0.5, 0.0, 0.0), mass=1.0):
    """One live creature with the given N (G and B zero) in a tiny world."""
    cfg = small_cfg()
    state = empty(cfg)
    cre = state.entities.creature
    cre = update(
        cre,
        alive=cre.alive.at[0].set(True),
        body=update(cre.body, N=cre.body.N.at[0].set(jnp.asarray(reserves, jnp.float32))),
        shape=update(cre.shape, mass=cre.shape.mass.at[0].set(mass)),
    )
    return update(state, entities=update(state.entities, creature=cre))


def test_nutrient_stock_counts_living_pools_only():
    cfg = small_cfg()
    state = stock_state(reserves=(0.2, 0.3, 0.1))
    carcass = state.entities.carcass
    carcass = update(
        carcass,
        alive=carcass.alive.at[0].set(True),
        edible=update(
            carcass.edible, n=carcass.edible.n.at[0].set(jnp.asarray([0.1, 0.2, 0.0], jnp.float32))
        ),
    )
    state = update(state, entities=update(state.entities, carcass=carcass))
    expected = np.asarray([0.3, 0.5, 0.1], dtype=np.float64)
    assert np.allclose(accounting.nutrient_stock(state), expected)
    assert accounting.berry_stock(state) == 0.0

    cre = state.entities.creature
    cre = update(cre, alive=cre.alive.at[0].set(False))
    carcass = update(
        state.entities.carcass, alive=state.entities.carcass.alive.at[0].set(False)
    )
    state = update(state, entities=update(state.entities, creature=cre, carcass=carcass))
    assert np.allclose(accounting.nutrient_stock(state), 0.0)


def test_check_books_reports_relative_error():
    inflow = np.asarray([1.0, 2.0, 3.0])
    outflow = np.asarray([0.5, 1.0, 1.5])
    delta = inflow - outflow
    relative, closed = accounting.check_books(inflow, outflow, delta)
    assert closed and np.all(relative < accounting.RTOL)
    _, broken = accounting.check_books(inflow, outflow, delta + 1e-3)
    assert not broken


def test_transfer_terms_enter_the_balance():
    """Internal conversions appear on both sides; they must conserve mass."""
    start = stock_state(reserves=(0.5, 0.0, 0.0))
    end_state = stock_state(reserves=(0.4, 0.0, 0.1))
    # Plant 0.1 internally becomes toxin 0.1.
    result = fake_result(
        inflow=np.zeros((C.N_INFLOW_CATEGORIES, C.N_NUTRIENTS)),
        outflow=np.zeros((C.N_OUTFLOW_CATEGORIES, C.N_NUTRIENTS)),
        transfer_from=np.asarray([0.1, 0.0, 0.0]),
        transfer_to=np.asarray([0.0, 0.0, 0.1]),
    )
    books = accounting.nutrient_books(start, end_state, result)
    assert books.closed
    assert books.transfers_closed
    assert np.all(books.relative_error < accounting.RTOL)
    assert np.allclose(books.delta, np.asarray([-0.1, 0.0, 0.1]))

    broken = fake_result(transfer_from=np.asarray([0.1, 0.0, 0.0]))
    _, transfers_closed = accounting.check_transfers(broken)
    assert not transfers_closed


def test_nutrient_books_close_on_small_world():
    """500 production ticks: external flows and internal transfers both close."""
    cfg = small_cfg()
    start = initial_state(cfg)
    result = run(start, cfg, 500, compile=True)
    books = accounting.nutrient_books(start, result.state, result)
    assert books.closed, books
    assert np.all(books.relative_error < accounting.RTOL)
    assert books.transfers_closed
    # Founders = 2 < 8 -> the world reseeds at least once (external inflow).
    assert result.inflow[C.INFLOW_RESEED].sum() > 0.0
    # Metabolic demand is the backbone outflow; other categories may be zero
    # depending on what the creatures found in the world (bites, carcass rot).
    assert result.outflow[C.OUTFLOW_METABOLISM].sum() > 0.0
    assert np.all(result.outflow >= 0.0)
    assert int(result.state.entities.creature.alive.sum()) > 0


def test_compiled_and_eager_ledgers_agree():
    """The production scan and the eager reference report the same flows."""
    cfg = small_cfg()
    start = initial_state(cfg)
    eager = run(start, cfg, 3, compile=False)
    compiled = run(start, cfg, 3, compile=True)
    for name in ("inflow", "outflow", "transfer_from", "transfer_to", "digested"):
        a = getattr(eager, name)
        b = getattr(compiled, name)
        assert a.shape == b.shape
        # Same tolerance as tests/test_events.py for eager<=>compiled ledger
        # totals (float32 fusion differences are ULP-level).
        assert np.allclose(a, b, rtol=1e-6, atol=1e-9), name
    assert eager.counts == compiled.counts
    assert eager.dropped_events == compiled.dropped_events


def _bite_world(cfg: Config):
    """Bush at (11,10) and a creature at (10,10) whose output says 'bite'."""
    state = empty(cfg)
    bush = state.entities.bush
    bush = update(
        bush,
        alive=bush.alive.at[0].set(True),
        uid=bush.uid.at[0].set(100),
        transform=update(
            bush.transform,
            pos=bush.transform.pos.at[0].set(jnp.asarray([11.0, 10.0], jnp.float32)),
        ),
        shape=update(
            bush.shape,
            radius=bush.shape.radius.at[0].set(1.0),
            mass=bush.shape.mass.at[0].set(jnp.inf),
        ),
        edible=update(
            bush.edible,
            stock=bush.edible.stock.at[0].set(0.4),
            kind=bush.edible.kind.at[0].set(0),
        ),
    )
    cre = state.entities.creature
    last = jnp.zeros((N, C.N_OUTPUTS), jnp.float32).at[0, C.OUT_BITE].set(1.0)
    cre = update(
        cre,
        alive=cre.alive.at[0].set(True),
        identity=update(
            cre.identity,
            uid=cre.identity.uid.at[0].set(1),
            lineage_id=cre.identity.lineage_id.at[0].set(1),
        ),
        transform=update(
            cre.transform,
            pos=cre.transform.pos.at[0].set(jnp.asarray([10.0, 10.0], jnp.float32)),
        ),
        shape=update(
            cre.shape,
            radius=cre.shape.radius.at[0].set(0.5),
            mass=cre.shape.mass.at[0].set(1.0),
        ),
        body=update(
            cre.body,
            B=cre.body.B.at[0].set(FLESH * 0.5),
            N=cre.body.N.at[0].set(jnp.asarray([0.2, 0.2, 0.2])),
            I=cre.body.I.at[0].set(1.0),
            mu=cre.body.mu.at[0].set(1.0),
        ),
        motor=update(cre.motor, last_out=last),
    )
    return update(state, entities=update(state.entities, creature=cre, bush=bush))


def test_bush_bite_ledger_and_berry_book():
    """One environment tick + one bite: both the nutrient and berry books close."""
    cfg = small_cfg()
    start = _bite_world(cfg)
    regenerated, _ = environment.apply(start, cfg)
    end, _ = interaction.apply(regenerated, cfg)

    result = result_from_ledger(end.ledger)
    nutrient_books = accounting.nutrient_books(start, end, result)
    assert nutrient_books.closed, nutrient_books
    assert nutrient_books.inflow[C.INFLOW_BITES].sum() > 0.0

    regen = accounting.berry_regeneration(start, cfg)
    berry = accounting.berry_books(start, end, result, cfg)  # default regen from cfg
    assert berry.closed, berry
    assert float(berry.outflow[0]) == pytest.approx(0.05, abs=1e-6)
    assert float(berry.inflow[0]) == pytest.approx(float(regen), abs=1e-7)


def test_berry_regeneration_reads_config_and_mirrors_environment():
    """The berry mirror uses cfg.world.bush_regen_* and the startup ramp (§4.2, §9)."""
    cfg = small_cfg()
    start = _bite_world(cfg)  # one bush with stock 0.4
    env_out, _ = environment.apply(start, cfg)
    added = accounting.berry_stock(env_out) - accounting.berry_stock(start)
    assert accounting.berry_regeneration(start, cfg) == pytest.approx(added, abs=1e-7)

    # M1 config values drive the mirror instead of module constants.
    scaled = cfg.replaced(
        world=WorldConfig(founders=2, bush_regen_rate=0.1, bush_regen_base=0.01)
    )
    expected = 0.1 * 0.4 * 0.6 + 0.01
    assert accounting.berry_regeneration(start, scaled) == pytest.approx(expected, abs=1e-7)

    # The startup ramp multiplies the increment before the [0, 1] clip.
    ramped = cfg.replaced(
        world=WorldConfig(
            founders=2,
            bush_regen_rate=0.1,
            bush_regen_base=0.01,
            ramp_ticks=10,
            ramp_berry_multiplier=2.0,
        )
    )
    assert accounting.berry_regeneration(start, ramped) == pytest.approx(
        2.0 * expected, abs=1e-7
    )


def _dead_creature(
    state,
    slot,
    uid,
    reserves,
    tissue,
    *,
    loss=None,
    nutrient: int = C.NO_REF,
    source_uid: int = C.NO_REF,
    same_lineage: bool = False,
    age: int = 0,
):
    cre = state.entities.creature
    loss = [0.0] * C.N_DAMAGE_SOURCES if loss is None else loss
    cre = update(
        cre,
        alive=cre.alive.at[slot].set(True),
        identity=update(cre.identity, uid=cre.identity.uid.at[slot].set(uid)),
        body=update(
            cre.body,
            N=cre.body.N.at[slot].set(jnp.asarray(reserves, jnp.float32)),
            B=cre.body.B.at[slot].set(jnp.asarray(tissue, jnp.float32)),
            I=cre.body.I.at[slot].set(0.0),
            age=cre.body.age.at[slot].set(age),
        ),
        damage=update(
            cre.damage,
            loss=cre.damage.loss.at[slot].set(jnp.asarray(loss, jnp.float32)),
            nutrient=cre.damage.nutrient.at[slot].set(nutrient),
            source_uid=cre.damage.source_uid.at[slot].set(source_uid),
            same_lineage=cre.damage.same_lineage.at[slot].set(same_lineage),
        ),
        shape=update(cre.shape, mass=cre.shape.mass.at[slot].set(1.0)),
    )
    return update(state, entities=update(state.entities, creature=cre))


def _event_rows(buffer) -> list[dict]:
    rows = []
    for i in range(int(buffer.count)):
        rows.append(
            {
                "kind": int(buffer.kind[i]),
                "uid": int(buffer.uid[i]),
                "target_kind": int(buffer.target_kind[i]),
                "target_uid": int(buffer.target_uid[i]),
                "same_lineage": bool(buffer.same_lineage[i]),
                "cause": int(buffer.cause[i]),
                "nutrient": int(buffer.nutrient[i]),
                "loss": [float(v) for v in buffer.loss[i]],
                "success": bool(buffer.success[i]),
                "reason": int(buffer.reason[i]),
            }
        )
    return rows


def test_crafted_tick_closes_with_all_body_categories():
    """A bite + a full stomach exercises every body outflow category at once."""
    cfg = small_cfg()
    start = _bite_world(cfg)
    cre = start.entities.creature
    cre = update(
        cre,
        body=update(
            cre.body,
            # Fill the stomach part-way: a full stomach would leave no room for
            # the bite, and total q is bounded by the remaining stomach capacity.
            G=cre.body.G.at[0].set(jnp.asarray([0.05, 0.05, 0.05])),
            N=cre.body.N.at[0].set(jnp.asarray([0.199, 0.499, 0.299])),
            X=cre.body.X.at[0].set(0.2),
            I=cre.body.I.at[0].set(0.5),
        ),
    )
    start = update(start, entities=update(start.entities, creature=cre))
    mid, _ = interaction.apply(start, cfg)
    end, _ = metabolism.apply(mid, cfg)

    result = result_from_ledger(end.ledger)
    books = accounting.nutrient_books(start, end, result)
    assert books.closed, books
    assert books.transfers_closed
    assert np.all(books.relative_error < accounting.RTOL)
    inflow = np.asarray(end.ledger.inflow)
    outflow = np.asarray(end.ledger.outflow)
    assert inflow[C.INFLOW_BITES].sum() > 0.0                  # bush bite
    assert outflow[C.OUTFLOW_METABOLISM].sum() > 0.0          # demand + repair
    assert outflow[C.OUTFLOW_EXCRETION].sum() > 0.0           # unabsorbed P/A
    assert outflow[C.OUTFLOW_TOXIN_LOAD].sum() > 0.0          # unabsorbed T
    assert outflow[C.OUTFLOW_OVERFLOW].sum() > 0.0            # above N_cap
    assert np.asarray(end.ledger.digested).sum() > 0.0        # G -> N


def test_death_and_reseed_ledger():
    """Death loss/ decomposition are booked; the reseed injects B + N."""
    cfg = small_cfg()
    state = empty(cfg)
    # Two deaths, one carcass slot in this tiny world -> one decomposition.
    state = _dead_creature(state, 0, 11, (0.2, 0.2, 0.2), (0.03, 0.07, 0.0))
    state = _dead_creature(state, 1, 12, (0.2, 0.2, 0.2), (0.03, 0.07, 0.0))
    out, buffer = death.apply(state, cfg)

    expected_death_loss = 0.5 * (jnp.asarray([0.2, 0.2, 0.2]) * 2.0)
    assert jnp.allclose(out.ledger.outflow[C.OUTFLOW_DEATH_LOSS], expected_death_loss, atol=1e-6)
    # The death that found no carcass slot loses its whole edible n.
    nutrition = jnp.asarray([0.03, 0.07, 0.0]) + 0.5 * jnp.asarray([0.2, 0.2, 0.2])
    assert jnp.allclose(out.ledger.outflow[C.OUTFLOW_DECOMPOSITION], nutrition, atol=1e-6)

    # Reseed inflow equals the B + N actually written into the new bodies.
    seeded = out.entities.creature
    injected = jnp.sum((seeded.body.N + seeded.body.B) * seeded.alive[:, None], axis=0)
    assert jnp.allclose(out.ledger.inflow[C.INFLOW_RESEED], injected, atol=1e-6)
    assert jnp.sum(out.ledger.inflow[C.INFLOW_RESEED]) > 0.0

    # The ledger closes the death+reseed tick as well.
    books = accounting.nutrient_books(state, out, result_from_ledger(out.ledger))
    assert books.closed, books


def test_death_event_carries_loss_and_nutrient():
    """The death event reports the decayed loss vector and the missing nutrient."""
    cfg = small_cfg()
    state = empty(cfg)
    state = _dead_creature(
        state, 0, 11, (0.2, 0.2, 0.2), (0.03, 0.07, 0.0),
        loss=(0.7, 0.0, 0.0, 0.0, 0.0), nutrient=C.PLANT,
    )
    state = _dead_creature(
        state, 1, 12, (0.2, 0.2, 0.2), (0.03, 0.07, 0.0),
        loss=(0.0, 0.9, 0.0, 0.0, 0.0),
    )
    _, buffer = death.apply(state, cfg)
    deaths = {
        row["uid"]: row
        for row in _event_rows(buffer)
        if row["kind"] == events.kind_code("death")
    }
    assert set(deaths) == {11, 12}
    first = deaths[11]
    assert first["cause"] == events.cause_code("nutrition")
    assert first["nutrient"] == C.PLANT
    assert first["loss"] == pytest.approx([0.7, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    second = deaths[12]
    assert second["cause"] == events.cause_code("toxin")
    assert second["nutrient"] == -1
    assert second["loss"] == pytest.approx([0.0, 0.9, 0.0, 0.0, 0.0], abs=1e-6)


def test_death_cause_is_the_largest_decayed_loss():
    """Every loss source maps to its death cause; zero loss falls back to other."""
    cfg = small_cfg()
    expected = {
        C.LOSS_NUTRITION: "nutrition",
        C.LOSS_TOXIN: "toxin",
        C.LOSS_HEAT: "heat",
        C.LOSS_COLD: "cold",
        C.LOSS_BITE: "bite",
    }
    for source, name in expected.items():
        state = empty(cfg)
        loss = [0.0] * C.N_DAMAGE_SOURCES
        loss[source] = 1.0
        state = _dead_creature(
            state, 0, 21, (0.2, 0.2, 0.2), (0.03, 0.07, 0.0),
            loss=loss,
            nutrient=C.ANIMAL,
            source_uid=7,
            same_lineage=True,
        )
        _, buffer = death.apply(state, cfg)
        events_rows = [
            row for row in _event_rows(buffer) if row["kind"] == events.kind_code("death")
        ]
        assert len(events_rows) == 1, name
        row = events_rows[0]
        assert row["cause"] == events.cause_code(name)
        assert row["loss"][source] == pytest.approx(1.0, abs=1e-6)
        if name == "bite":
            assert row["target_kind"] == C.CREATURE
            assert row["target_uid"] == 7
            assert row["same_lineage"]
        if name == "nutrition":
            assert row["nutrient"] == C.ANIMAL

    # All-zero loss -> other; the age cap wins over any loss.
    state = empty(cfg)
    state = _dead_creature(state, 0, 22, (0.2, 0.2, 0.2), (0.03, 0.07, 0.0))
    _, buffer = death.apply(state, cfg)
    row = next(r for r in _event_rows(buffer) if r["kind"] == events.kind_code("death"))
    assert row["cause"] == events.cause_code("other")

    state = empty(cfg)
    state = _dead_creature(
        state, 0, 23, (0.2, 0.2, 0.2), (0.03, 0.07, 0.0),
        loss=(0.0, 0.0, 6.0, 0.0, 0.0), age=2000,
    )
    _, buffer = death.apply(state, cfg)
    row = next(r for r in _event_rows(buffer) if r["kind"] == events.kind_code("death"))
    assert row["cause"] == events.cause_code("age")
    assert row["nutrient"] == -1


def test_reseed_records_birth_events():
    """Each reseeded individual also gets a birth event with reason=reseed."""
    cfg = small_cfg()
    state = empty(cfg)
    out, buffer = death.apply(state, cfg)
    rows = _event_rows(buffer)
    births = [row for row in rows if row["kind"] == events.kind_code("birth")]
    reseeds = [row for row in rows if row["kind"] == events.kind_code("reseed")]
    alive_uids = {
        int(uid) for uid in out.entities.creature.identity.uid[out.entities.creature.alive]
    }
    assert len(births) == len(reseeds) == len(alive_uids) > 0
    assert {row["uid"] for row in births} == alive_uids
    assert {row["uid"] for row in reseeds} == alive_uids
    for row in births:
        assert row["reason"] == events.reason_code("reseed")
        assert row["target_kind"] == C.NO_REF
        assert row["success"]


def test_birth_ledger():
    """0.02*rho_parent birth loss is booked; the child payment is internal."""
    cfg = small_cfg()
    state = empty(cfg)
    cre = state.entities.creature
    last = jnp.zeros((N, C.N_OUTPUTS), jnp.float32)
    cre = update(
        cre,
        alive=cre.alive.at[0].set(True),
        identity=update(
            cre.identity,
            uid=cre.identity.uid.at[0].set(1),
            lineage_id=cre.identity.lineage_id.at[0].set(1),
        ),
        transform=update(
            cre.transform,
            pos=cre.transform.pos.at[0].set(jnp.asarray([10.0, 10.0], jnp.float32)),
        ),
        shape=update(
            cre.shape,
            radius=cre.shape.radius.at[0].set(0.5),
            mass=cre.shape.mass.at[0].set(1.0),
        ),
        body=update(
            cre.body,
            B=cre.body.B.at[0].set(jnp.asarray([0.15, 0.35, 0.0])),
            N=cre.body.N.at[0].set(jnp.asarray([0.5, 0.5, 0.5])),
            I=cre.body.I.at[0].set(1.0),
            mu=cre.body.mu.at[0].set(1.0),
        ),
        motor=update(cre.motor, last_out=last),
    )
    state = update(state, entities=update(state.entities, creature=cre))
    out, buffer = reproduction.apply(state, cfg)
    assert int(buffer.count) == 1 and bool(buffer.success[0])
    parent_rho = jnp.ones(3) / 3.0  # genome zeros -> softmax = uniform diet
    expected_loss = 0.02 * parent_rho
    assert jnp.allclose(out.ledger.outflow[C.OUTFLOW_BIRTH_LOSS], expected_loss, atol=1e-6)
    assert float(jnp.sum(out.ledger.inflow)) == 0.0
    books = accounting.nutrient_books(state, out, result_from_ledger(out.ledger))
    assert books.closed, books
