"""Conservation accounting (SANDBOX.md §3.5, §13 test_accounting).

The synthetic world is one bush and one creature: a bite crosses the nutrient
book boundary, then the body system's digestion / absorption / demand / repair
flows leave it.  The books must close per nutrient type with relative error
below 1e-5, and the berry-mass book closes against regeneration.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.config import Capacities, Config, MotionConfig
from life_sandbox.core import accounting, components as C, events
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import interaction, metabolism

N = 4
FLESH = jnp.asarray([0.3, 0.7, 0.0], jnp.float32)


def small_cfg() -> Config:
    return Config.default().replaced(
        capacities=Capacities(creature=N, bush=6, carcass=2, neutral=1),
        motion=MotionConfig(noise_scale=0.0),
    )


def rho_genome(cfg: Config, rho=(0.2, 0.5, 0.3)) -> jnp.ndarray:
    genome = jnp.zeros((N, cfg.brain.genome_dim), jnp.float32)
    logits = jnp.log(jnp.asarray(rho, jnp.float32))
    return genome.at[:, 0:3].set(jnp.broadcast_to(logits, (N, 3)))


def scenario_state(cfg: Config, *, stock=0.5):
    state = empty(cfg)
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
            N=cre.body.N.at[0].set(jnp.asarray([0.2, 0.2, 0.2], jnp.float32)),
            B=cre.body.B.at[0].set(FLESH * 0.5),
            I=cre.body.I.at[0].set(1.0),
            mu=cre.body.mu.at[0].set(1.0),
        ),
        genome=C.Genome(g=rho_genome(cfg)),
        motor=update(cre.motor, last_out=last),
    )
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
            stock=bush.edible.stock.at[0].set(stock),
            kind=bush.edible.kind.at[0].set(0),
        ),
    )
    return update(state, entities=update(state.entities, creature=cre, bush=bush))


def test_nutrient_stock_counts_living_pools_only(tiny_cfg=None):
    cfg = small_cfg()
    state = scenario_state(cfg)
    expected = np.asarray(FLESH * 0.5 + jnp.asarray([0.2, 0.2, 0.2]), dtype=np.float64)
    assert np.allclose(accounting.nutrient_stock(state), expected)
    stock = accounting.nutrient_stock(state)
    cre = state.entities.creature
    cre = update(
        cre,
        alive=cre.alive.at[0].set(False),
        body=update(cre.body, N=cre.body.N.at[0].set(10.0)),
    )
    assert np.allclose(
        accounting.nutrient_stock(update(state, entities=update(state.entities, creature=cre))),
        np.zeros(3),
    )
    assert stock.sum() > 0.0


def test_check_books_reports_relative_error():
    inflow = np.asarray([1.0, 2.0, 3.0])
    outflow = np.asarray([0.5, 1.0, 1.5])
    delta = inflow - outflow
    relative, closed = accounting.check_books(inflow, outflow, delta)
    assert closed and np.all(relative < 1e-5)
    _, broken = accounting.check_books(inflow, outflow, delta + 1e-3)
    assert not broken


def test_berry_mass_book_closes():
    """regeneration = bitten mass + delta stock on the bush (§3.5)."""
    cfg = small_cfg()
    start = scenario_state(cfg, stock=0.4)
    regen = accounting.berry_regeneration(start)
    # One environment tick regenerates first, then the creature bites.
    bush = start.entities.bush
    bush = update(bush, edible=update(bush.edible, stock=bush.edible.stock + regen))
    regenerated = update(start, entities=update(start.entities, bush=bush))
    end, buffer = interaction.apply(regenerated, cfg)
    report = accounting.berry_books(start, end, buffer)
    assert report.closed
    assert np.all(report.relative_error < 1e-5)
    assert float(report.outflow[0]) == pytest.approx(0.05, abs=1e-6)
    assert float(report.inflow[0]) == pytest.approx(float(regen), abs=1e-7)


def test_nutrient_books_close_with_explicit_ledger():
    """A bush bite crosses the boundary; body flows leave it; books close."""
    cfg = small_cfg()
    start = scenario_state(cfg, stock=0.5)
    mid, bite_events = interaction.apply(start, cfg)
    end, body_events = metabolism.apply(mid, cfg)
    buffer = events.concat(bite_events, body_events)

    rho = np.asarray([0.2, 0.5, 0.3])
    eps = 0.9 * np.sqrt(rho)
    body_mid = mid.entities.creature.body
    stomach = np.asarray(body_mid.G[0], dtype=np.float64)
    reserves_mid = np.asarray(body_mid.N[0], dtype=np.float64)
    delta = metabolism.R_DIG * stomach
    absorbed = eps * delta
    reserves = reserves_mid + absorbed
    capacity = rho
    overflow = np.maximum(reserves - capacity, 0.0)
    # demand: base + bite surcharge (u_bite = 1) + repair (sum reserves > 0.5)
    demand = (
        metabolism.BASE_METABOLISM
        + 0.002 * 1.0
        + metabolism.REPAIR_RATE
    )
    ledger = accounting.Ledger()
    inflow, _ = accounting.event_bite_flow(bite_events)
    ledger.add_inflow("bush_bite", inflow)
    ledger.add_outflow("excretion", (1.0 - eps) * delta)
    ledger.add_outflow("overflow", overflow)
    ledger.add_outflow("metabolism", rho * demand)
    report = ledger.close(start, end)
    assert report.closed, report
    assert np.all(report.relative_error < 1e-5)
    assert np.allclose(ledger.inflow, inflow)

    # The engine's own residual book agrees with the explicit ledger.
    residual = accounting.nutrient_books(start, end, buffer)
    assert residual.closed
    assert np.allclose(residual.outflow, ledger.outflow, atol=1e-6)


def test_unbalanced_ledger_is_detected():
    cfg = small_cfg()
    start = scenario_state(cfg)
    end = start
    ledger = accounting.Ledger()
    ledger.add_outflow("demand", np.asarray([0.001, 0.001, 0.001]))
    report = ledger.close(start, end)
    assert not report.closed
    assert np.any(report.relative_error > 1e-5)
