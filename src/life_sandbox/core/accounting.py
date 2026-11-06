"""Conservation accounting for the three nutrient types (SANDBOX.md §3.5).

The nutrient book tracks all nutrient mass inside the living world:

    stock_j = sum_alive_creatures (G_j + N_j + B_j) + sum_alive_carcasses n_j

Systems write their flows into the per-tick `Ledger` draft area where they
happen (interaction, body, reproduce, die; environment writes rot transfers and
decomposition), and `step.run` returns the float64 sums over the segment.  This
module turns those ledger totals into a closed-book report:

    inflow_j - outflow_j + transfer_to_j - transfer_from_j = delta_stock_j

where the internal conversions that SANDBOX.md §3.5 asks to record separately
(carcass rot P/A -> T) appear as transfer terms, and the separate check
`sum(transfer_from) == sum(transfer_to)` verifies that conversions conserve
mass.  All comparisons use the relative-error rule of §3.5 (rtol < 1e-5).

The berry-mass book is separate from the nutrient book:

    regeneration = bitten_mass + delta_bush_stock

with `bitten_mass` taken from `inflow[INFLOW_BITES].sum()` (the mass crossed
from the berry book into the nutrient book).  For a one-tick segment the
regeneration can be derived from the start state and the M1 config values
(`cfg.world.bush_regen_rate`, `cfg.world.bush_regen_base`) plus the same startup
ramp multiplier as the environment; for longer segments the caller must pass
the summed regeneration explicitly (the engine does not keep a second
berry-mass ledger in M0).

This module is host-side (numpy, float64) and observer-like: it never enters the
pure-function state path, never consumes randomness and never writes state, so
opening accounting cannot change a run (PRINCIPLES.md §2.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from life_sandbox.core import components as C, ramp

RTOL: float = 1e-5


def _array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def nutrient_stock(state) -> np.ndarray:
    """Per-type nutrient mass in living creatures and carcasses (§3.5)."""
    ents = state.entities
    alive = _array(ents.creature.alive)
    creature = (
        _array(ents.creature.body.G) + _array(ents.creature.body.N) + _array(ents.creature.body.B)
    )
    total = (creature * alive[:, None]).sum(axis=0)
    carcass = _array(ents.carcass.edible.n)
    carcass_alive = _array(ents.carcass.alive)
    return total + (carcass * carcass_alive[:, None]).sum(axis=0)


def berry_stock(state) -> float:
    """Total berry mass on living bushes (the separate berry-mass book)."""
    ents = state.entities
    return float((_array(ents.bush.edible.stock) * _array(ents.bush.alive)).sum())


def berry_regeneration(state, cfg, params=None) -> float:
    """Berry regeneration one environment tick produces from this state (§4.2).

    Mirrors environment.berry_growth exactly: the logistic increment
    rate*stock*(1-stock) + base is scaled by ramp.berry_multiplier(tick, cfg),
    then clipped to the [0, 1] stock range (so per bush the added mass is
    min(increment, 1 - stock)).  The coefficients are the M1 config values
    cfg.world.bush_regen_rate / bush_regen_base, or the runtime \`params\` when a
    batched scan supplies them.
    """
    ents = state.entities
    stock = _array(ents.bush.edible.stock)
    alive = _array(ents.bush.alive)
    if params is None:
        rate = float(cfg.world.bush_regen_rate)
        base = float(cfg.world.bush_regen_base)
    else:
        rate = float(params.bush_regen_rate)
        base = float(params.bush_regen_base)
    multiplier = float(ramp.berry_multiplier(state.tick, cfg, params))
    nut = float(cfg.world.bush_regen_nut_multiplier)
    kinds = _array(ents.bush.edible.kind)
    factor = np.where(kinds == 2, nut, 1.0)
    growth = (rate * stock * (1.0 - stock) + base) * factor * multiplier
    added = np.minimum(growth, np.maximum(1.0 - stock, 0.0))
    return float((added * alive).sum())


def check_books(
    inflow: np.ndarray, outflow: np.ndarray, delta: np.ndarray, rtol: float = RTOL
) -> tuple[np.ndarray, bool]:
    """Check `inflow = outflow + delta_stock` per type; returns (rel_error, ok)."""
    inflow = _array(inflow)
    outflow = _array(outflow)
    delta = _array(delta)
    lhs = inflow
    rhs = outflow + delta
    scale = np.maximum(1.0, np.maximum(np.abs(lhs), np.abs(rhs)))
    relative = np.abs(lhs - rhs) / scale
    return relative, bool(np.all(relative < rtol))


def check_transfers(result, rtol: float = RTOL) -> tuple[np.ndarray, bool]:
    """Check that recorded internal conversions conserve mass (§3.5)."""
    from_amount = _array(result.transfer_from)
    to_amount = _array(result.transfer_to)
    lhs = np.sum(from_amount)
    rhs = np.sum(to_amount)
    scale = max(1.0, abs(float(lhs)), abs(float(rhs)))
    relative = np.asarray(abs(lhs - rhs) / scale)
    return relative, bool(relative < rtol)


@dataclass(frozen=True)
class Books:
    """A closed-book report for one segment and one book (nutrient or berry)."""

    inflow: np.ndarray
    outflow: np.ndarray
    delta: np.ndarray
    relative_error: np.ndarray
    closed: bool
    dropped_events: int = 0
    transfer_from: np.ndarray | None = None
    transfer_to: np.ndarray | None = None
    transfer_relative_error: np.ndarray | None = None
    transfers_closed: bool = True
    note: str = ""


def books(
    inflow: np.ndarray,
    outflow: np.ndarray,
    start_stock,
    end_stock,
    *,
    rtol: float = RTOL,
    dropped_events: int = 0,
    transfer_from: np.ndarray | None = None,
    transfer_to: np.ndarray | None = None,
    transfer_relative_error: np.ndarray | None = None,
    transfers_closed: bool = True,
    note: str = "",
) -> Books:
    """Assemble a Books report from explicit flows and endpoint stocks."""
    start = _array(start_stock)
    end = _array(end_stock)
    delta = end - start
    relative, closed = check_books(inflow, outflow, delta, rtol)
    return Books(
        inflow=_array(inflow),
        outflow=_array(outflow),
        delta=delta,
        relative_error=relative,
        closed=closed,
        dropped_events=int(dropped_events),
        transfer_from=transfer_from,
        transfer_to=transfer_to,
        transfer_relative_error=transfer_relative_error,
        transfers_closed=transfers_closed,
        note=note,
    )


def nutrient_books(start, end, result, *, rtol: float = RTOL) -> Books:
    """Nutrient books for a segment, from the run's float64 ledger totals.

    `result` is a `core.step.RunResult`.  Internal conversions recorded as
    transfers enter the balance on the receiving/sending side (§3.5), and
    `transfers_closed` checks that they conserve total mass.
    """
    inflow = _array(result.inflow).sum(axis=0)
    outflow = _array(result.outflow).sum(axis=0)
    transfer_from = _array(result.transfer_from)
    transfer_to = _array(result.transfer_to)
    start_stock = nutrient_stock(start)
    end_stock = nutrient_stock(end)
    delta = end_stock - start_stock
    # inflow + transfer_to = outflow + transfer_from + delta.
    relative, closed = check_books(
        inflow + transfer_to, outflow + transfer_from, delta, rtol
    )
    transfer_relative, transfers_closed = check_transfers(result, rtol)
    return Books(
        inflow=inflow,
        outflow=outflow,
        delta=delta,
        relative_error=relative,
        closed=closed,
        dropped_events=int(result.dropped_events),
        transfer_from=transfer_from,
        transfer_to=transfer_to,
        transfer_relative_error=transfer_relative,
        transfers_closed=transfers_closed,
        note="nutrient book from ledger totals; transfers recorded separately",
    )


def berry_books(
    start,
    end,
    result,
    cfg,
    *,
    regeneration: float | None = None,
    rtol: float = RTOL,
) -> Books:
    """Berry-mass book: regeneration = bitten mass + delta stock (§3.5).

    `bitten` comes from `result.inflow[INFLOW_BITES].sum()`.  For a one-tick
    segment `regeneration` defaults to the start state's regen (mirroring the
    environment with the same cfg); pass the summed regeneration for longer
    segments.
    """
    if regeneration is None:
        regeneration = berry_regeneration(start, cfg)
    bitten = float(np.sum(_array(result.inflow[C.INFLOW_BITES])))
    return books(
        np.asarray([regeneration], dtype=np.float64),
        np.asarray([bitten], dtype=np.float64),
        np.asarray([berry_stock(start)], dtype=np.float64),
        np.asarray([berry_stock(end)], dtype=np.float64),
        rtol=rtol,
        dropped_events=int(result.dropped_events),
        note="berry mass = regeneration; outflow = bitten mass from inflow[BITES]",
    )


def ledger_totals(result) -> dict[str, np.ndarray]:
    """Float64 totals of every ledger field (diagnostics for tests/reports)."""
    return {
        "inflow": _array(result.inflow),
        "outflow": _array(result.outflow),
        "transfer_from": _array(result.transfer_from),
        "transfer_to": _array(result.transfer_to),
        "digested": _array(result.digested),
    }
