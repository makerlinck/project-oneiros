"""Conservation accounting for the three nutrient types (SANDBOX.md §3.5).

The nutrient book tracks all nutrient mass inside the living world:

    stock_j = sum_alive_creatures (G_j + N_j + B_j) + sum_alive_carcasses n_j

External inflow is measured from the event buffer: every bite whose target is a
bush takes `dn` from the berry mass outside the nutrient book and adds it to the
biter's stomach.  Internal transfers (digestion G->N, absorption N, growth
N->B, bites on carcasses and living creatures, birth payments) cancel exactly
because they only move mass between the tracked pools.

Outflow categories of §3.5 (metabolic demand, unabsorbed plant/animal matter,
reserve overflow, unabsorbed toxin leaving through X, birth loss, death loss,
carcass decomposition) are *not* individually observable from the state and
the event buffer in M0: the event schema has no metabolic event and WorldState
carries no flow counters.  `nutrient_books` therefore reports the measured
residual outflow `inflow - delta_stock` (mass conservation) and allows the
caller to pass an independently computed `outflow` (for example the sum of
system-reported flows in a Ledger) to be checked against it.  The berry-mass
book is fully checkable without extra reporting:

    regeneration = bitten_mass + delta_stock     (§3.5, "berry mass book")

This module is host-side (numpy, float64) and observer-like: it never enters
the pure-function state path, never consumes randomness and never writes state,
so opening accounting cannot change a run (PRINCIPLES.md §2.3).

Closed-book tolerance: relative error < 1e-5 per type (§3.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from life_sandbox.core import components as C
from life_sandbox.core import events

RTOL: float = 1e-5
BERRY_REGEN_RATE: float = 0.03
BERRY_REGEN_BASE: float = 1e-4


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


def berry_regeneration(state) -> float:
    """Berry regeneration produced by one environment tick from this state (§4.2)."""
    ents = state.entities
    stock = _array(ents.bush.edible.stock)
    alive = _array(ents.bush.alive)
    regen = BERRY_REGEN_RATE * stock * (1.0 - stock) + BERRY_REGEN_BASE
    return float((regen * alive).sum())


def event_bite_flow(buffer: events.EventBuffer) -> tuple[np.ndarray, float]:
    """Nutrient inflow and bitten berry mass from bush bites in the buffer."""
    count = int(buffer.count)
    if count == 0:
        return np.zeros(C.N_NUTRIENTS, dtype=np.float64), 0.0
    kind = _array(buffer.kind[:count])
    target = _array(buffer.target_kind[:count])
    dn = _array(buffer.dn[:count])
    mask = (kind == events.kind_code("bite")) & (target == C.BUSH)
    taken = dn[mask]
    return taken.sum(axis=0), float(taken.sum())


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


@dataclass(frozen=True)
class Books:
    """A closed-book report for one segment and one book (nutrient or berry)."""

    inflow: np.ndarray
    outflow: np.ndarray
    delta: np.ndarray
    relative_error: np.ndarray
    closed: bool
    dropped_events: int = 0
    note: str = ""


def books(
    inflow: np.ndarray,
    outflow: np.ndarray,
    start_stock,
    end_stock,
    *,
    rtol: float = RTOL,
    dropped_events: int = 0,
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
        note=note,
    )


def nutrient_books(
    start,
    end,
    buffer: events.EventBuffer,
    *,
    outflow: np.ndarray | None = None,
    rtol: float = RTOL,
) -> Books:
    """Nutrient books for a segment.

    Inflow is measured from bush-bite events.  When `outflow` is not supplied
    it is the conservation residual `inflow - delta_stock` (the book closes by
    construction); pass independently computed outflows to test them against
    the residual.
    """
    inflow, _ = event_bite_flow(buffer)
    start_stock = nutrient_stock(start)
    end_stock = nutrient_stock(end)
    delta = end_stock - start_stock
    if outflow is None:
        outflow = inflow - delta
        note = "outflow is the conservation residual"
    else:
        note = "outflow supplied by the caller"
    return books(
        inflow,
        outflow,
        start_stock,
        end_stock,
        rtol=rtol,
        dropped_events=int(buffer.dropped),
        note=note,
    )


def berry_books(
    start,
    end,
    buffer: events.EventBuffer,
    *,
    regeneration: float | None = None,
    rtol: float = RTOL,
) -> Books:
    """Berry-mass book: regeneration = bitten mass + delta stock (§3.5)."""
    if regeneration is None:
        regeneration = berry_regeneration(start)
    _, bitten = event_bite_flow(buffer)
    inflow = np.asarray([regeneration], dtype=np.float64)
    outflow = np.asarray([bitten], dtype=np.float64)
    return books(
        inflow,
        outflow,
        np.asarray([berry_stock(start)], dtype=np.float64),
        np.asarray([berry_stock(end)], dtype=np.float64),
        rtol=rtol,
        dropped_events=int(buffer.dropped),
        note="berry mass = regeneration; outflow = bitten mass",
    )


@dataclass
class Ledger:
    """Explicit per-category flow accumulator for experiments and integration.

    Categories are keyed by name, only for the report; the totals are what the
    book check compares.  Use it to attribute outflows that the engine does not
    report as events (metabolic demand, excretion, overflow, birth loss, ...).
    """

    inflow: np.ndarray = field(
        default_factory=lambda: np.zeros(C.N_NUTRIENTS, dtype=np.float64)
    )
    outflow: np.ndarray = field(
        default_factory=lambda: np.zeros(C.N_NUTRIENTS, dtype=np.float64)
    )
    inflow_categories: dict[str, np.ndarray] = field(default_factory=dict)
    outflow_categories: dict[str, np.ndarray] = field(default_factory=dict)

    def add_inflow(self, name: str, amount) -> None:
        amount = _array(amount)
        self.inflow += amount
        self.inflow_categories[name] = self.inflow_categories.get(
            name, np.zeros(C.N_NUTRIENTS, dtype=np.float64)
        ) + amount

    def add_outflow(self, name: str, amount) -> None:
        amount = _array(amount)
        self.outflow += amount
        self.outflow_categories[name] = self.outflow_categories.get(
            name, np.zeros(C.N_NUTRIENTS, dtype=np.float64)
        ) + amount

    def close(self, start, end, *, rtol: float = RTOL) -> Books:
        delta = nutrient_stock(end) - nutrient_stock(start)
        relative, closed = check_books(self.inflow, self.outflow, delta, rtol)
        return Books(
            inflow=self.inflow.copy(),
            outflow=self.outflow.copy(),
            delta=delta,
            relative_error=relative,
            closed=closed,
            note="explicit ledger",
        )
