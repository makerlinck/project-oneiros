"""Ledger write helpers (SANDBOX.md §3.5).

Systems accumulate flows where they happen; these helpers keep the immutable
update boilerplate in one place.  Every helper is a pure function of the ledger
and returns a new ledger.
"""

from __future__ import annotations

import jax.numpy as jnp

from life_sandbox.core.components import (
    Ledger,
    N_INFLOW_CATEGORIES,
    N_OUTFLOW_CATEGORIES,
)
from life_sandbox.core.struct import update


def add_inflow(ledger: Ledger, category: int, amount) -> Ledger:
    """Add a per-nutrient inflow vector (float32[3]) to one category."""
    values = jnp.asarray(amount, dtype=jnp.float32)
    return update(
        ledger,
        inflow=ledger.inflow.at[category].add(values).reshape(N_INFLOW_CATEGORIES, -1),
    )


def add_outflow(ledger: Ledger, category: int, amount) -> Ledger:
    """Add a per-nutrient outflow vector (float32[3]) to one category."""
    values = jnp.asarray(amount, dtype=jnp.float32)
    return update(
        ledger,
        outflow=ledger.outflow.at[category].add(values).reshape(N_OUTFLOW_CATEGORIES, -1),
    )


def add_transfer(ledger: Ledger, from_amount, to_amount) -> Ledger:
    """Record an internal conversion (e.g. carcass rot: plant/animal -> toxin)."""
    return update(
        ledger,
        transfer_from=ledger.transfer_from + jnp.asarray(from_amount, dtype=jnp.float32),
        transfer_to=ledger.transfer_to + jnp.asarray(to_amount, dtype=jnp.float32),
    )


def add_digested(ledger: Ledger, amount) -> Ledger:
    """Record G -> N digestion (diagnostic only: same nutrient types)."""
    return update(ledger, digested=ledger.digested + jnp.asarray(amount, dtype=jnp.float32))
