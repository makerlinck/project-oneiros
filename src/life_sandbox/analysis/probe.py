"""Pure helpers for the scripted calibration probe (SANDBOX.md §9).

The probe is a calibration tool only: it lives under analysis/, never appears in
an experiment arm, and reads true body state.  These helpers exist so the
species-selection rules can be unit tested without running the engine.
"""

from __future__ import annotations

import jax.numpy as jnp

from life_sandbox.core import components as C

DIGEST_RATE: float = 0.0135
ABSORPTION: float = 0.9
ANIMAL_MARGIN: float = 1.3
STOCK_MIN: float = 0.3
TOXIC_SHARE: float = 0.05


def animal_share_per_slot(comp, kind):
    """Animal content of every bush slot: comp is (species, nutrient)."""
    return comp[:, C.ANIMAL][kind]


def toxic_share_per_slot(comp, kind):
    return comp[:, C.TOXIN][kind]


def eligible_species_mask(rho, comp, kind, alive, c_tot):
    """A bush slot is eligible if its animal supply covers 1.3x demand."""
    rho_animal = rho[:, C.ANIMAL]
    share = animal_share_per_slot(comp, kind)
    supplied = DIGEST_RATE * ABSORPTION * jnp.sqrt(rho_animal)[:, None] * share[None, :]
    return alive[None, :] & (supplied >= ANIMAL_MARGIN * rho_animal[:, None] * c_tot)


def _pick(candidates, distance, stock):
    base = jnp.where(candidates, distance, jnp.inf)
    stocked = jnp.where(candidates & (stock[None, :] >= STOCK_MIN), distance, jnp.inf)
    picked = jnp.where(jnp.any(jnp.isfinite(stocked), axis=1, keepdims=True), stocked, base)
    return jnp.where(
        jnp.any(jnp.isfinite(picked), axis=1, keepdims=True), picked, distance
    )


def select_bush_slot(
    rho,
    comp,
    kind,
    alive,
    stock,
    distance,
    c_tot,
    body_toxin,
    stationary: bool = False,
):
    """Return the bush slot each creature targets (v2 rules, fixed indices)."""
    if stationary:
        return jnp.argmin(jnp.where(alive[None, :], distance, jnp.inf), axis=1)
    eligible = eligible_species_mask(rho, comp, kind, alive, c_tot)
    pick = _pick(eligible, distance, stock)
    slot = jnp.argmin(pick, axis=1)
    toxic = toxic_share_per_slot(comp, kind)
    own_toxic = toxic[slot] > TOXIC_SHARE
    needs_toxin = body_toxin >= 0.5 * rho[:, C.TOXIN]
    switch = own_toxic & needs_toxin
    non_toxic = eligible & (toxic[None, :] <= TOXIC_SHARE)
    alternative = _pick(non_toxic, distance, stock)
    pick = jnp.where(switch[:, None], alternative, pick)
    return jnp.argmin(pick, axis=1)
