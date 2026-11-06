"""Pure helpers for the scripted calibration probe (SANDBOX.md §9).

The probe is a calibration tool only: it lives under analysis/, never appears in
an experiment arm, and reads true body state.  These helpers exist so the
species-selection rules can be unit tested without running the engine.
"""

from __future__ import annotations

import jax.numpy as jnp

from life_sandbox.core import components as C

STOCK_MIN: float = 0.05
TOXIC_SHARE: float = 0.05


def animal_share_per_slot(comp, kind):
    """Animal content of every bush slot: comp is (species, nutrient)."""
    return comp[:, C.ANIMAL][kind]


def toxic_share_per_slot(comp, kind):
    return comp[:, C.TOXIN][kind]


def coverage_scores(rho, comp):
    """Per-species coverage: min over P/A of c_{s,j} / sqrt(rho_j)."""
    ratios = comp[None, :, :2] / jnp.sqrt(jnp.maximum(rho[:, None, :2], 1e-12))
    return ratios.min(axis=-1)


def _rank_row(ranked, rank, kind, alive, stock, distance):
    species = ranked[:, rank]
    eligible = alive[None, :] & (kind[None, :] == species[:, None])
    stocked = eligible & (stock[None, :] >= STOCK_MIN)
    row = jnp.where(stocked, distance, jnp.inf)
    row = jnp.where(
        jnp.any(jnp.isfinite(row), axis=1, keepdims=True),
        row,
        jnp.where(eligible, distance, jnp.inf),
    )
    row = jnp.where(
        jnp.any(jnp.isfinite(row), axis=1, keepdims=True),
        row,
        jnp.where(alive[None, :], distance, jnp.inf),
    )
    return row, species


def select_bush_slot(
    rho,
    comp,
    kind,
    alive,
    stock,
    distance,
    body_toxin,
    stationary: bool = False,
):
    """Return the bush slot each creature targets (coverage rule, v3)."""
    if stationary:
        return jnp.argmin(jnp.where(alive[None, :], distance, jnp.inf), axis=1)
    scores = coverage_scores(rho, comp)
    ranked = jnp.argsort(-scores, axis=1)
    top, _species = _rank_row(ranked, 0, kind, alive, stock, distance)
    second, species_second = _rank_row(ranked, 1, kind, alive, stock, distance)
    third, species_third = _rank_row(ranked, 2, kind, alive, stock, distance)

    toxic_per_species = comp[:, C.TOXIN]
    own_species = ranked[:, 0]
    needs_toxin = body_toxin >= 0.5 * rho[:, C.TOXIN]
    switch = needs_toxin & (toxic_per_species[own_species] > TOXIC_SHARE)

    second_ok = toxic_per_species[species_second] <= TOXIC_SHARE
    third_ok = toxic_per_species[species_third] <= TOXIC_SHARE
    alternative = jnp.where(second_ok[:, None], second, jnp.where(third_ok[:, None], third, top))
    row = jnp.where(switch[:, None], alternative, top)
    return jnp.argmin(row, axis=1)
