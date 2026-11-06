"""Unit tests for the scripted probe's species selection (analysis/probe.py)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from life_sandbox.analysis import probe
from life_sandbox.core import components as C

COMP = jnp.asarray(
    [
        [0.85, 0.15, 0.0],  # cold species
        [0.75, 0.25, 0.0],  # temperate species
        [0.60, 0.40, 0.0],  # nut species
    ],
    jnp.float32,
)


def test_animal_share_is_indexed_by_species_not_nutrient():
    kind = jnp.asarray([0, 2], jnp.int32)
    share = probe.animal_share_per_slot(COMP, kind)
    assert float(share[0]) == pytest.approx(0.15)
    assert float(share[1]) == pytest.approx(0.40)


def test_toxic_share_uses_the_toxin_column():
    comp = jnp.asarray(
        [[0.6, 0.4, 0.0], [0.0, 0.0, 0.4], [0.0, 0.0, 0.6]], jnp.float32
    )
    kind = jnp.asarray([0, 1, 2], jnp.int32)
    toxic = probe.toxic_share_per_slot(comp, kind)
    assert [float(value) for value in toxic] == pytest.approx([0.0, 0.4, 0.6])


def test_cold_species_is_ineligible_for_animal_heavy_diet():
    rho = jnp.asarray([[0.05, 0.9, 0.05]], jnp.float32)
    kind = jnp.asarray([0, 1, 2], jnp.int32)
    alive = jnp.ones((3,), jnp.bool_)
    mask = probe.eligible_species_mask(rho, COMP, kind, alive, 0.002)
    assert not bool(mask[0, 0])  # cold: 0.15 animal
    assert bool(mask[0, 1])  # temperate: 0.25 animal
    assert bool(mask[0, 2])  # nut: 0.40 animal


def test_low_animal_diet_accepts_cold_bush():
    rho = jnp.asarray([[0.93, 0.02, 0.05]], jnp.float32)
    kind = jnp.asarray([0, 2], jnp.int32)
    alive = jnp.ones((2,), jnp.bool_)
    mask = probe.eligible_species_mask(rho, COMP, kind, alive, 0.002)
    assert bool(mask[0, 0])


def test_selection_prefers_eligible_species_over_nearer_ineligible_one():
    rho = jnp.asarray([[0.05, 0.9, 0.05]], jnp.float32)
    kind = jnp.asarray([0, 2], jnp.int32)  # slot 0 cold (ineligible), slot 1 nut
    alive = jnp.ones((2,), jnp.bool_)
    stock = jnp.ones((2,), jnp.float32)
    distance = jnp.asarray([[1.0, 10.0]], jnp.float32)
    slot = probe.select_bush_slot(
        rho, COMP, kind, alive, stock, distance, 0.002, jnp.asarray([0.1])
    )
    assert int(slot[0]) == 1


def test_selection_switches_away_from_toxic_bush_when_toxin_reserve_is_high():
    comp = jnp.asarray(
        [[0.6, 0.4, 0.0], [0.6, 0.4, 0.0], [0.6, 0.4, 0.0]], jnp.float32
    )
    comp = comp.at[0, C.TOXIN].set(0.4)  # slot 0 toxic
    rho = jnp.asarray([[0.35, 0.6, 0.05]], jnp.float32)
    kind = jnp.asarray([0, 2], jnp.int32)
    alive = jnp.ones((2,), jnp.bool_)
    stock = jnp.ones((2,), jnp.float32)
    distance = jnp.asarray([[1.0, 10.0]], jnp.float32)
    slot = probe.select_bush_slot(
        rho, comp, kind, alive, stock, distance, 0.002, jnp.asarray([1.0])
    )
    assert int(slot[0]) == 1


def test_stationary_selection_ignores_eligibility_and_picks_nearest():
    rho = jnp.asarray([[0.05, 0.9, 0.05]], jnp.float32)
    kind = jnp.asarray([0, 2], jnp.int32)
    alive = jnp.ones((2,), jnp.bool_)
    stock = jnp.ones((2,), jnp.float32)
    distance = jnp.asarray([[1.0, 10.0]], jnp.float32)
    slot = probe.select_bush_slot(
        rho, COMP, kind, alive, stock, distance, 0.002, jnp.asarray([0.0]), stationary=True
    )
    assert int(slot[0]) == 0
