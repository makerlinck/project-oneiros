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


def test_coverage_picks_nut_for_balanced_diet():
    rho = jnp.asarray([[0.5, 0.5, 0.0]], jnp.float32)
    scores = probe.coverage_scores(rho, COMP)
    assert int(jnp.argmax(scores[0])) == 2


def test_coverage_does_not_pick_nut_for_plant_heavy_diet():
    rho = jnp.asarray([[0.9, 0.08, 0.02]], jnp.float32)
    scores = probe.coverage_scores(rho, COMP)
    assert int(jnp.argmax(scores[0])) == 1


def test_selection_picks_species_by_coverage():
    rho = jnp.asarray([[0.5, 0.5, 0.0]], jnp.float32)
    kind = jnp.asarray([0, 1, 2], jnp.int32)
    alive = jnp.ones((3,), jnp.bool_)
    stock = jnp.ones((3,), jnp.float32)
    distance = jnp.asarray([[1.0, 1.0, 10.0]], jnp.float32)
    slot = probe.select_bush_slot(rho, COMP, kind, alive, stock, distance, jnp.asarray([0.1]))
    assert int(slot[0]) == 2


def test_selection_uses_stock_threshold_005():
    rho = jnp.asarray([[0.5, 0.5, 0.0]], jnp.float32)
    kind = jnp.asarray([0, 1, 2], jnp.int32)
    alive = jnp.ones((3,), jnp.bool_)
    stock = jnp.asarray([0.01, 0.01, 0.2], jnp.float32)
    distance = jnp.asarray([[1.0, 2.0, 3.0]], jnp.float32)
    slot = probe.select_bush_slot(rho, COMP, kind, alive, stock, distance, jnp.asarray([0.1]))
    assert int(slot[0]) == 2


def test_selection_switches_away_from_toxic_species():
    comp = jnp.asarray([[0.6, 0.4, 0.0], [0.75, 0.25, 0.0], [0.6, 0.4, 0.0]], jnp.float32)
    comp = comp.at[2, C.TOXIN].set(0.4)
    rho = jnp.asarray([[0.5, 0.5, 0.0]], jnp.float32)
    kind = jnp.asarray([0, 1, 2], jnp.int32)
    alive = jnp.ones((3,), jnp.bool_)
    stock = jnp.ones((3,), jnp.float32)
    distance = jnp.asarray([[1.0, 2.0, 0.5]], jnp.float32)
    slot = probe.select_bush_slot(rho, comp, kind, alive, stock, distance, jnp.asarray([1.0]))
    assert int(slot[0]) != 2


def test_stationary_selection_ignores_coverage_and_picks_nearest():
    rho = jnp.asarray([[0.5, 0.5, 0.0]], jnp.float32)
    kind = jnp.asarray([0, 2], jnp.int32)
    alive = jnp.ones((2,), jnp.bool_)
    stock = jnp.ones((2,), jnp.float32)
    distance = jnp.asarray([[1.0, 10.0]], jnp.float32)
    slot = probe.select_bush_slot(
        rho, COMP, kind, alive, stock, distance, jnp.asarray([0.0]), stationary=True
    )
    assert int(slot[0]) == 0
