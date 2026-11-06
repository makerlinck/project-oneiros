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


# --- explicit probe parameters (ProbeParams dataclass) -----------------------


def test_probe_params_defaults_match_the_m1_probe():
    params = probe.ProbeParams()
    assert params.selection == "coverage_v3"
    assert params.hunger_threshold == pytest.approx(0.8)
    assert params.bite_force == pytest.approx(1.0)
    assert params.knob == pytest.approx(0.5)
    assert params.knob_mode == "true-b"
    assert params.stock_min == pytest.approx(probe.STOCK_MIN)
    assert params.toxic_share == pytest.approx(probe.TOXIC_SHARE)
    assert params.stationary is False
    assert params.skip_bite_rate == pytest.approx(0.0)


def test_probe_params_validate():
    with pytest.raises(ValueError):
        probe.ProbeParams(selection="v2")
    with pytest.raises(ValueError):
        probe.ProbeParams(knob_mode="b")
    with pytest.raises(ValueError):
        probe.ProbeParams(skip_bite_rate=1.5)
    with pytest.raises(ValueError):
        probe.ProbeParams(stock_min=-0.1)
    with pytest.raises(ValueError):
        probe.ProbeParams(toxic_share=2.0)


def _selection_scene():
    # Diet strongly favours the nut species; two nut bushes (near but empty, far
    # but stocked) and one stocked plant bush.
    rho = jnp.asarray([[0.1, 0.1, 0.8]], jnp.float32)
    kind = jnp.asarray([2, 2, 1], jnp.int32)
    alive = jnp.ones((3,), jnp.bool_)
    stock = jnp.asarray([0.0, 0.5, 0.5], jnp.float32)
    distance = jnp.asarray([[1.0, 5.0, 2.0]], jnp.float32)
    return rho, kind, alive, stock, distance


def test_selection_reads_stock_min_from_params():
    rho, kind, alive, stock, distance = _selection_scene()
    body_toxin = jnp.asarray([0.0])
    # Default threshold 0.05: the near nut bush is empty, so the stocked nut
    # bush wins even though it is farther away -> slot 1.
    default = probe.select_bush_slot(rho, COMP, kind, alive, stock, distance, body_toxin)
    assert int(default[0]) == 1
    # No stock threshold: the nearest nut bush is acceptable -> slot 0.
    loose = probe.select_bush_slot(
        rho,
        COMP,
        kind,
        alive,
        stock,
        distance,
        body_toxin,
        params=probe.ProbeParams(stock_min=0.0),
    )
    assert int(loose[0]) == 0
    # Legacy keyword still wins over the dataclass value.
    legacy = probe.select_bush_slot(
        rho,
        COMP,
        kind,
        alive,
        stock,
        distance,
        body_toxin,
        stock_min=0.0,
        params=probe.ProbeParams(stock_min=0.3),
    )
    assert int(legacy[0]) == 0


def test_selection_reads_toxic_share_from_params():
    comp = jnp.asarray(
        [[0.6, 0.4, 0.0], [0.75, 0.25, 0.0], [0.8, 0.5, 0.4]], jnp.float32
    )
    rho = jnp.asarray([[0.5, 0.5, 0.0]], jnp.float32)
    kind = jnp.asarray([0, 1, 2], jnp.int32)
    alive = jnp.ones((3,), jnp.bool_)
    stock = jnp.ones((3,), jnp.float32)
    distance = jnp.asarray([[2.0, 3.0, 1.0]], jnp.float32)
    body_toxin = jnp.asarray([1.0])  # toxin demand high -> look for a toxic species
    default = probe.select_bush_slot(rho, comp, kind, alive, stock, distance, body_toxin)
    assert int(default[0]) != 2  # default share 0.05 -> the toxic species is avoided
    tolerant = probe.select_bush_slot(
        rho,
        comp,
        kind,
        alive,
        stock,
        distance,
        body_toxin,
        params=probe.ProbeParams(toxic_share=0.5),
    )
    assert int(tolerant[0]) == 2  # toxic share 0.4 is acceptable now


def test_hungry_uses_hunger_threshold_and_stomach_capacity():
    stomach = jnp.asarray([0.1, 0.23, 0.24, 0.3], jnp.float32)
    mask = probe.hungry(stomach, probe.ProbeParams(hunger_threshold=0.8))
    assert [bool(value) for value in mask] == [True, True, False, False]
    # Lower threshold -> hungry only for a emptier stomach.
    strict = probe.hungry(stomach, probe.ProbeParams(hunger_threshold=0.4))
    assert [bool(value) for value in strict] == [True, False, False, False]


def test_knob_outputs_modes():
    b = jnp.asarray([0.4, 0.0, -0.6], jnp.float32)
    true_b = probe.knob_outputs(b, probe.ProbeParams(knob_mode="true-b"))
    assert [float(value) for value in true_b] == pytest.approx([0.0, 0.375, 1.0])
    constant = probe.knob_outputs(b, probe.ProbeParams(knob_mode="constant", knob=0.2))
    assert [float(value) for value in constant] == pytest.approx([0.2, 0.2, 0.2])
    zero = probe.knob_outputs(b, probe.ProbeParams(knob_mode="zero"))
    assert [float(value) for value in zero] == pytest.approx([0.0, 0.0, 0.0])


def test_bite_outputs_and_skip_bite_mask():
    ok = jnp.asarray([True, False])
    outputs = probe.bite_outputs(ok, probe.ProbeParams(bite_force=0.7))
    assert [float(value) for value in outputs] == pytest.approx([0.7, 0.0])

    none = probe.skip_bite_mask(3, 8, probe.ProbeParams(skip_bite_rate=0.0))
    assert none.shape == (8,)
    assert none.dtype == jnp.bool_
    assert not bool(jnp.any(none))
    all_skipped = probe.skip_bite_mask(3, 8, probe.ProbeParams(skip_bite_rate=1.0))
    assert bool(jnp.all(all_skipped))
    again = probe.skip_bite_mask(3, 8, probe.ProbeParams(skip_bite_rate=1.0))
    assert bool(jnp.array_equal(all_skipped, again))

