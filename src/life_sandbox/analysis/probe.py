"""Pure helpers for the scripted calibration probe (SANDBOX.md §9).

The probe is a calibration tool only: it lives under analysis/, never appears in
an experiment arm, and reads true body state.  These helpers exist so the
probe's parameters and species-selection rules can be unit tested without
running the engine.

Probe settings used to be module globals in scripts/diagnostics.py (KNOB,
KNOB_MODE, HUNGER_THRESHOLD, BITE_FORCE, STOCK_MIN, ...).  They are now an
explicit ProbeParams dataclass: every helper takes one, so a run records its
probe settings in one object and no global state leaks between runs.

    ProbeParams field      script global it replaces
    hp.selection           (the v3 coverage rule; only one rule exists today)
    hp.hunger_threshold    HUNGER_THRESHOLD
    hp.bite_force          BITE_FORCE
    hp.knob                KNOB
    hp.knob_mode           KNOB_MODE
    hp.stock_min           STOCK_MIN
    hp.toxic_share         TOXIC_SHARE (toxin switch rule)
    hp.stationary          STATIONARY
    hp.pin_bush_stock      PIN_BUSH_STOCK
    hp.skip_bite_rate      SKIP_BITE_RATE
    hp.avoid_toxin         AVOID_TOXIN (reported, not acted on by this module)
    hp.min_stock           MIN_STOCK (reported)
    hp.avoid_hot           AVOID_HOT (reported)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp

from life_sandbox.core import components as C

STOCK_MIN: float = 0.05
TOXIC_SHARE: float = 0.05
# Stomach capacity G_cap (SANDBOX 5.1); the probe's hunger threshold is relative
# to it, matching the script's "stomach < HUNGER_THRESHOLD * 0.3".
STOMACH_CAP: float = 0.3

SELECTIONS: tuple[str, ...] = ("coverage_v3",)
KNOB_MODES: tuple[str, ...] = ("constant", "true-b", "zero")


@dataclass(frozen=True)
class ProbeParams:
    """Explicit settings of one scripted-probe run (no module globals)."""

    selection: str = "coverage_v3"
    hunger_threshold: float = 0.8
    bite_force: float = 1.0
    knob: float = 0.5
    knob_mode: str = "true-b"
    stock_min: float = STOCK_MIN
    toxic_share: float = TOXIC_SHARE
    stationary: bool = False
    pin_bush_stock: bool = False
    skip_bite_rate: float = 0.0
    avoid_toxin: bool = False
    min_stock: float = 0.0
    avoid_hot: float = 0.0

    def __post_init__(self) -> None:
        if self.selection not in SELECTIONS:
            raise ValueError(
                "unknown probe selection " + repr(self.selection) + "; expected " + repr(SELECTIONS)
            )
        if self.knob_mode not in KNOB_MODES:
            raise ValueError(
                "unknown knob mode " + repr(self.knob_mode) + "; expected " + repr(KNOB_MODES)
            )
        if not 0.0 <= self.skip_bite_rate <= 1.0:
            raise ValueError("skip_bite_rate must be in [0, 1]")
        if not 0.0 <= self.stock_min <= 1.0:
            raise ValueError("stock_min must be in [0, 1] (berry stock range)")
        if not 0.0 <= self.toxic_share <= 1.0:
            raise ValueError("toxic_share must be in [0, 1]")


def _resolve(
    params: ProbeParams | None,
    *,
    stationary: bool | None = None,
    stock_min: float | None = None,
) -> ProbeParams:
    resolved = ProbeParams() if params is None else params
    if stationary is not None:
        resolved = replace(resolved, stationary=bool(stationary))
    if stock_min is not None:
        resolved = replace(resolved, stock_min=float(stock_min))
    return resolved


def animal_share_per_slot(comp, kind):
    """Animal content of every bush slot: comp is (species, nutrient)."""
    return comp[:, C.ANIMAL][kind]


def toxic_share_per_slot(comp, kind):
    """Toxin content of every bush slot: comp is (species, nutrient)."""
    return comp[:, C.TOXIN][kind]


def coverage_scores(rho, comp):
    """Per-species coverage: min over P/A of c_{s,j} / sqrt(rho_j)."""
    ratios = comp[None, :, :2] / jnp.sqrt(jnp.maximum(rho[:, None, :2], 1e-12))
    return ratios.min(axis=-1)


def _rank_row(ranked, rank, kind, alive, stock, distance, stock_min):
    species = ranked[:, rank]
    eligible = alive[None, :] & (kind[None, :] == species[:, None])
    stocked = eligible & (stock[None, :] >= stock_min)
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
    stationary: bool | None = None,
    stock_min: float | None = None,
    params: ProbeParams | None = None,
):
    """Return the bush slot each creature targets (coverage rule, v3).

    'params' carries the explicit probe settings; the old 'stationary' and
    'stock_min' keyword arguments still override the matching fields so the
    existing script call sites keep working while it switches to ProbeParams.
    """
    settings = _resolve(params, stationary=stationary, stock_min=stock_min)
    if settings.stationary:
        return jnp.argmin(jnp.where(alive[None, :], distance, jnp.inf), axis=1)
    scores = coverage_scores(rho, comp)
    ranked = jnp.argsort(-scores, axis=1)
    top, _species = _rank_row(
        ranked, 0, kind, alive, stock, distance, settings.stock_min
    )
    second, species_second = _rank_row(
        ranked, 1, kind, alive, stock, distance, settings.stock_min
    )
    third, species_third = _rank_row(
        ranked, 2, kind, alive, stock, distance, settings.stock_min
    )

    toxic_per_species = comp[:, C.TOXIN]
    own_species = ranked[:, 0]
    needs_toxin = body_toxin >= 0.5 * rho[:, C.TOXIN]
    switch = needs_toxin & (toxic_per_species[own_species] > settings.toxic_share)

    second_ok = toxic_per_species[species_second] <= settings.toxic_share
    third_ok = toxic_per_species[species_third] <= settings.toxic_share
    alternative = jnp.where(second_ok[:, None], second, jnp.where(third_ok[:, None], third, top))
    row = jnp.where(switch[:, None], alternative, top)
    return jnp.argmin(row, axis=1)


def hungry(stomach, params: ProbeParams | None = None):
    """Probe hunger: stomach fill below hunger_threshold * G_cap (SANDBOX 5.1)."""
    settings = _resolve(params)
    return jnp.asarray(stomach) < jnp.float32(settings.hunger_threshold * STOMACH_CAP)


def knob_outputs(body_temperature, params: ProbeParams | None = None):
    """Metabolic knob of the scripted probe from the true core temperature b.

    Modes of the M1 diagnostics: "constant" (params.knob), "true-b" (0 when
    b > 0.3, 1 when b < -0.5, linear between) and "zero".
    """
    settings = _resolve(params)
    body_temperature = jnp.asarray(body_temperature, dtype=jnp.float32)
    if settings.knob_mode == "zero":
        return jnp.zeros_like(body_temperature)
    if settings.knob_mode == "true-b":
        return jnp.where(
            body_temperature > 0.3,
            jnp.float32(0.0),
            jnp.where(
                body_temperature < -0.5,
                jnp.float32(1.0),
                jnp.clip((0.3 - body_temperature) / 0.8, 0.0, 1.0),
            ),
        )
    return jnp.full_like(body_temperature, jnp.float32(settings.knob))


def bite_outputs(bite_ok, params: ProbeParams | None = None):
    """Bite output of the scripted probe: params.bite_force where bite_ok."""
    settings = _resolve(params)
    return jnp.where(jnp.asarray(bite_ok), jnp.float32(settings.bite_force), jnp.float32(0.0))


def skip_bite_mask(tick, n_slots: int, params: ProbeParams | None = None):
    """Deterministic bite-skip mask for the probe (per-tick pseudo-random slots).

    Mirrors the diagnostics rule (tick * 1103515245 + slot * 12345) % 100 <
    skip_bite_rate * 100, so a run stays deterministic without RNG keys.
    """
    settings = _resolve(params)
    slots = jnp.arange(int(n_slots), dtype=jnp.int32)
    if settings.skip_bite_rate <= 0.0:
        return jnp.zeros((int(n_slots),), dtype=jnp.bool_)
    tick_i32 = jnp.asarray(tick, dtype=jnp.int32)
    selector = (tick_i32 * jnp.int32(1103515245) + slots * jnp.int32(12345)) % jnp.int32(100)
    return selector < jnp.int32(int(settings.skip_bite_rate * 100))
