"""Pure summarizers for the M1 diagnostics report (SANDBOX 9, PRINCIPLES 8).

scripts/diagnostics.py used to compute its JSON summary inline while running the
engine.  This module owns that summarisation now: every function is a pure
function of explicitly passed event streams / trajectory samples, so the rules
can be unit tested on hand-made data and the CLI stays a thin wrapper.

Data model (engine indices are decoded before they reach the pure functions):

    Birth        tick, uid, parent_uid, reseed
    Death        tick, uid, cause (name), nutrient (name or None),
                 same_lineage, loss (one entry per damage source)
    WorldRecord  per-world event stream: births by uid, deaths, bush bites by
                 uid, reseed counter, birth_fail reasons / parents / nutrients

WorldRecord.from_script_dict adapts the dictionaries that
diagnostics.run_worlds() builds today, and worlds_from_script_result maps a
whole result; the engine schema (core.events / core.components) is touched only
inside that adapter.

Conventions preserved from the M1 report, so swapping the CLI over is a no-op on
the JSON keys:

  * a birth with reseed=True is a rescue individual: tracked by births_reseed /
    lifespan_reseed, and never part of the reproduction-born cohorts;
  * R0 groups anyone with a birth record as "reproduction_born"; founders are
    individuals without a birth record.  births_total counts reproduction
    births (parent_uid >= 0), so founders that never reproduce stay in the
    founders denominator and lower its mean R0;
  * cohort_truncated only counts lives that could have completed the age cap:
    founders always, reproduction-born only when born_tick <= ticks - 2000 (= 5 *
    a_sen, SANDBOX 5.5);
  * intake_by_maturity uses the per-tick digested mass already accumulated by the
    runner (R_DIG * sum(G)); individual_ticks = sum(population samples) *
    sample_ticks;
  * births_by_parent is derived from the birth events themselves
    (parent_uid >= 0), grouped by (world, uid); a uid repeated across worlds is
    two different individuals.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from life_sandbox.core import components as C
from life_sandbox.core import events as events_mod

A_MATURE: int = 100  # SANDBOX 5.2: maturity a_mature; script's A_MATURE
TRUNCATION_AGE: int = 2000  # 5 * a_sen (SANDBOX 5.5): age cap for R0 truncation
NUTRIENT_LABELS: tuple[str, ...] = ("plant", "animal", "toxin")
DEFAULT_SAMPLE_TICKS: int = 128
LEDGER_KEYS: tuple[str, ...] = (
    "inflow",
    "outflow",
    "transfer_from",
    "transfer_to",
    "digested",
)


# --- inputs -----------------------------------------------------------------


@dataclass(frozen=True)
class Birth:
    """One birth event (reproduction or reseed rescue)."""

    tick: int
    uid: int
    parent_uid: int
    reseed: bool


@dataclass(frozen=True)
class Death:
    """One death event with the decaying damage-source vector (SANDBOX 5.5)."""

    tick: int
    uid: int
    cause: str
    nutrient: str | None
    same_lineage: bool
    loss: tuple[float, ...]


@dataclass(frozen=True)
class WorldRecord:
    """Per-world event stream, engine indices already decoded to names."""

    births: Mapping[int, Birth] = field(default_factory=dict)
    deaths: tuple[Death, ...] = ()
    bites: Mapping[int, int] = field(default_factory=dict)
    reseeds: int = 0
    birth_fail: Mapping[str, int] = field(default_factory=dict)
    birth_fail_parents: Mapping[str, frozenset[int]] = field(default_factory=dict)
    birth_fail_nutrients: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_script_dict(cls, data: Mapping[str, Any]) -> "WorldRecord":
        """Adapt the dict diagnostics.run_worlds() puts in result["worlds"]."""
        births = {
            int(uid): Birth(
                tick=int(tick),
                uid=int(uid),
                parent_uid=int(parent),
                reseed=bool(reseed),
            )
            for uid, (tick, reseed, parent) in data["births"].items()
        }
        deaths = []
        for tick, uid, cause, nutrient, same, loss in data["deaths"]:
            nutrient_index = int(nutrient)
            deaths.append(
                Death(
                    tick=int(tick),
                    uid=int(uid),
                    cause=events_mod.DEATH_CAUSES[int(cause)],
                    nutrient=(
                        NUTRIENT_LABELS[nutrient_index]
                        if 0 <= nutrient_index < len(NUTRIENT_LABELS)
                        else None
                    ),
                    same_lineage=bool(same),
                    loss=tuple(float(value) for value in loss),
                )
            )
        return cls(
            births=births,
            deaths=tuple(deaths),
            bites={int(uid): int(count) for uid, count in data["bites"].items()},
            reseeds=int(data["reseeds"]),
            birth_fail={str(reason): int(count) for reason, count in data["birth_fail"].items()},
            birth_fail_parents={
                str(reason): frozenset(int(parent) for parent in parents)
                for reason, parents in data["birth_fail_parents"].items()
            },
            birth_fail_nutrients={
                str(label): int(count) for label, count in data["birth_fail_nutrients"].items()
            },
        )


def worlds_from_script_result(result: Mapping[str, Any]) -> tuple[WorldRecord, ...]:
    """Map diagnostics.run_worlds()["worlds"] to WorldRecord tuples."""
    return tuple(WorldRecord.from_script_dict(data) for data in result["worlds"])


@dataclass(frozen=True)
class SummaryParams:
    """Experiment constants the summary needs (replaces the script's globals)."""

    founders: int
    capacity: int
    cohort_len: int
    ticks: int
    sample_ticks: int = DEFAULT_SAMPLE_TICKS
    mature_age: int = A_MATURE
    truncation_age: int = TRUNCATION_AGE
    nutrient_labels: tuple[str, ...] = NUTRIENT_LABELS
    n_damage_sources: int = C.N_DAMAGE_SOURCES


def empty_ledger(n_worlds: int) -> dict[str, np.ndarray]:
    """Zero ledger with the engine's category layout (useful in tests)."""
    return {
        "inflow": np.zeros((n_worlds, C.N_INFLOW_CATEGORIES, C.N_NUTRIENTS), dtype=np.float64),
        "outflow": np.zeros((n_worlds, C.N_OUTFLOW_CATEGORIES, C.N_NUTRIENTS), dtype=np.float64),
        "transfer_from": np.zeros((n_worlds, C.N_NUTRIENTS), dtype=np.float64),
        "transfer_to": np.zeros((n_worlds, C.N_NUTRIENTS), dtype=np.float64),
        "digested": np.zeros((n_worlds, C.N_NUTRIENTS), dtype=np.float64),
    }


# --- per-metric summaries ---------------------------------------------------


def births_by_parent(worlds: Sequence[WorldRecord]) -> dict[tuple[int, int], int]:
    """Offspring count per (world, parent uid), from the birth events only.

    A uid that appears in several worlds is one parent per world (SANDBOX 3.1:
    uids are per-world).  Reseed births have parent_uid < 0 and are skipped.
    """
    counts: dict[tuple[int, int], int] = {}
    for world, record in enumerate(worlds):
        for birth in record.births.values():
            if birth.parent_uid >= 0:
                key = (world, int(birth.parent_uid))
                counts[key] = counts.get(key, 0) + 1
    return counts


def lifespan_stats(ages: Sequence[int], mature_age: int = A_MATURE) -> dict:
    """Count, median and matured-only median of a list of lifespans."""
    ages = [int(age) for age in ages]
    if not ages:
        return {"count": 0, "median": None, "matured": 0, "matured_median": None}
    matured = [age for age in ages if age >= mature_age]
    return {
        "count": len(ages),
        "median": float(np.median(ages)),
        "matured": len(matured),
        "matured_median": float(np.median(matured)) if matured else None,
    }


def lifespan_summary(worlds: Sequence[WorldRecord], mature_age: int = A_MATURE) -> dict:
    """Lifespans split by birth origin (reproduction vs reseed rescue)."""
    reproduction: list[int] = []
    reseed: list[int] = []
    for record in worlds:
        for death in record.deaths:
            born = record.births.get(death.uid)
            if born is None:
                continue
            age = death.tick - born.tick
            (reseed if born.reseed else reproduction).append(age)
    return {
        "reproduction": lifespan_stats(reproduction, mature_age),
        "reseed": lifespan_stats(reseed, mature_age),
    }


def death_summary(
    worlds: Sequence[WorldRecord],
    nutrient_labels: Sequence[str] = NUTRIENT_LABELS,
    n_damage_sources: int = C.N_DAMAGE_SOURCES,
) -> dict:
    """Death causes, decaying-loss means, missing nutrients and bite deaths."""
    causes: Counter = Counter()
    nutrient_counts = {label: 0 for label in nutrient_labels}
    bite_deaths = {"same_lineage": 0, "other": 0}
    loss_by_cause: dict[str, list[tuple[float, ...]]] = {}
    for record in worlds:
        for death in record.deaths:
            causes[death.cause] += 1
            loss_by_cause.setdefault(death.cause, []).append(death.loss)
            if death.cause == "nutrition" and death.nutrient in nutrient_counts:
                nutrient_counts[death.nutrient] += 1
            if death.cause == "bite":
                bite_deaths["same_lineage" if death.same_lineage else "other"] += 1
    loss_summary = {
        name: [
            float(np.mean([row[index] for row in rows])) for index in range(n_damage_sources)
        ]
        for name, rows in loss_by_cause.items()
    }
    return {
        "causes": dict(causes),
        "loss_mean": loss_summary,
        "nutrition_types": nutrient_counts,
        "bite_deaths": bite_deaths,
    }


def per_world_rows(
    worlds: Sequence[WorldRecord],
    seeds: Sequence[int],
    populations: Sequence[Sequence[int]],
) -> list[dict]:
    """One summary row per world (population extremes, births, deaths, reseeds)."""
    pop = np.asarray(populations)
    rows = []
    for world, record in enumerate(worlds):
        born_groups = Counter(birth.reseed for birth in record.births.values())
        rows.append(
            {
                "seed": int(seeds[world]),
                "final_population": int(pop[-1, world]),
                "min_population": int(pop[:, world].min()),
                "max_population": int(pop[:, world].max()),
                "births_reproduction": int(born_groups.get(False, 0)),
                "births_reseed": int(born_groups.get(True, 0)),
                "deaths": len(record.deaths),
                "reseeds": int(record.reseeds),
            }
        )
    return rows


def population_summary(populations: Sequence[Sequence[int]], capacity: int) -> dict:
    """Population trajectory summary (rows, final, min, mean, capacity hits)."""
    pop = np.asarray(populations)
    if pop.size == 0:
        return {
            "population_rows": [],
            "population_final": [],
            "population_min": [],
            "population_mean": None,
            "capacity_fraction": None,
        }
    return {
        "population_rows": pop.tolist(),
        "population_final": pop[-1].tolist(),
        "population_min": pop.min(axis=0).tolist(),
        "population_mean": float(pop.mean()),
        "capacity_fraction": float(np.mean(pop >= capacity)),
    }


def cohort_summary(
    worlds: Sequence[WorldRecord],
    cohort_len: int,
) -> list[dict]:
    """Reproduction-born cohorts (birth tick // cohort_len) and their behaviour.

    Reseed rescue individuals are excluded from reproduction-born cohorts.
    """
    cohorts: dict[int, dict] = {}
    for record in worlds:
        for death in record.deaths:
            born = record.births.get(death.uid)
            if born is None or born.reseed:
                continue
            age = death.tick - born.tick
            key = int(born.tick // cohort_len)
            entry = cohorts.setdefault(
                key,
                {
                    "born": 0,
                    "bites": 0,
                    "lifespans": [],
                    "lifetime_ticks": 0,
                    "biters": 0,
                    "died": 0,
                },
            )
            entry["lifespans"].append(age)
            entry["lifetime_ticks"] += age
            entry["died"] += 1
        for uid, born in record.births.items():
            if born.reseed:
                continue
            key = int(born.tick // cohort_len)
            entry = cohorts.setdefault(
                key,
                {
                    "born": 0,
                    "bites": 0,
                    "lifespans": [],
                    "lifetime_ticks": 0,
                    "biters": 0,
                    "died": 0,
                },
            )
            entry["born"] += 1
            bites = int(record.bites.get(uid, 0))
            entry["bites"] += bites
            if bites:
                entry["biters"] += 1
    rows = []
    for key in sorted(cohorts):
        entry = cohorts[key]
        born = max(entry["born"], 1)
        ticks = entry["lifetime_ticks"]
        lifespans = entry["lifespans"]
        rows.append(
            {
                "cohort": key,
                "born": entry["born"],
                "died": entry["died"],
                "biters": entry["biters"],
                "bites_per_individual": entry["bites"] / born,
                "fraction_biters": entry["biters"] / born,
                "bites_per_lifetime_tick": (entry["bites"] / ticks) if ticks else None,
                "mean_lifespan": float(np.mean(lifespans)) if lifespans else None,
            }
        )
    return rows


def first_birth_age_summary(worlds: Sequence[WorldRecord]) -> dict:
    """Age at first birth, split by founder vs reproduction-born parent."""
    by_parent = births_by_parent(worlds)
    first_ticks: dict[tuple[int, int], int] = {}
    for (world, parent), _count in by_parent.items():
        record = worlds[world]
        first_ticks[(world, parent)] = min(
            birth.tick for birth in record.births.values() if birth.parent_uid == parent
        )
    founder_first: list[int] = []
    reproduction_first: list[int] = []
    for (world, parent), first_tick in first_ticks.items():
        born = worlds[world].births.get(parent)
        if born is None:
            founder_first.append(first_tick)
        else:
            reproduction_first.append(first_tick - born.tick)
    all_ages = founder_first + reproduction_first
    return {
        "count": len(all_ages),
        "median": float(np.median(all_ages)) if all_ages else None,
        "founders_median": float(np.median(founder_first)) if founder_first else None,
        "reproduction_median": (
            float(np.median(reproduction_first)) if reproduction_first else None
        ),
    }


def birth_fail_summary(worlds: Sequence[WorldRecord]) -> dict:
    """Birth-failure counts by reason, unique blocked parents and no_reserve nutrients.

    Parents are keyed by (world, uid) so a uid repeated across worlds is not
    collapsed; no_reserve is read straight from the birth_fail events
    (SANDBOX 5.5) instead of being re-derived from body state.
    """
    counts: Counter = Counter()
    nutrients: Counter = Counter()
    parents: dict[str, set[tuple[int, int]]] = {}
    for world, record in enumerate(worlds):
        counts.update(record.birth_fail)
        nutrients.update(record.birth_fail_nutrients)
        for reason, blocked in record.birth_fail_parents.items():
            parents.setdefault(reason, set()).update(
                (world, int(parent)) for parent in blocked
            )
    return {
        "counts": dict(counts),
        "unique_parents": {reason: len(value) for reason, value in parents.items()},
        "no_reserve_nutrients": dict(nutrients),
    }


def _group_stats(counts: Sequence[int], individuals_in_group: int) -> dict:
    return {
        "individuals": int(individuals_in_group),
        "births": int(sum(counts)),
        "mean_r0": (sum(counts) / individuals_in_group) if individuals_in_group else None,
        "reproducing_parents": len(counts),
        "mean_r0_reproducing": float(np.mean(counts)) if counts else None,
        "max_r0": int(max(counts)) if counts else 0,
        "fraction_reproducing_with_2plus": (
            float(np.mean([value >= 2 for value in counts])) if counts else None
        ),
        "fraction_reproducing_with_3plus": (
            float(np.mean([value >= 3 for value in counts])) if counts else None
        ),
    }


def _completed(dead: int, births: int) -> dict:
    return {
        "dead_individuals": int(dead),
        "births": int(births),
        "mean_r0": (births / dead) if dead else None,
    }


def r0_summary(
    worlds: Sequence[WorldRecord],
    founders: int,
    ticks: int,
    truncation_age: int = TRUNCATION_AGE,
) -> dict:
    """Reproductive number R0, including founder denominator and age truncation.

    R0 is derived from the birth events' parent_uid (grouped by world + uid).
    Founders that never reproduce stay in the denominator.  cohort_truncated
    counts only individuals born at or before ticks - truncation_age, which is
    the age cap 5 * a_sen (SANDBOX 5.5).
    """
    by_parent = births_by_parent(worlds)
    n_worlds = len(worlds)
    founders_total = int(founders) * n_worlds
    births_total = int(sum(by_parent.values()))
    individuals = founders_total + births_total

    founder_counts: list[int] = []
    reproduction_counts: list[int] = []
    for (world, parent), count in by_parent.items():
        if parent in worlds[world].births:
            reproduction_counts.append(int(count))
        else:
            founder_counts.append(int(count))

    r0 = {
        "births_total": births_total,
        "individuals_ever": individuals,
        "founders": _group_stats(founder_counts, founders_total),
        "reproduction_born": _group_stats(reproduction_counts, births_total),
        "mean_r0_all_individuals": births_total / max(individuals, 1),
        "completed_lifetimes": {},
    }

    by_world: dict[int, dict[int, int]] = {}
    for (world, parent), count in by_parent.items():
        by_world.setdefault(world, {})[parent] = count
    dead_founders = 0
    dead_reproduction = 0
    completed_births = {"founders": 0, "reproduction_born": 0}
    for world, record in enumerate(worlds):
        dead_uids = {death.uid for death in record.deaths}
        for death in record.deaths:
            if death.uid in record.births:
                dead_reproduction += 1
            else:
                dead_founders += 1
        for parent, count in by_world.get(world, {}).items():
            if parent in dead_uids:
                key = "reproduction_born" if parent in record.births else "founders"
                completed_births[key] += int(count)

    r0["completed_lifetimes"] = {
        "founders": _completed(dead_founders, completed_births["founders"]),
        "reproduction_born": _completed(dead_reproduction, completed_births["reproduction_born"]),
        "all": _completed(
            dead_founders + dead_reproduction,
            completed_births["founders"] + completed_births["reproduction_born"],
        ),
        "note": "only individuals with a death record inside the run; alive-at-end excluded",
    }

    all_counts = founder_counts + reproduction_counts
    r0["reproducing_parents"] = len(all_counts)
    r0["mean_r0_reproducing"] = float(np.mean(all_counts)) if all_counts else None
    r0["max_r0"] = int(max(all_counts)) if all_counts else 0
    r0["fraction_reproducing_with_2plus"] = (
        float(np.mean([value >= 2 for value in all_counts])) if all_counts else None
    )
    r0["fraction_reproducing_with_3plus"] = (
        float(np.mean([value >= 3 for value in all_counts])) if all_counts else None
    )

    cutoff = int(ticks) - int(truncation_age)
    trunc_n = {"founders": 0, "reproduction_born": 0}
    trunc_births = {"founders": 0, "reproduction_born": 0}
    if cutoff >= 0:
        for record in worlds:
            trunc_n["founders"] += int(founders)
            for birth in record.births.values():
                if not birth.reseed and birth.tick <= cutoff:
                    trunc_n["reproduction_born"] += 1
            for birth in record.births.values():
                if birth.parent_uid < 0:
                    continue
                parent_born = record.births.get(birth.parent_uid)
                parent_tick = 0 if parent_born is None else int(parent_born.tick)
                if parent_tick <= cutoff:
                    key = "reproduction_born" if parent_born is not None else "founders"
                    trunc_births[key] += 1

    def _trunc(group: str) -> dict:
        return {
            "individuals": trunc_n[group],
            "births": trunc_births[group],
            "mean_r0": (trunc_births[group] / trunc_n[group]) if trunc_n[group] else None,
        }

    r0["cohort_truncated"] = {
        "cutoff_tick": cutoff,
        "rule": "only individuals born at or before tick_end - "
        + str(int(truncation_age))
        + " (age cap); all such lives are complete",
        "founders": _trunc("founders"),
        "reproduction_born": _trunc("reproduction_born"),
        "all": {
            "individuals": trunc_n["founders"] + trunc_n["reproduction_born"],
            "births": trunc_births["founders"] + trunc_births["reproduction_born"],
            "mean_r0": (
                (trunc_births["founders"] + trunc_births["reproduction_born"])
                / (trunc_n["founders"] + trunc_n["reproduction_born"])
                if (trunc_n["founders"] + trunc_n["reproduction_born"])
                else None
            ),
        },
    }
    return r0


def population_ticks(
    populations: Sequence[Sequence[int]], sample_ticks: int
) -> float:
    """Individual-ticks represented by the population samples (script formula)."""
    return float(np.sum(np.asarray(populations))) * float(sample_ticks)


def intake_by_maturity(maturity: Sequence[float]) -> dict:
    """Intake split by maturity stage, from the runner's per-tick digested mass."""
    juvenile_digested, mature_digested, juvenile_ticks, mature_ticks = (
        float(maturity[0]),
        float(maturity[1]),
        float(maturity[2]),
        float(maturity[3]),
    )
    return {
        "juvenile": {
            "individual_ticks": juvenile_ticks,
            "digested_mass": juvenile_digested,
            "digested_per_individual_tick": (
                juvenile_digested / juvenile_ticks if juvenile_ticks else None
            ),
        },
        "mature": {
            "individual_ticks": mature_ticks,
            "digested_mass": mature_digested,
            "digested_per_individual_tick": (
                mature_digested / mature_ticks if mature_ticks else None
            ),
        },
        "note": "digested = R_DIG * sum(G) accumulated per tick; equals intake over a lifetime",
    }


def intake_summary(
    populations: Sequence[Sequence[int]],
    sample_ticks: int,
    harvested_mass: float,
    demand_mass: float,
    maturity: Sequence[float],
) -> dict:
    """Harvest/demand per individual-tick plus the maturity split."""
    individual_ticks = population_ticks(populations, sample_ticks)
    return {
        "intake_per_individual_tick": harvested_mass / max(individual_ticks, 1.0),
        "demand_per_individual_tick": demand_mass / max(individual_ticks, 1.0),
        "individual_ticks": individual_ticks,
        "intake_by_maturity": intake_by_maturity(maturity),
    }


def ledger_summary(ledger: Mapping[str, Any]) -> dict:
    """Float totals of every ledger field plus the intake/demand mass."""
    totals = {
        name: np.asarray(ledger[name]).sum(axis=0).tolist() for name in LEDGER_KEYS
    }
    harvest = float(np.asarray(ledger["inflow"])[:, C.INFLOW_BITES, :].sum())
    demand = float(np.asarray(ledger["outflow"])[:, C.OUTFLOW_METABOLISM, :].sum())
    return {"totals": totals, "harvested_mass": harvest, "demand_mass": demand}


def event_count_summary(counts: Sequence[Sequence[int]], event_kinds: Sequence[str]) -> dict:
    """Non-zero event totals by kind, summed over worlds."""
    if len(counts) == 0:
        return {}
    totals = np.asarray(counts, dtype=np.int64).sum(axis=0)
    return {
        name: int(value)
        for name, value in zip(event_kinds, totals.tolist())
        if value
    }


def track_summary(
    tracker: Sequence[Sequence[Mapping[str, Any]]],
    nutrient_labels: Sequence[str] = NUTRIENT_LABELS,
) -> dict:
    """Behaviour-tracking summary over per-tick samples (theta approach, stock)."""
    max_ratio: list[float] = []
    ever_above = 0
    above_fractions: list[float] = []
    limiting_counts: Counter = Counter()
    absolute_limiting: Counter = Counter()
    stock_means: list[float] = []
    low_stock: list[float] = []
    for samples in tracker:
        stats: dict[int, dict] = {}
        for sample in samples:
            alive = np.asarray(sample["alive"])
            stock = np.asarray(sample["stock"])[np.asarray(sample["bush_alive"])]
            if stock.size:
                stock_means.append(float(np.mean(stock)))
                low_stock.append(float(np.mean(stock < 0.1)))
            for index in np.nonzero(alive)[0]:
                uid = int(sample["uid"][index])
                ratio = float(
                    np.asarray(sample["sum_n"])[index]
                    / max(float(np.asarray(sample["theta"])[index]), 1e-6)
                )
                entry = stats.setdefault(
                    uid,
                    {"max": 0.0, "above": 0, "n": 0, "lim": Counter(), "abs_lim": Counter()},
                )
                entry["max"] = max(entry["max"], ratio)
                entry["above"] += int(ratio >= 1.0)
                entry["n"] += 1
                entry["lim"][int(sample["limiting"][index])] += 1
                entry["abs_lim"][int(sample["limiting_absolute"][index])] += 1
        for entry in stats.values():
            max_ratio.append(entry["max"])
            ever_above += int(entry["above"] > 0)
            above_fractions.append(entry["above"] / max(entry["n"], 1))
            limiting_counts[entry["lim"].most_common(1)[0][0]] += 1
            absolute_limiting[entry["abs_lim"].most_common(1)[0][0]] += 1
    return {
        "individuals_tracked": len(max_ratio),
        "median_max_sumN_over_theta": float(np.median(max_ratio)) if max_ratio else None,
        "fraction_ever_above_theta": ever_above / max(len(max_ratio), 1),
        "mean_fraction_time_above_theta": (
            float(np.mean(above_fractions)) if above_fractions else None
        ),
        "limiting_nutrient_most_common": {
            nutrient_labels[key]: value for key, value in limiting_counts.items()
        },
        "limiting_nutrient_absolute": {
            nutrient_labels[key]: value for key, value in absolute_limiting.items()
        },
        "bush_stock_mean": float(np.mean(stock_means)) if stock_means else None,
        "bush_stock_below_0_1_fraction": float(np.mean(low_stock)) if low_stock else None,
    }


def stationary_individuals(
    worlds: Sequence[WorldRecord],
    founder_uids: Sequence[Sequence[int]],
    stationary_meta: Sequence[Mapping[str, Any]],
    nutrient_labels: Sequence[str] = NUTRIENT_LABELS,
) -> list[dict]:
    """Per-founder rows of the stationary probe (script's individuals_export)."""
    rows = []
    for world, meta in enumerate(stationary_meta):
        record = worlds[world]
        uids = founder_uids[world]
        birth_ticks: dict[int, list[int]] = {}
        for birth in record.births.values():
            if birth.parent_uid >= 0:
                birth_ticks.setdefault(int(birth.parent_uid), []).append(int(birth.tick))
        for index in range(len(meta["rho"])):
            uid = int(uids[index])
            ticks = sorted(birth_ticks.get(uid, []))
            death = next((row for row in record.deaths if row.uid == uid), None)
            rows.append(
                {
                    "world": world,
                    "uid": uid,
                    "rho": [float(value) for value in meta["rho"][index]],
                    "theta": float(meta["theta"][index]),
                    "species": int(meta["species"][index]),
                    "temperature_geo_shade": float(meta["temperature_geo_shade"][index]),
                    "births": len(ticks),
                    "birth_ages": [int(tick) for tick in ticks],
                    "death_age": int(death.tick) if death is not None else None,
                    "death_cause": death.cause if death is not None else None,
                    "death_nutrient": death.nutrient if death is not None else None,
                }
            )
    return rows


# --- top-level composition --------------------------------------------------


def summarize(
    worlds: Sequence[WorldRecord],
    params: SummaryParams,
    *,
    seeds: Sequence[int],
    populations: Sequence[Sequence[int]],
    counts: Sequence[Sequence[int]] | None = None,
    ledger: Mapping[str, Any] | None = None,
    maturity: Sequence[float] | None = None,
    dropped_events: int = 0,
    tracker: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    founder_uids: Sequence[Sequence[int]] | None = None,
    stationary_meta: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    """Assemble the diagnostics JSON summary from pure inputs.

    Mirrors the keys of scripts/diagnostics.py:summarize(), so the CLI can call
    this directly once it is switched over.  Only the per-world event records
    (plus the raw counts/ledger/maturity aggregates) are required.
    """
    n_worlds = len(worlds)
    if counts is None:
        counts = [[0] * len(events_mod.EVENT_KINDS) for _ in range(n_worlds)]
    if ledger is None:
        ledger = empty_ledger(n_worlds)
    if maturity is None:
        maturity = (0.0, 0.0, 0.0, 0.0)
    else:
        # The runner accumulates a (world, 4) array; the report wants one
        # four-vector (same sum the script did inline).
        maturity = np.asarray(maturity, dtype=np.float64)
        if maturity.ndim > 1:
            maturity = maturity.sum(axis=0)

    pop_summary = population_summary(populations, params.capacity)
    per_world = per_world_rows(worlds, seeds, populations)
    lifespans = lifespan_summary(worlds, params.mature_age)
    deaths = death_summary(worlds, params.nutrient_labels, params.n_damage_sources)
    event_counts = event_count_summary(counts, events_mod.EVENT_KINDS)
    ledger_stats = ledger_summary(ledger)
    intake = intake_summary(
        populations,
        params.sample_ticks,
        ledger_stats["harvested_mass"],
        ledger_stats["demand_mass"],
        maturity,
    )
    track = track_summary(tracker, params.nutrient_labels) if tracker else None
    individuals_export = None
    if stationary_meta is not None and founder_uids is not None:
        individuals_export = stationary_individuals(
            worlds, founder_uids, stationary_meta, params.nutrient_labels
        )
    return {
        "intake_per_individual_tick": intake["intake_per_individual_tick"],
        "demand_per_individual_tick": intake["demand_per_individual_tick"],
        "individual_ticks": intake["individual_ticks"],
        "intake_by_maturity": intake["intake_by_maturity"],
        "individuals_export": individuals_export,
        "r0": r0_summary(
            worlds,
            params.founders,
            params.ticks,
            params.truncation_age,
        ),
        "first_birth_age": first_birth_age_summary(worlds),
        "birth_fail": birth_fail_summary(worlds),
        "track": track,
        "dropped_events": int(dropped_events),
        "event_counts": event_counts,
        "ledger_totals": ledger_stats["totals"],
        "population_rows": pop_summary["population_rows"],
        "per_world": per_world,
        "population_final": pop_summary["population_final"],
        "population_min": pop_summary["population_min"],
        "population_mean": pop_summary["population_mean"],
        "capacity_fraction": pop_summary["capacity_fraction"],
        "births_reproduction": int(
            sum(row["births_reproduction"] for row in per_world)
        ),
        "births_reseed": int(sum(row["births_reseed"] for row in per_world)),
        "deaths": int(sum(len(record.deaths) for record in worlds)),
        "reseeds": int(sum(record.reseeds for record in worlds)),
        "death_causes": deaths["causes"],
        "death_loss_mean": deaths["loss_mean"],
        "nutrition_types": deaths["nutrition_types"],
        "bite_deaths": deaths["bite_deaths"],
        "lifespan_reproduction": lifespans["reproduction"],
        "lifespan_reseed": lifespans["reseed"],
        "cohorts_reproduction": cohort_summary(worlds, params.cohort_len),
    }


# --- generation / species-level summaries (M1 K calibration) ----------------


def generation_summary(
    generations: Sequence[Mapping[int, int]],
    births_by_parent: Mapping[tuple[int, int], int],
) -> dict:
    """R0 per generation from a per-world uid -> generation map.

    generations[world] maps uid -> generation (founders 0, child = parent + 1,
    reseeded individuals generation 0).  births_by_parent keys are (world, uid).
    """
    gen_individuals: Counter = Counter()
    gen_births: Counter = Counter()
    for mapping in generations:
        for generation in mapping.values():
            gen_individuals[int(generation)] += 1
    for (world, parent), count in births_by_parent.items():
        mapping = generations[int(world)]
        generation = int(mapping.get(int(parent), 0))
        gen_births[generation] += int(count)
    summary = {
        "individuals_by_generation": {},
        "births_by_generation": {},
        "r0_by_generation": {},
    }
    for generation in sorted(set(gen_individuals) | set(gen_births)):
        individuals = gen_individuals.get(generation, 0)
        births = gen_births.get(generation, 0)
        summary["individuals_by_generation"][str(generation)] = individuals
        summary["births_by_generation"][str(generation)] = births
        summary["r0_by_generation"][str(generation)] = (
            births / individuals if individuals else None
        )
    return summary


def stock_by_species(
    samples_by_world: Sequence[Sequence[Mapping[str, Any]]],
    labels: Mapping[int, str],
) -> dict:
    """Mean stock and fraction below 0.1 per bush species across tracker samples."""
    result: dict = {}
    for kind_id, label in labels.items():
        means, lows = [], []
        for samples in samples_by_world:
            for sample in samples:
                mask = np.asarray(sample["bush_alive"]) & (
                    np.asarray(sample["bush_kind"]) == kind_id
                )
                if np.any(mask):
                    values = np.asarray(sample["stock"])[mask]
                    means.append(float(np.mean(values)))
                    lows.append(float(np.mean(values < 0.1)))
        result[label] = {
            "mean": float(np.mean(means)) if means else None,
            "below_0_1_fraction": float(np.mean(lows)) if lows else None,
        }
    return result


def end_population_rho(
    samples_by_world: Sequence[Sequence[Mapping[str, Any]]],
) -> dict:
    """Diet rho_A distribution over individuals alive in the last sample."""
    values = []
    for samples in samples_by_world:
        if not samples:
            continue
        last = samples[-1]
        alive = np.asarray(last["alive"])
        values.extend(float(v) for v in np.asarray(last["rho_animal"])[alive])
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "individuals": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "fraction_below_0_2": float(np.mean(array < 0.2)),
        "fraction_below_0_34": float(np.mean(array < 0.34)),
    }


def bites_by_species(
    bite_targets: Sequence[Mapping[int, int]],
    bush_uids: Sequence[Sequence[int]],
    bush_kinds: Sequence[Sequence[int]],
    labels: Mapping[int, str],
) -> dict:
    """Bush-bite counts split by bush species (target uid -> species)."""
    counts = {label: 0 for label in labels.values()}
    counts["unknown"] = 0
    for world, targets in enumerate(bite_targets):
        mapping = {
            int(uid): labels.get(int(kind), "unknown")
            for uid, kind in zip(bush_uids[world], bush_kinds[world])
        }
        for target, count in targets.items():
            counts[mapping.get(int(target), "unknown")] += int(count)
    return counts



def meat_intake_by_actor(events: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """Animal mass taken per actor from living/carcass bites (dn, not absorbed).

    bite events carry the mass actually taken into the stomach in dn; absorbed
    is zero in this engine (absorption happens later in metabolism), so the meat
    metric must read dn.
    """
    result: dict[int, float] = {}
    for event in events:
        if int(event["target_kind"]) not in (C.CREATURE, C.CARCASS):
            continue
        uid = int(event["uid"])
        result[uid] = result.get(uid, 0.0) + float(np.asarray(event["dn"])[C.ANIMAL])
    return result

