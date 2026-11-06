"""Unit tests for the pure M1 diagnostics summarizers (analysis/summary.py).

Every metric is checked against hand-computed numbers on hand-made event streams.
The cases the M1 report needs (SANDBOX 9):
  * the same uid appearing in several worlds stays several individuals;
  * founders that never reproduce stay in the R0 denominator;
  * total extinction;
  * reseed rescue individuals;
  * R0 cohort truncation at tick_end - 2000 (5 * a_sen);
  * intake split by maturity.
"""

from __future__ import annotations

from collections import Counter

import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.analysis import summary
from life_sandbox.analysis.summary import Birth, Death, SummaryParams, WorldRecord
from life_sandbox.core import components as C
from life_sandbox.core import events as events_mod


def birth(tick: int, uid: int, parent: int = -1, reseed: bool = False) -> Birth:
    return Birth(tick=int(tick), uid=int(uid), parent_uid=int(parent), reseed=bool(reseed))


def death(
    tick: int,
    uid: int,
    cause: str = "nutrition",
    nutrient: str | None = None,
    same_lineage: bool = False,
    loss: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
) -> Death:
    return Death(
        tick=int(tick),
        uid=int(uid),
        cause=cause,
        nutrient=nutrient,
        same_lineage=bool(same_lineage),
        loss=tuple(float(value) for value in loss),
    )


def world(
    births: list[Birth] | None = None,
    deaths: list[Death] | None = None,
    bites: dict[int, int] | None = None,
    reseeds: int = 0,
    birth_fail: dict[str, int] | None = None,
    birth_fail_parents: dict[str, set[int]] | None = None,
    birth_fail_nutrients: dict[str, int] | None = None,
) -> WorldRecord:
    return WorldRecord(
        births={item.uid: item for item in (births or [])},
        deaths=tuple(deaths or []),
        bites=dict(bites or {}),
        reseeds=int(reseeds),
        birth_fail=dict(birth_fail or {}),
        birth_fail_parents={
            reason: frozenset(parents)
            for reason, parents in (birth_fail_parents or {}).items()
        },
        birth_fail_nutrients=dict(birth_fail_nutrients or {}),
    )


# --- births / first birth age ------------------------------------------------


def test_births_by_parent_uses_parent_uid_and_ignores_reseed():
    worlds = [
        world(
            births=[
                birth(100, 10, parent=1),
                birth(1200, 11, parent=1),
                birth(200, 12, parent=2),
            ]
        ),
        world(births=[birth(50, 10, parent=2), birth(60, 99, parent=-1, reseed=True)]),
    ]
    counts = summary.births_by_parent(worlds)
    assert counts == {(0, 1): 2, (0, 2): 1, (1, 2): 1}


def test_first_birth_age_splits_founder_and_reproduction_parents():
    worlds = [world(births=[birth(100, 10, parent=1), birth(500, 20, parent=10)])]
    ages = summary.first_birth_age_summary(worlds)
    # founder parent 1 first birth at 100; parent 10 was born at 100 and first
    # reproduced at 500 -> 400.
    assert ages == {
        "count": 2,
        "median": 250.0,
        "founders_median": 100.0,
        "reproduction_median": 400.0,
    }


def test_uid_repeats_across_worlds_are_separate_individuals():
    # uid 1 is a founder in world 0 but a reproduction-born individual in world 1.
    w0 = world(births=[birth(10, 100, parent=1)])
    w1 = world(births=[birth(5, 1, parent=2), birth(300, 100, parent=1)])
    assert summary.births_by_parent([w0, w1]) == {(0, 1): 1, (1, 2): 1, (1, 1): 1}

    ages = summary.first_birth_age_summary([w0, w1])
    assert ages["count"] == 3
    # founders: world0 parent 1 at 10, world1 parent 2 at 5 -> median 7.5
    assert ages["founders_median"] == pytest.approx(7.5)
    # world1 parent 1 was born at tick 5, first child at 300 -> 295
    assert ages["reproduction_median"] == pytest.approx(295.0)

    # Lifespans use each world's own birth record, not another world's uid.
    w0 = world(
        births=[birth(10, 100, parent=1)],
        deaths=[death(70, 100, cause="cold")],
    )
    w1 = world(
        births=[birth(5, 1, parent=2)],
        deaths=[death(100, 1, cause="cold")],
    )
    lifespans = summary.lifespan_summary([w0, w1])
    assert lifespans["reproduction"]["count"] == 2
    assert lifespans["reproduction"]["median"] == pytest.approx(77.5)


# --- lifespans / deaths ------------------------------------------------------


def test_lifespan_stats_hand_computed():
    assert summary.lifespan_stats([], mature_age=100) == {
        "count": 0,
        "median": None,
        "matured": 0,
        "matured_median": None,
    }
    stats = summary.lifespan_stats([50, 100, 150], mature_age=100)
    assert stats == {
        "count": 3,
        "median": 100.0,
        "matured": 2,
        "matured_median": 125.0,
    }


def test_death_summary_causes_loss_and_nutrients():
    worlds = [
        world(
            deaths=[
                death(10, 1, cause="nutrition", nutrient="animal", loss=(1, 2, 3, 4, 5)),
                death(20, 2, cause="nutrition", nutrient="plant", loss=(3, 4, 5, 6, 7)),
                death(30, 3, cause="bite", same_lineage=True, loss=(0, 0, 0, 0, 0)),
                death(40, 4, cause="bite", same_lineage=False, loss=(2, 2, 2, 2, 2)),
            ]
        )
    ]
    summary_deaths = summary.death_summary(worlds)
    assert summary_deaths["causes"] == {"nutrition": 2, "bite": 2}
    assert summary_deaths["loss_mean"]["nutrition"] == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0])
    assert summary_deaths["loss_mean"]["bite"] == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])
    assert summary_deaths["nutrition_types"] == {"plant": 1, "animal": 1, "toxin": 0}
    assert summary_deaths["bite_deaths"] == {"same_lineage": 1, "other": 1}


def test_total_extinction():
    worlds = [
        world(
            births=[birth(0, 10, parent=1)],
            deaths=[
                death(10, 1, cause="nutrition", nutrient="plant"),
                death(20, 10, cause="cold"),
                death(30, 2, cause="age"),
            ],
        )
    ]
    populations = [[3], [1], [0]]
    r0 = summary.r0_summary(worlds, founders=2, ticks=1000)
    assert r0["births_total"] == 1
    assert r0["individuals_ever"] == 3  # 2 founders + 1 reproduction birth
    assert r0["completed_lifetimes"]["founders"]["dead_individuals"] == 2
    assert r0["completed_lifetimes"]["reproduction_born"]["dead_individuals"] == 1
    # One birth in total, completed lifetimes of all three dead individuals:
    # the founder parent's R0 among complete lifetimes is 1/3.
    assert r0["completed_lifetimes"]["all"]["mean_r0"] == pytest.approx(1.0 / 3.0)

    population = summary.population_summary(populations, capacity=4)
    assert population["population_final"] == [0]
    assert population["population_min"] == [0]
    assert population["capacity_fraction"] == pytest.approx(0.0)
    assert summary.death_summary(worlds)["causes"] == {"nutrition": 1, "cold": 1, "age": 1}
    # Lifespans only cover in-run-born individuals (founders have no birth record).
    assert summary.lifespan_summary(worlds)["reproduction"]["count"] == 1


# --- reseed rescue -----------------------------------------------------------


def test_reseed_individuals_are_tracked_separately():
    worlds = [
        world(
            births=[
                birth(50, 100, parent=-1, reseed=True),
                birth(60, 101, parent=1),
            ],
            deaths=[death(150, 100, cause="age"), death(160, 101, cause="bite")],
            reseeds=1,
        )
    ]
    lifespans = summary.lifespan_summary(worlds)
    assert lifespans["reproduction"]["count"] == 1
    assert lifespans["reproduction"]["median"] == pytest.approx(100.0)
    assert lifespans["reseed"]["count"] == 1
    assert lifespans["reseed"]["median"] == pytest.approx(100.0)

    cohorts = summary.cohort_summary(worlds, cohort_len=50)
    assert len(cohorts) == 1
    assert cohorts[0]["born"] == 1  # only the reproduction-born individual
    assert cohorts[0]["cohort"] == 1  # tick 60 -> 60 // 50

    rows = summary.per_world_rows(worlds, seeds=[7], populations=[[2], [1]])
    assert rows[0]["births_reproduction"] == 1
    assert rows[0]["births_reseed"] == 1
    assert rows[0]["reseeds"] == 1

    # Reseed births (parent_uid = -1) are not reproduction events for R0.
    r0 = summary.r0_summary(worlds, founders=1, ticks=1000)
    assert r0["births_total"] == 1
    assert r0["individuals_ever"] == 2


# --- R0 ----------------------------------------------------------------------


def test_never_reproducing_founders_stay_in_the_denominator():
    worlds = [world(births=[birth(100, 10, parent=1)]) for _ in range(2)]
    r0 = summary.r0_summary(worlds, founders=2, ticks=1000)
    assert r0["births_total"] == 2
    assert r0["individuals_ever"] == 6  # 2 worlds x 2 founders + 2 births
    founders = r0["founders"]
    assert founders["individuals"] == 4
    assert founders["births"] == 2
    assert founders["mean_r0"] == pytest.approx(0.5)
    assert founders["reproducing_parents"] == 2
    assert founders["mean_r0_reproducing"] == pytest.approx(1.0)
    assert r0["mean_r0_all_individuals"] == pytest.approx(2 / 6)
    assert r0["reproducing_parents"] == 2


def test_r0_group_stats_and_completed_lifetimes():
    worlds = [
        world(
            births=[birth(10, 10, parent=1), birth(250, 11, parent=10)],
            deaths=[death(300, 1), death(210, 10)],
        )
    ]
    r0 = summary.r0_summary(worlds, founders=2, ticks=1000)
    assert r0["births_total"] == 2
    assert r0["individuals_ever"] == 4
    assert r0["mean_r0_all_individuals"] == pytest.approx(0.5)
    assert r0["founders"]["births"] == 1
    assert r0["reproduction_born"]["births"] == 1
    assert r0["reproducing_parents"] == 2
    assert r0["mean_r0_reproducing"] == pytest.approx(1.0)
    assert r0["max_r0"] == 1
    assert r0["fraction_reproducing_with_2plus"] == pytest.approx(0.0)

    completed = r0["completed_lifetimes"]
    assert completed["founders"]["dead_individuals"] == 1
    assert completed["founders"]["births"] == 1
    assert completed["reproduction_born"]["dead_individuals"] == 1
    assert completed["reproduction_born"]["births"] == 1
    assert completed["all"]["mean_r0"] == pytest.approx(1.0)


def test_cohort_truncation_uses_tick_end_minus_2000():
    worlds = [
        world(
            births=[
                birth(1000, 10, parent=1),  # reproduction-born, before cutoff
                birth(3500, 11, parent=1),  # reproduction-born, after cutoff
                birth(2000, 12, parent=10),  # parent born 1000 -> inside
                birth(4000, 13, parent=11),  # parent born 3500 -> outside
            ]
        )
    ]
    r0 = summary.r0_summary(worlds, founders=1, ticks=5000)
    truncated = r0["cohort_truncated"]
    assert truncated["cutoff_tick"] == 3000
    # Founders are always counted; births of uid 10 (1000) and uid 12 (2000)
    # happened at or before 3000.
    assert truncated["founders"]["individuals"] == 1
    assert truncated["reproduction_born"]["individuals"] == 2
    # Births by parent 1 (a founder): uid 10 and 11 are both inside the parent
    # cutoff; births by parent 10 (born 1000) are inside, parent 11 (born 3500)
    # is outside.
    assert truncated["founders"]["births"] == 2
    assert truncated["reproduction_born"]["births"] == 1
    assert truncated["all"]["individuals"] == 3
    assert truncated["all"]["births"] == 3
    assert truncated["all"]["mean_r0"] == pytest.approx(1.0)

    # A run shorter than the age cap has no complete lifetimes to truncate.
    short = summary.r0_summary(worlds, founders=1, ticks=1000)
    assert short["cohort_truncated"]["cutoff_tick"] == -1000
    assert short["cohort_truncated"]["all"]["individuals"] == 0
    assert short["cohort_truncated"]["all"]["mean_r0"] is None


# --- cohorts ----------------------------------------------------------------


def test_cohort_summary_hand_computed():
    worlds = [
        world(
            births=[
                birth(100, 10, parent=1),
                birth(120, 11, parent=1),
                birth(600, 12, parent=1),
            ],
            deaths=[death(200, 10), death(620, 11)],
            bites={10: 4, 11: 0, 12: 2},
        )
    ]
    rows = summary.cohort_summary(worlds, cohort_len=512)
    assert [row["cohort"] for row in rows] == [0, 1]
    first = rows[0]
    assert first["born"] == 2
    # uid 10 (born 100, death 200 -> age 100) and uid 11 (born 120, death 620
    # -> age 500) are both in cohort 0.
    assert first["died"] == 2
    assert first["biters"] == 1
    assert first["bites_per_individual"] == pytest.approx(2.0)
    assert first["fraction_biters"] == pytest.approx(0.5)
    assert first["bites_per_lifetime_tick"] == pytest.approx(4.0 / 600.0)
    assert first["mean_lifespan"] == pytest.approx(300.0)
    second = rows[1]
    assert second["born"] == 1
    assert second["died"] == 0
    assert second["biters"] == 1
    assert second["bites_per_lifetime_tick"] is None
    assert second["mean_lifespan"] is None


# --- populations / per-world -------------------------------------------------


def test_per_world_rows_and_population_summary():
    worlds = [
        world(
            births=[birth(10, 100, parent=1), birth(20, 101, parent=-1, reseed=True)],
            deaths=[death(30, 100)],
            reseeds=1,
        ),
        world(deaths=[death(40, 1)]),
    ]
    populations = [[4, 2], [3, 1]]
    rows = summary.per_world_rows(worlds, seeds=[11, 22], populations=populations)
    assert rows[0] == {
        "seed": 11,
        "final_population": 3,
        "min_population": 3,
        "max_population": 4,
        "births_reproduction": 1,
        "births_reseed": 1,
        "deaths": 1,
        "reseeds": 1,
    }
    assert rows[1]["seed"] == 22
    assert rows[1]["final_population"] == 1
    assert rows[1]["max_population"] == 2

    population = summary.population_summary(populations, capacity=4)
    assert population["population_rows"] == [[4, 2], [3, 1]]
    assert population["population_final"] == [3, 1]
    assert population["population_min"] == [3, 1]
    assert population["population_mean"] == pytest.approx(2.5)
    # One of four samples is at capacity (4 >= 4).
    assert population["capacity_fraction"] == pytest.approx(0.25)
    assert summary.population_summary([], capacity=4)["population_mean"] is None


# --- birth failures ----------------------------------------------------------


def test_birth_fail_summary_keys_parents_by_world():
    worlds = [
        world(
            birth_fail={"no_reserve": 2, "occupied": 1},
            birth_fail_parents={"no_reserve": {5, 6}, "occupied": {5}},
            birth_fail_nutrients={"plant": 1, "animal": 1},
        ),
        world(
            birth_fail={"no_reserve": 1},
            birth_fail_parents={"no_reserve": {5}},
            birth_fail_nutrients={"toxin": 1},
        ),
    ]
    result = summary.birth_fail_summary(worlds)
    assert result["counts"] == {"no_reserve": 3, "occupied": 1}
    # uid 5 appears in both worlds: (0,5), (0,6), (1,5) = three blocked parents.
    assert result["unique_parents"] == {"no_reserve": 3, "occupied": 1}
    assert result["no_reserve_nutrients"] == {"plant": 1, "animal": 1, "toxin": 1}


# --- intake / ledger / events ------------------------------------------------


def test_intake_by_maturity_hand_computed():
    maturity = (2.0, 4.0, 10.0, 20.0)
    by_maturity = summary.intake_by_maturity(maturity)
    assert by_maturity["juvenile"] == {
        "individual_ticks": 10.0,
        "digested_mass": 2.0,
        "digested_per_individual_tick": pytest.approx(0.2),
    }
    assert by_maturity["mature"]["digested_per_individual_tick"] == pytest.approx(0.2)
    empty = summary.intake_by_maturity((0.0, 4.0, 0.0, 20.0))
    assert empty["juvenile"]["digested_per_individual_tick"] is None
    assert empty["mature"]["digested_per_individual_tick"] == pytest.approx(0.2)


def test_intake_and_demand_per_individual_tick():
    populations = [[4, 2], [6, 2]]
    intake = summary.intake_summary(
        populations,
        sample_ticks=128,
        harvested_mass=179.2,
        demand_mass=89.6,
        maturity=(2.0, 4.0, 10.0, 20.0),
    )
    assert intake["individual_ticks"] == pytest.approx(14 * 128)
    assert intake["intake_per_individual_tick"] == pytest.approx(0.1)
    assert intake["demand_per_individual_tick"] == pytest.approx(0.05)
    assert summary.population_ticks(populations, 128) == pytest.approx(14 * 128)


def test_ledger_summary_and_event_counts():
    ledger = summary.empty_ledger(2)
    ledger["inflow"][0, C.INFLOW_BITES, :] = [1.0, 2.0, 3.0]
    ledger["outflow"][1, C.OUTFLOW_METABOLISM, :] = [0.5, 0.5, 0.5]
    ledger["digested"][0] = [4.0, 5.0, 6.0]
    stats = summary.ledger_summary(ledger)
    assert stats["totals"]["inflow"][C.INFLOW_BITES] == [1.0, 2.0, 3.0]
    assert stats["totals"]["digested"] == [4.0, 5.0, 6.0]
    assert stats["harvested_mass"] == pytest.approx(6.0)
    assert stats["demand_mass"] == pytest.approx(1.5)

    kinds = events_mod.EVENT_KINDS
    counts = [[0] * len(kinds) for _ in range(2)]
    counts[0][events_mod.kind_code("birth")] = 1
    counts[1][events_mod.kind_code("birth")] = 2
    counts[1][events_mod.kind_code("death")] = 1
    assert summary.event_count_summary(counts, kinds) == {"birth": 3, "death": 1}
    assert summary.event_count_summary([], kinds) == {}


# --- tracking / stationary ---------------------------------------------------


def test_track_summary_hand_computed():
    samples = [
        {
            "uid": jnp.asarray([1]),
            "alive": jnp.asarray([True]),
            "sum_n": jnp.asarray([0.5]),
            "theta": jnp.asarray([0.4]),
            "limiting": jnp.asarray([0]),
            "limiting_absolute": jnp.asarray([2]),
            "stock": jnp.asarray([0.2, 0.05]),
            "bush_alive": jnp.asarray([True, True]),
        },
        {
            "uid": jnp.asarray([1]),
            "alive": jnp.asarray([True]),
            "sum_n": jnp.asarray([0.3]),
            "theta": jnp.asarray([0.4]),
            "limiting": jnp.asarray([1]),
            "limiting_absolute": jnp.asarray([0]),
            "stock": jnp.asarray([0.1, 0.05]),
            "bush_alive": jnp.asarray([True, True]),
        },
    ]
    result = summary.track_summary([samples], nutrient_labels=("plant", "animal", "toxin"))
    assert result["individuals_tracked"] == 1
    assert result["median_max_sumN_over_theta"] == pytest.approx(1.25)
    assert result["fraction_ever_above_theta"] == pytest.approx(1.0)
    assert result["mean_fraction_time_above_theta"] == pytest.approx(0.5)
    # Equal counts: most_common keeps the first-seen limiting nutrient.
    assert result["limiting_nutrient_most_common"] == {"plant": 1}
    assert result["limiting_nutrient_absolute"] == {"toxin": 1}
    assert result["bush_stock_mean"] == pytest.approx(0.1)
    assert result["bush_stock_below_0_1_fraction"] == pytest.approx(0.5)


def test_stationary_individuals_export():
    worlds = [
        world(
            births=[birth(30, 100, parent=1), birth(40, 101, parent=1)],
            deaths=[death(200, 1, cause="bite", nutrient="animal")],
        )
    ]
    meta = [
        {
            "rho": np.asarray([[0.5, 0.5, 0.0]]),
            "theta": np.asarray([0.7]),
            "species": np.asarray([1]),
            "temperature_geo_shade": np.asarray([0.2]),
        }
    ]
    rows = summary.stationary_individuals(worlds, founder_uids=[[1]], stationary_meta=meta)
    assert len(rows) == 1
    row = rows[0]
    assert row["uid"] == 1
    assert row["births"] == 2
    assert row["birth_ages"] == [30, 40]
    assert row["death_age"] == 200
    assert row["death_cause"] == "bite"
    assert row["death_nutrient"] == "animal"
    assert row["rho"] == pytest.approx([0.5, 0.5, 0.0])


# --- adapter + top-level composition -----------------------------------------


def test_worldrecord_from_script_dict_decodes_engine_indices():
    data = {
        "births": {10: (5, False, 1), 11: (6, True, -1)},
        "deaths": [
            (
                50,
                10,
                events_mod.cause_code("nutrition"),
                C.ANIMAL,
                True,
                (1.0, 2.0, 3.0, 4.0, 5.0),
            )
        ],
        "bites": Counter({10: 2}),
        "reseeds": 1,
        "birth_fail": Counter({"no_reserve": 1}),
        "birth_fail_parents": {"no_reserve": {1}},
        "birth_fail_nutrients": Counter({"plant": 1}),
    }
    record = WorldRecord.from_script_dict(data)
    assert record.births[10] == Birth(tick=5, uid=10, parent_uid=1, reseed=False)
    assert record.births[11].reseed is True
    assert record.deaths[0].cause == "nutrition"
    assert record.deaths[0].nutrient == "animal"
    assert record.deaths[0].same_lineage is True
    assert record.bites == {10: 2}
    assert record.reseeds == 1
    assert record.birth_fail == {"no_reserve": 1}
    assert record.birth_fail_parents["no_reserve"] == frozenset({1})
    assert record.birth_fail_nutrients == {"plant": 1}
    assert summary.worlds_from_script_result({"worlds": [data]})[0] == record


def test_summarize_matches_diagnostics_json_keys():
    worlds = [world(births=[birth(10, 100, parent=1)], reseeds=0)]
    params = SummaryParams(founders=1, capacity=4, cohort_len=512, ticks=1000)
    result = summary.summarize(
        worlds,
        params,
        seeds=[1],
        populations=[[2], [1]],
        counts=[[0] * len(events_mod.EVENT_KINDS)],
        maturity=(0.0, 0.0, 0.0, 0.0),
    )
    expected_keys = {
        "intake_per_individual_tick",
        "demand_per_individual_tick",
        "individual_ticks",
        "intake_by_maturity",
        "individuals_export",
        "r0",
        "first_birth_age",
        "birth_fail",
        "track",
        "dropped_events",
        "event_counts",
        "ledger_totals",
        "population_rows",
        "per_world",
        "population_final",
        "population_min",
        "population_mean",
        "capacity_fraction",
        "births_reproduction",
        "births_reseed",
        "deaths",
        "reseeds",
        "death_causes",
        "death_loss_mean",
        "nutrition_types",
        "bite_deaths",
        "lifespan_reproduction",
        "lifespan_reseed",
        "cohorts_reproduction",
    }
    assert set(result) == expected_keys
    assert result["births_reproduction"] == 1
    assert result["reseeds"] == 0
    assert result["individuals_export"] is None
    assert result["track"] is None
    assert result["intake_by_maturity"]["juvenile"]["individual_ticks"] == 0.0


def test_generation_summary_tracks_multi_world_uid_collisions():
    generations = [
        {1: 0, 2: 0, 10: 1},
        {1: 0, 2: 0, 10: 2},
    ]
    births_by_parent = {(0, 1): 3, (0, 2): 0, (1, 1): 1, (1, 10): 2}
    out = summary.generation_summary(generations, births_by_parent)
    assert out["individuals_by_generation"] == {"0": 4, "1": 1, "2": 1}
    assert out["births_by_generation"] == {"0": 4, "1": 0, "2": 2}
    assert out["r0_by_generation"]["0"] == pytest.approx(1.0)
    assert out["r0_by_generation"]["1"] == pytest.approx(0.0)
    assert out["r0_by_generation"]["2"] == pytest.approx(2.0)


def test_generation_summary_handles_missing_parent_as_founder():
    out = summary.generation_summary([{7: 0}], {(0, 99): 2})
    assert out["individuals_by_generation"] == {"0": 1}
    assert out["r0_by_generation"]["0"] == pytest.approx(2.0)


def test_stock_by_species_splits_kinds_and_low_stock_fraction():
    samples = [
        [
            {
                "bush_alive": np.asarray([True, True, False]),
                "bush_kind": np.asarray([0, 2, 0]),
                "stock": np.asarray([0.2, 0.4, 0.0]),
            }
        ],
        [
            {
                "bush_alive": np.asarray([True]),
                "bush_kind": np.asarray([2]),
                "stock": np.asarray([0.05]),
            }
        ],
    ]
    out = summary.stock_by_species(samples, {0: "cold", 2: "nut"})
    assert out["cold"]["mean"] == pytest.approx(0.2)
    assert out["cold"]["below_0_1_fraction"] == pytest.approx(0.0)
    assert out["nut"]["mean"] == pytest.approx((0.4 + 0.05) / 2)
    assert out["nut"]["below_0_1_fraction"] == pytest.approx(0.5)


def test_end_population_rho_uses_last_sample_alive_only():
    samples = [
        [
            {
                "alive": np.asarray([True, True]),
                "rho_animal": np.asarray([0.1, 0.3]),
            }
        ],
        [
            {
                "alive": np.asarray([True, False]),
                "rho_animal": np.asarray([0.2, 0.9]),
            }
        ],
    ]
    out = summary.end_population_rho(samples)
    assert out["individuals"] == 3
    assert out["mean"] == pytest.approx((0.1 + 0.3 + 0.2) / 3)
    assert out["fraction_below_0_2"] == pytest.approx(1 / 3)


def test_bites_by_species_maps_target_uid_per_world():
    targets = [{10: 3, 20: 1}, {10: 4}]
    bush_uids = [[10, 20], [10]]
    bush_kinds = [[2, 0], [0]]
    out = summary.bites_by_species(targets, bush_uids, bush_kinds, {0: "cold", 2: "nut"})
    assert out["nut"] == 3
    assert out["cold"] == 1 + 4
    assert out["unknown"] == 0

