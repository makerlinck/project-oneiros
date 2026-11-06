"""M1 world calibration driver (SANDBOX section 9, stage 0).

Random search over the calibration knobs agreed in the review:
    bush_regen_rate, bush_regen_base, bush capacity, base_metabolism,
    founder_reserves, ramp_ticks, temperature_scale.

Round 1: N configs x W worlds x 20*L ticks, batched with vmap on one device.
The batch is one config's worlds at a time; on a 3090, W=16 reaches ~2400
world-ticks/s (measured), so a 24-config round is ~20-30 minutes including one
compile per config (the segment equals one L chunk).
Round 2: the best configs with more worlds plus the randomized-genome control.

The driver is a host-side experiment script: it never changes the engine, seeds
are exploratory (< 10_000) and every record stores the config and the seeds.
L is a placeholder (cfg.world.l_ticks) until the calibration record replaces it;
the L estimate uses births after the ramp and deaths whose age reaches
a_mature, which is an approximation (true maturity also needs feeding).

Usage:
    python scripts/m1_search.py --stage round1 --configs 32 --worlds 4
    python scripts/m1_search.py --stage round2 --configs-file runs/m1_search/round1/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import jax
import jax.numpy as jnp
import numpy as np

from life_sandbox.config import (
    BodyConfig,
    BrainConfig,
    Capacities,
    Config,
    LearningConfig,
    RunConfig,
    WorldConfig,
)
from life_sandbox.core import events as events_mod
from life_sandbox.core import accounting
from life_sandbox.core.state import population
from life_sandbox.core.step import run, run_batch
from life_sandbox.world.init import initial_state

A_MATURE = 100.0  # §5.2: mass reaches 1.0 after a_mature ticks of feeding


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


COARSE_GRID: list[dict] = [
    {"base_metabolism": metabolism, "temperature_scale": temperature}
    for metabolism in (0.0005, 0.0008, 0.0012)
    for temperature in (0.1, 0.3)
]


def coarse_config(index: int) -> tuple[Config, dict]:
    """Deterministic targeted sweep of the feasibility corner (coarse mode).

    Maximal food (180 bushes, high regeneration), maximal ramp and founder
    reserves; varies only the two knobs that the sanity runs showed to dominate
    mortality (metabolism and temperature amplitude).
    """
    cell = COARSE_GRID[index % len(COARSE_GRID)]
    params = {
        "bush_regen_rate": 0.10,
        "bush_regen_base": 1e-3,
        "bush_capacity": 180,
        "base_metabolism": cell["base_metabolism"],
        "founder_reserves": 0.5,
        "ramp_ticks": 600,
        "temperature_scale": cell["temperature_scale"],
    }
    base = Config.default()
    cfg = base.replaced(
        capacities=Capacities(creature=256, bush=params["bush_capacity"], carcass=64, neutral=3),
        world=WorldConfig(
            bush_regen_rate=params["bush_regen_rate"],
            bush_regen_base=params["bush_regen_base"],
            temperature_scale=params["temperature_scale"],
            founders=64,
            ramp_ticks=params["ramp_ticks"],
        ),
        body=BodyConfig(
            base_metabolism=params["base_metabolism"],
            founder_reserves=params["founder_reserves"],
        ),
        brain=BrainConfig(kind="plastic_rnn"),
        learning=LearningConfig(plasticity=False, prediction=False),
        run=RunConfig(seed=0, ticks=500, segment_ticks=64),
    )
    cfg.validate()
    return cfg, params


def sample_config(rng: random.Random, index: int) -> tuple[Config, dict]:
    """Draw one calibration config from the agreed round-1 space."""
    bush_capacity = rng.choice([60, 90, 120, 150, 180])
    params = {
        "bush_regen_rate": round(rng.uniform(0.02, 0.10), 5),
        "bush_regen_base": round(_log_uniform(rng, 1e-4, 1e-3), 7),
        "bush_capacity": bush_capacity,
        "base_metabolism": round(_log_uniform(rng, 0.0005, 0.0025), 7),
        "founder_reserves": round(rng.uniform(0.3, 0.5), 4),
        "ramp_ticks": int(rng.uniform(100, 600)),
        "temperature_scale": round(rng.uniform(0.1, 1.0), 3),
    }
    base = Config.default()
    cfg = base.replaced(
        capacities=Capacities(
            creature=256, bush=bush_capacity, carcass=64, neutral=3
        ),
        world=WorldConfig(
            bush_regen_rate=params["bush_regen_rate"],
            bush_regen_base=params["bush_regen_base"],
            temperature_scale=params["temperature_scale"],
            founders=64,
            ramp_ticks=params["ramp_ticks"],
        ),
        body=BodyConfig(
            base_metabolism=params["base_metabolism"],
            founder_reserves=params["founder_reserves"],
        ),
        brain=BrainConfig(kind="plastic_rnn"),
        learning=LearningConfig(plasticity=False, prediction=False),
        # Segment length is a throughput knob, but a 500-tick scan takes minutes
        # of XLA compile per config (measured on the 3090), so the search uses
        # 64-tick segments: compile is seconds and warmed throughput is ~2400
        # world-ticks/s for 16 worlds.
        run=RunConfig(seed=0, ticks=500, segment_ticks=64),
    )
    cfg.validate()
    return cfg, params


def _worlds(cfg: Config, seeds: list[int]) -> object:
    states = [initial_state(cfg, seed=seed) for seed in seeds]
    return jax.tree.map(lambda *values: jnp.stack(values), *states)


def _world_state(states, index: int):
    return jax.tree.map(lambda value: value[index], states)


def measure(cfg: Config, seeds: list[int], l_ticks: int, cycles: int) -> dict:
    """Run one config over every seed in lockstep and collect survival metrics."""
    states = _worlds(cfg, seeds)
    start_stock = np.stack([accounting.nutrient_stock(_world_state(states, i)) for i in range(len(seeds))])
    inflow = np.zeros((len(seeds), 2, 3), dtype=np.float64)
    outflow = np.zeros((len(seeds), 7, 3), dtype=np.float64)
    transfer_from = np.zeros((len(seeds), 3), dtype=np.float64)
    transfer_to = np.zeros((len(seeds), 3), dtype=np.float64)
    counts = np.zeros((len(seeds), len(events_mod.EVENT_KINDS)), dtype=np.int64)
    population_rows = []
    started = time.perf_counter()
    for _ in range(cycles):
        result = run_batch(states, cfg, l_ticks)
        states = result.states
        inflow += result.inflow
        outflow += result.outflow
        transfer_from += result.transfer_from
        transfer_to += result.transfer_to
        counts += result.counts
        population_rows.append([int(population(_world_state(states, i))) for i in range(len(seeds))])
    wall = time.perf_counter() - started
    end_stock = np.stack([accounting.nutrient_stock(_world_state(states, i)) for i in range(len(seeds))])
    delta = end_stock - start_stock
    inflow_total = inflow.sum(axis=1)
    outflow_total = outflow.sum(axis=1)
    residual = inflow_total - outflow_total - delta - transfer_from + transfer_to
    scale = np.maximum(1.0, np.maximum(np.abs(inflow_total), np.abs(outflow_total)))
    closure = np.abs(residual) / scale

    pop = np.asarray(population_rows)
    births = counts[:, events_mod.kind_code("birth")]
    reseeds = counts[:, events_mod.kind_code("reseed")]
    capacity = int(cfg.capacities.creature)
    capacity_ticks = int(np.sum(pop >= capacity))
    deaths = counts[:, events_mod.kind_code("death")]
    return {
        "seeds": seeds,
        "l_ticks": l_ticks,
        "cycles": cycles,
        "wall_seconds": round(wall, 2),
        "population_final": pop[-1].tolist(),
        "population_min": pop.min(axis=0).tolist(),
        "population_max": pop.max(axis=0).tolist(),
        "births": births.tolist(),
        "deaths": deaths.tolist(),
        "reseeds": reseeds.tolist(),
        "capacity_ticks": capacity_ticks,
        "closure_max_rel": float(np.max(closure)),
        "ledger_inflow": inflow.sum(axis=0).tolist(),
        "ledger_outflow": outflow.sum(axis=0).tolist(),
        "survived": [int(value > 0) for value in pop[-1]],
        "score": _score(pop, births, reseeds, capacity_ticks, len(seeds)),
    }


def _score(pop, births, reseeds, capacity_ticks: int, n_worlds: int) -> float:
    """Ranking score: sustainable populations first, rescues are heavily penalised.

    'survived' alone is misleading because reseeds keep dead worlds alive, so
    the score uses the fraction of worlds whose population never fell to the
    reseed threshold (8) and the time-averaged population, then births.
    """
    reseed_threshold = 8
    self_sustaining = float(np.mean(pop.min(axis=0) > reseed_threshold))
    mean_pop = float(np.mean(pop) / 256.0)
    birth_rate = float(np.sum(births)) / max(1, n_worlds)
    rescue_penalty = float(np.sum(reseeds)) / max(1, n_worlds)
    capacity_penalty = capacity_ticks / max(1, n_worlds)
    return round(
        100.0 * self_sustaining + 50.0 * mean_pop + 0.2 * birth_rate
        - 5.0 * rescue_penalty - 0.5 * capacity_penalty,
        4,
    )


def estimate_lifespan(cfg: Config, seed: int, l_ticks: int, cycles: int) -> dict:
    """Single-world run with events, used for the L estimate (round 2)."""
    state = initial_state(cfg, seed=seed)
    births: dict[int, int] = {}
    ages: list[int] = []
    causes: dict[str, int] = {}
    reseeds = 0
    total_ticks = cycles * l_ticks
    for _ in range(cycles):
        result = run(state, cfg, l_ticks, collect_events=True)
        state = result.state
        table = result.events
        if table is None or not len(table["uid"]):
            continue
        kind = table["kind"]
        uid = table["uid"]
        tick = table["tick"]
        cause = table["cause"]
        birth_mask = kind == events_mod.kind_code("birth")
        for b_tick, b_uid in zip(tick[birth_mask].tolist(), uid[birth_mask].tolist()):
            births[int(b_uid)] = int(b_tick)
        death_mask = kind == events_mod.kind_code("death")
        for d_tick, d_uid in zip(tick[death_mask].tolist(), uid[death_mask].tolist()):
            parent_birth = births.get(int(d_uid))
            if parent_birth is not None:
                ages.append(int(d_tick) - parent_birth)
        for code, name in enumerate(events_mod.DEATH_CAUSES):
            count = int(np.sum(death_mask & (cause == code)))
            if count:
                causes[name] = causes.get(name, 0) + count
        reseeds += int(np.sum(kind == events_mod.kind_code("reseed")))
    mature = [age for age in ages if age >= A_MATURE]
    return {
        "seed": seed,
        "ticks": total_ticks,
        "births": len(births),
        "deaths": len(ages),
        "death_causes": causes,
        "reseeds": reseeds,
        "lifespan_mature_median": float(np.median(mature)) if mature else None,
        "mature_deaths": len(mature),
        "population_final": int(population(state)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="m1-search")
    parser.add_argument("--stage", choices=("round1", "round2"), default="round1")
    parser.add_argument("--configs", type=int, default=32)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--l-ticks", type=int, default=500)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=9000)
    parser.add_argument("--out", default="runs/m1_search/round1")
    parser.add_argument("--configs-file", default=None)
    parser.add_argument("--control", action="store_true", help="also run the randomize_genome control")
    parser.add_argument("--sample-mode", choices=("random", "coarse"), default="random")
    args = parser.parse_args()
    if not args.control:
        args.control = args.stage == "round2"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "configs.jsonl"
    done = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["index"])
    rng = random.Random(args.seed_base + 17)
    ranked = []
    total = args.configs if args.stage == "round1" else 0
    for index in range(total):
        cfg, params = (
            coarse_config(index) if args.sample_mode == "coarse" else sample_config(rng, index)
        )
        if index in done:
            continue
        seeds = [args.seed_base + index * 100 + world for world in range(args.worlds)]
        metrics = measure(cfg, seeds, args.l_ticks, args.cycles)
        record = {
            "stage": args.stage,
            "index": index,
            "params": params,
            "config_hash": cfg.config_hash(),
            "metrics": metrics,
        }
        with records_path.open("a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        ranked.append(record)
        print("config", index, "score", metrics["score"], "survived", metrics["survived"], flush=True)

    if args.stage == "round1":
        ranked = [
            json.loads(line) for line in records_path.read_text().splitlines() if line.strip()
        ]
        ranked.sort(key=lambda record: record["metrics"]["score"], reverse=True)
        (out_dir / "summary.json").write_text(json.dumps(ranked, ensure_ascii=False, indent=2))
        print("top3:", [(record["index"], record["metrics"]["score"]) for record in ranked[:3]])
        return

    # round 2: detailed runs for the best configs, plus the randomize-genome control
    source = Path(args.configs_file or "runs/m1_search/round1/summary.json")
    ranked = json.loads(source.read_text()) if source.exists() else []
    detailed = []
    for record in ranked[:3]:
        base = Config.default()
        params = record["params"]
        cfg = base.replaced(
            capacities=Capacities(creature=256, bush=params["bush_capacity"], carcass=64, neutral=3),
            world=WorldConfig(
                bush_regen_rate=params["bush_regen_rate"],
                bush_regen_base=params["bush_regen_base"],
                temperature_scale=params["temperature_scale"],
                founders=64,
                ramp_ticks=params["ramp_ticks"],
            ),
            body=BodyConfig(
                base_metabolism=params["base_metabolism"],
                founder_reserves=params["founder_reserves"],
            ),
            brain=BrainConfig(kind="plastic_rnn"),
            learning=LearningConfig(plasticity=False, prediction=False),
            run=RunConfig(seed=0, ticks=args.l_ticks, segment_ticks=64),
        )
        worlds = [estimate_lifespan(cfg, args.seed_base + 5000 + i, args.l_ticks, args.cycles) for i in range(10)]
        entry = {"index": record["index"], "params": params, "worlds": worlds}
        if args.control:
            control_cfg = cfg.replaced(
                reproduction=cfg.reproduction.__class__(randomize_genome=True)
            )
            entry["control_randomize_genome"] = [
                estimate_lifespan(control_cfg, args.seed_base + 6000 + i, args.l_ticks, args.cycles)
                for i in range(10)
            ]
        detailed.append(entry)
        print("detailed", record["index"], "done", flush=True)
    (out_dir / "detailed.json").write_text(json.dumps(detailed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
