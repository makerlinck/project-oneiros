"""M1 round 2: detailed runs for the best round-1 configs (SANDBOX section 9).

For the top-3 configs from round 1 this driver runs 10 worlds x 20L ticks with
the per-tick event stream, for the normal arm and the randomize-genome control,
and computes the M1 acceptance quantities:

- L: median death age of individuals born after the ramp that reached maturity
  (approximated by age >= a_mature: true maturity also needs feeding, so this
  is a conservative lower bound for the pool),
- self-sustaining fraction: worlds without any reseed whose final population
  is at least 20,
- 20L persistence: after the ramp and 20*L ticks, the population is still alive,
- death-cause fractions, birth/death rates, capacity touches (per L chunk).

Batched event flattening runs on the host per 64-tick segment; the GPU runs
vmap(step) inside one lax.scan per segment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from life_sandbox.core import events as events_mod
from life_sandbox.core.state import population
from life_sandbox.core.step import _compiled_batch_segment
from life_sandbox.world.init import initial_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m1_search import config_from_params  # noqa: E402

A_MATURE = 100
SEGMENT = 64


def _world(states, index: int):
    return jax.tree.map(lambda value: value[index], states)


def _worlds(cfg, seeds):
    states = [initial_state(cfg, seed=seed) for seed in seeds]
    return jax.tree.map(lambda *values: jnp.stack(values), *states)


def _flatten_events(buffers, world: int):
    """Return the valid event rows of one world inside a stacked segment buffer."""
    counts = np.asarray(buffers.count)[:, world]
    capacity = buffers.kind.shape[2]
    mask = np.arange(capacity)[None, :] < counts[:, None]
    columns = {}
    for name in ("kind", "uid", "tick", "cause", "target_kind"):
        columns[name] = np.asarray(getattr(buffers, name))[:, world][mask]
    return columns


def run_worlds(cfg, seeds, l_ticks: int, cycles: int) -> dict:
    """Batched run with per-world event accumulation."""
    states = _worlds(cfg, seeds)
    worlds = [
        {"births": {}, "deaths": [], "reseeds": 0, "counts": np.zeros(len(events_mod.EVENT_KINDS))}
        for _ in seeds
    ]
    population_rows = []
    counts_total = np.zeros(len(events_mod.EVENT_KINDS))
    for _ in range(cycles):
        remaining = l_ticks
        while remaining > 0:
            size = min(SEGMENT, remaining)
            states, counts, buffers, _ledgers = _compiled_batch_segment(cfg, size)(states)
            counts_np = np.asarray(counts)
            counts_total += counts_np.sum(axis=0)
            for world in range(len(seeds)):
                worlds[world]["counts"] += counts_np[world]
                columns = _flatten_events(buffers, world)
                births = columns["kind"] == events_mod.kind_code("birth")
                for tick, uid in zip(columns["tick"][births].tolist(), columns["uid"][births].tolist()):
                    worlds[world]["births"][int(uid)] = int(tick)
                deaths = columns["kind"] == events_mod.kind_code("death")
                for tick, uid, cause in zip(
                    columns["tick"][deaths].tolist(),
                    columns["uid"][deaths].tolist(),
                    columns["cause"][deaths].tolist(),
                ):
                    worlds[world]["deaths"].append((int(tick), int(uid), int(cause)))
                worlds[world]["reseeds"] += int(np.sum(columns["kind"] == events_mod.kind_code("reseed")))
            remaining -= size
        population_rows.append([int(population(_world(states, i))) for i in range(len(seeds))])
    return {
        "worlds": worlds,
        "population": np.asarray(population_rows),
        "counts_total": counts_total,
        "states": states,
    }


def summarize(cfg, result, seeds, l_ticks: int) -> dict:
    pop = result["population"]
    ramp = int(cfg.world.ramp_ticks)
    capacity = int(cfg.capacities.creature)
    per_world = []
    all_lifespans: list[int] = []
    causes: dict[str, int] = {}
    mature_counts: list[int] = []
    for world, data in enumerate(result["worlds"]):
        ages = []
        for tick, uid, cause in data["deaths"]:
            born = data["births"].get(uid)
            if born is not None and born >= ramp:
                age = tick - born
                ages.append(age)
                if age >= A_MATURE:
                    all_lifespans.append(age)
                    mature_counts.append(world)
            name = events_mod.DEATH_CAUSES[cause]
            causes[name] = causes.get(name, 0) + 1
        lifespans = [age for age in ages if age >= A_MATURE]
        per_world.append(
            {
                "seed": seeds[world],
                "births": len(data["births"]),
                "deaths": len(data["deaths"]),
                "reseeds": data["reseeds"],
                "final_population": int(pop[-1, world]),
                "min_population": int(pop[:, world].min()),
                "max_population": int(pop[:, world].max()),
                "lifespan_mature_median": float(np.median(lifespans)) if lifespans else None,
                "mature_deaths": len(lifespans),
            }
        )
    l_estimate = float(np.median(all_lifespans)) if all_lifespans else None
    total_births = sum(len(data["births"]) for data in result["worlds"])
    total_deaths = sum(len(data["deaths"]) for data in result["worlds"])
    stable = sum(
        1
        for world, data in enumerate(result["worlds"])
        if data["reseeds"] == 0 and int(pop[-1, world]) >= 20
    )
    # 20L persistence: every cycle after ramp + 20*L must keep the world alive.
    persistent = None
    start_cycle = None
    if l_estimate is not None:
        start_cycle = int(np.ceil((ramp + 20.0 * l_estimate) / float(l_ticks)))
        if start_cycle < pop.shape[0]:
            alive = pop[start_cycle:, :] > 0
            persistent = float(np.mean(alive.all(axis=0)))
    capacity_fraction = float(np.mean(pop >= capacity))
    return {
        "params": None,
        "L_estimate": l_estimate,
        "L_samples": len(all_lifespans),
        "per_world": per_world,
        "births": total_births,
        "deaths": total_deaths,
        "birth_death_ratio": (total_births / total_deaths) if total_deaths else None,
        "reseeds": sum(data["reseeds"] for data in result["worlds"]),
        "stable_worlds": stable,
        "stable_fraction": stable / len(seeds),
        "persistence_20L_fraction": persistent,
        "persistence_start_cycle": start_cycle,
        "capacity_fraction": capacity_fraction,
        "death_causes": causes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="m1-round2")
    parser.add_argument("--summary", default="runs/m1_search/round2feas/summary.json")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--worlds", type=int, default=10)
    parser.add_argument("--l-ticks", type=int, default=512)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=9200)
    parser.add_argument("--out", default="runs/m1_search/round2")
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()

    ranked = json.loads(Path(args.summary).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detailed = []
    for rank, record in enumerate(ranked[: args.top]):
        params = record["params"]
        cfg = config_from_params(params)
        seeds = [args.seed_base + rank * 100 + i for i in range(args.worlds)]
        result = run_worlds(cfg, seeds, args.l_ticks, args.cycles)
        summary = summarize(cfg, result, seeds, args.l_ticks)
        summary["params"] = params
        summary["rank"] = rank
        summary["round1_index"] = record["index"]
        if args.control:
            control_cfg = cfg.replaced(
                reproduction=cfg.reproduction.__class__(randomize_genome=True)
            )
            control = run_worlds(control_cfg, seeds, args.l_ticks, args.cycles)
            control_summary = summarize(control_cfg, control, seeds, args.l_ticks)
            control_summary["params"] = params
            control_summary["arm"] = "randomize_genome"
            summary["control"] = control_summary
        detailed.append(summary)
        print(
            "rank",
            rank,
            "L",
            summary["L_estimate"],
            "stable",
            summary["stable_fraction"],
            "births",
            summary["births"],
            "deaths",
            summary["deaths"],
            "reseeds",
            summary["reseeds"],
            flush=True,
        )
    (out / "detailed.json").write_text(json.dumps(detailed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
