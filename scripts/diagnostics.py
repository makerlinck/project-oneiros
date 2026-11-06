"""M1 diagnostics on the design-value world (SANDBOX §9, PRINCIPLES §5).

Requested by the 2026-11-06 review:
  forager  -- a scripted forager probe (calibration tool ONLY: it lives here,
              is not a brain-registry entry and never appears in an experiment)
  fixed    -- fixed arm cohort statistics, for evolvability
  control  -- randomize-genome control cohort statistics

The scripted forager walks to the nearest living bush, stays there and bites
while the stomach is below 80% capacity.  If it keeps a population alive, the
design-value world is energy-feasible and the failure of the fixed arm is an
evolvability/behaviour problem rather than a world-design problem.

Outputs JSON with population trajectory, births/deaths, death causes (cause,
missing nutrient, same-lineage bites), capacity fraction, and per-cohort
foraging statistics (bush bites in the individual's lifetime, lifespan,
fraction that ever fed).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from life_sandbox.config import BrainConfig, Capacities, Config, LearningConfig, RunConfig, WorldConfig
from life_sandbox.core import components as C
from life_sandbox.core import events as events_mod
from life_sandbox.core.state import population
from life_sandbox.core.step import SYSTEMS, _event_counts, step as engine_step
from life_sandbox.core.struct import update
from life_sandbox.world import fields as world_fields
from life_sandbox.world.init import initial_state

CHUNK = 128
KNOB = 0.5  # scripted forager metabolic knob; CLI override for the temperature probe


def _wrap(angle):
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


KNOB_MODE = "constant"
AVOID_TOXIN = False


def forager_think(state, cfg):
    """Scripted forager: nearest bush, stay inside, bite when stomach is not full.

    --knob adaptive also raises the metabolic knob when the skin temperature
    difference says the environment is colder than the body (and lowers it in
    the heat); --avoid-toxin ignores berry species whose current composition is
    toxic.  Both are calibration-only variants of the same simple policy.
    """
    creature = state.entities.creature
    bush = state.entities.bush
    pos = creature.transform.pos
    theta = creature.transform.theta
    delta = bush.transform.pos[None, :, :] - pos[:, None, :]
    distance = jnp.linalg.norm(delta, axis=-1)
    distance = jnp.where(bush.alive[None, :], distance, jnp.inf)
    if AVOID_TOXIN:
        species = bush.edible.kind
        toxin_share = state.comp[:, C.TOXIN][species]
        distance = jnp.where(toxin_share[None, :] > 0.05, jnp.inf, distance)
    distance = jnp.where(jnp.isfinite(distance), distance, 1e6)
    nearest = jnp.argmin(distance, axis=1)
    target = bush.transform.pos[nearest]
    target_radius = bush.shape.radius[nearest]
    to_target = target - pos
    desired = jnp.arctan2(to_target[:, 1], to_target[:, 0])
    error = _wrap(desired - theta)
    reach = creature.shape.radius + target_radius + 0.2
    in_range = jnp.linalg.norm(to_target, axis=1) <= reach
    facing = jnp.abs(error) <= jnp.deg2rad(30.0)
    stomach = jnp.sum(creature.body.G, axis=1)
    hungry = stomach < 0.8 * 0.3
    zeros = jnp.zeros_like(theta)
    if KNOB_MODE == "adaptive":
        ambient, _ = world_fields.environment_at(state, cfg)
        skin = ambient - creature.body.b
        knob = jnp.clip(0.5 - 4.0 * skin, 0.0, 1.0)
    else:
        knob = jnp.full_like(theta, KNOB)
    outputs = jnp.stack(
        [
            jnp.where(in_range, -0.2, 1.0),  # approach, stop inside
            jnp.clip(error / 0.5, -1.0, 1.0),  # steer
            knob,  # metabolic knob
            jnp.where(in_range & facing & hungry, 1.0, 0.0),  # bite
            zeros,  # grab
            zeros,  # drop
        ],
        axis=1,
    )
    outputs = jnp.where(creature.alive[:, None], outputs, 0.0)
    motor = update(creature.motor, last_out=outputs)
    return update(state, entities=update(state.entities, creature=update(creature, motor=motor)))


def forager_think_system(state, cfg):
    """System-contract wrapper around the scripted policy (no events of its own)."""
    return forager_think(state, cfg), events_mod.empty(cfg)


_FORAGER_SYSTEMS = tuple(
    (name, forager_think_system if name == "think" else system) for name, system in SYSTEMS
)


def forager_step(state, cfg):
    """engine step order with the scripted forager in place of think."""
    state = update(state, ledger=C.Ledger.zeros())
    buffer = events_mod.empty(cfg)
    for _, system in _FORAGER_SYSTEMS:
        state, produced = system(state, cfg)
        buffer = events_mod.concat(buffer, produced)
    state = update(state, tick=(state.tick + jnp.asarray(1, jnp.int32)).astype(jnp.int32))
    return state, buffer


def _stack(states):
    return jax.tree.map(lambda *values: jnp.stack(values), *states)


def _world(states, index):
    return jax.tree.map(lambda value: value[index], states)


def _scan_chunk(step_fn, states, cfg, size: int, collect: bool):
    batched = jax.vmap(lambda state: step_fn(state, cfg))

    def body(carry, _):
        current, totals = carry
        current, buffer = batched(current)
        totals = totals + jax.vmap(_event_counts)(buffer)
        if collect:
            return (current, totals), (buffer,)
        return (current, totals), None

    initial = jnp.zeros((states.tick.shape[0], len(events_mod.EVENT_KINDS)), dtype=jnp.int32)
    if collect:
        (states, totals), (buffers,) = jax.lax.scan(body, (states, initial), None, length=size)
        return states, totals, buffers
    (states, totals), _ = jax.lax.scan(body, (states, initial), None, length=size)
    return states, totals, None


def _flatten(buffers, world: int) -> dict:
    counts = np.asarray(buffers.count)[:, world]
    capacity = buffers.kind.shape[2]
    mask = np.arange(capacity)[None, :] < counts[:, None]
    columns = {}
    for name in ("kind", "uid", "tick", "cause", "nutrient", "target_kind", "same_lineage"):
        columns[name] = np.asarray(getattr(buffers, name))[:, world][mask]
    return columns


def run_worlds(step_fn, cfg, seeds, ticks: int, collect: bool, chunk: int = CHUNK) -> dict:
    states = _stack([initial_state(cfg, seed=seed) for seed in seeds])
    worlds = [
        {"births": {}, "deaths": [], "bites": Counter(), "reseeds": 0} for _ in seeds
    ]
    totals = np.zeros((len(seeds), len(events_mod.EVENT_KINDS)), dtype=np.int64)
    populations = []
    jitted = jax.jit(lambda current: _scan_chunk(step_fn, current, cfg, chunk, collect))
    started = None
    import time

    started = time.perf_counter()
    for _ in range(ticks // chunk):
        states, counts, buffers = jitted(states)
        totals += np.asarray(counts)
        if collect:
            for world in range(len(seeds)):
                columns = _flatten(buffers, world)
                births = columns["kind"] == events_mod.kind_code("birth")
                for tick, uid in zip(columns["tick"][births].tolist(), columns["uid"][births].tolist()):
                    worlds[world]["births"][int(uid)] = int(tick)
                deaths = columns["kind"] == events_mod.kind_code("death")
                for tick, uid, cause, nutrient, same in zip(
                    columns["tick"][deaths].tolist(),
                    columns["uid"][deaths].tolist(),
                    columns["cause"][deaths].tolist(),
                    columns["nutrient"][deaths].tolist(),
                    columns["same_lineage"][deaths].tolist(),
                ):
                    worlds[world]["deaths"].append(
                        (int(tick), int(uid), int(cause), int(nutrient), bool(same))
                    )
                bites = (columns["kind"] == events_mod.kind_code("bite")) & (
                    columns["target_kind"] == C.BUSH
                )
                for uid in columns["uid"][bites].tolist():
                    worlds[world]["bites"][int(uid)] += 1
                worlds[world]["reseeds"] += int(np.sum(columns["kind"] == events_mod.kind_code("reseed")))
        populations.append([int(population(_world(states, i))) for i in range(len(seeds))])
    wall = time.perf_counter() - started
    return {
        "wall_seconds": round(wall, 2),
        "populations": populations,
        "counts": totals,
        "worlds": worlds,
        "states": states,
    }


def summarize(cfg, result, seeds, cohort_len: int) -> dict:
    pop = np.asarray(result["populations"])
    capacity = int(cfg.capacities.creature)
    causes: Counter = Counter()
    nutrient_counts = [0, 0, 0]
    bite_deaths = {"same_lineage": 0, "other": 0}
    per_world = []
    cohorts: dict[int, dict] = {}
    all_lifespans = []
    for world, data in enumerate(result["worlds"]):
        lifespans = []
        for tick, uid, cause, nutrient, same in data["deaths"]:
            born = data["births"].get(uid)
            causes[events_mod.DEATH_CAUSES[cause]] += 1
            if cause == events_mod.cause_code("nutrition") and 0 <= nutrient < C.N_NUTRIENTS:
                nutrient_counts[int(nutrient)] += 1
            if cause == events_mod.cause_code("bite"):
                bite_deaths["same_lineage" if same else "other"] += 1
            if born is not None:
                lifespans.append(tick - born)
        all_lifespans.extend(lifespans)
        per_world.append(
            {
                "seed": seeds[world],
                "final_population": int(pop[-1, world]),
                "min_population": int(pop[:, world].min()),
                "max_population": int(pop[:, world].max()),
                "births": len(data["births"]),
                "deaths": len(data["deaths"]),
                "reseeds": data["reseeds"],
                "lifespan_median": float(np.median(lifespans)) if lifespans else None,
            }
        )
        for uid, born in data["births"].items():
            key = int(born // cohort_len)
            entry = cohorts.setdefault(key, {"born": 0, "bites": 0, "lifespans": [], "biters": 0})
            entry["born"] += 1
            bites = data["bites"].get(uid, 0)
            entry["bites"] += bites
            if bites:
                entry["biters"] += 1
            for death in data["deaths"]:
                if death[1] == uid:
                    entry["lifespans"].append(death[0] - born)
                    break
    cohort_rows = []
    for key in sorted(cohorts):
        entry = cohorts[key]
        born = max(entry["born"], 1)
        cohort_rows.append(
            {
                "cohort": key,
                "born": entry["born"],
                "biters": entry["biters"],
                "bites_per_individual": entry["bites"] / born,
                "fraction_biters": entry["biters"] / born,
                "mean_lifespan": float(np.mean(entry["lifespans"])) if entry["lifespans"] else None,
            }
        )
    return {
        "per_world": per_world,
        "population_final": pop[-1].tolist(),
        "population_min": pop.min(axis=0).tolist(),
        "population_mean": float(pop.mean()),
        "capacity_fraction": float(np.mean(pop >= capacity)),
        "births": int(sum(len(data["births"]) for data in result["worlds"])),
        "deaths": int(sum(len(data["deaths"]) for data in result["worlds"])),
        "reseeds": int(sum(data["reseeds"] for data in result["worlds"])),
        "death_causes": dict(causes),
        "nutrition_types": {
            "plant": nutrient_counts[C.PLANT],
            "animal": nutrient_counts[C.ANIMAL],
            "toxin": nutrient_counts[C.TOXIN],
        },
        "bite_deaths": bite_deaths,
        "lifespan_median": float(np.median(all_lifespans)) if all_lifespans else None,
        "cohorts": cohort_rows,
    }


def design_config(args) -> Config:
    if args.tiny:
        capacities = Capacities(creature=16, bush=6, carcass=2, neutral=1)
        world = WorldConfig(founders=8, ramp_ticks=args.ramp_ticks)
    else:
        capacities = Capacities()
        world = WorldConfig(ramp_ticks=args.ramp_ticks)
    cfg = Config.default().replaced(
        capacities=capacities,
        world=world,
        brain=BrainConfig(kind="plastic_rnn"),
        learning=LearningConfig(plasticity=False, prediction=False),
        run=RunConfig(seed=0, ticks=CHUNK, segment_ticks=CHUNK),
    )
    if args.temperature_scale is not None:
        cfg = cfg.replaced(
            world=replace(
                cfg.world,
                temperature_scale=float(args.temperature_scale),
                ramp_ticks=int(args.ramp_ticks),
            )
        )
    if args.arm == "control":
        cfg = cfg.replaced(reproduction=cfg.reproduction.__class__(randomize_genome=True))
    cfg.validate()
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(prog="m1-diagnostics")
    parser.add_argument("--arm", choices=("forager", "fixed", "control"), required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=10240)
    parser.add_argument("--ramp-ticks", type=int, default=0)
    parser.add_argument("--cohort", type=int, default=512)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--temperature-scale", type=float, default=None)
    parser.add_argument("--knob", type=float, default=0.5)
    parser.add_argument("--knob-mode", choices=("constant", "adaptive"), default="constant")
    parser.add_argument("--avoid-toxin", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    global KNOB, KNOB_MODE, AVOID_TOXIN
    KNOB = float(args.knob)
    KNOB_MODE = args.knob_mode
    AVOID_TOXIN = bool(args.avoid_toxin)
    cfg = design_config(args)
    seeds = [9300 + i for i in range(args.worlds)]
    step_fn = forager_step if args.arm == "forager" else engine_step
    result = run_worlds(step_fn, cfg, seeds, args.ticks, collect=True)
    summary = summarize(cfg, result, seeds, args.cohort)
    summary["arm"] = args.arm
    summary["design"] = {
        "base_metabolism": cfg.body.base_metabolism,
        "bush_capacity": cfg.capacities.bush,
        "bush_regen_rate": cfg.world.bush_regen_rate,
        "bush_regen_base": cfg.world.bush_regen_base,
        "temperature_scale": cfg.world.temperature_scale,
        "founder_reserves": cfg.body.founder_reserves,
        "ramp_ticks": cfg.world.ramp_ticks,
    }
    summary["seeds"] = seeds
    summary["ticks"] = args.ticks
    summary["wall_seconds"] = result["wall_seconds"]
    out = Path(args.out or ("runs/diagnostics/" + args.arm + ".json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ("arm", "births", "deaths", "reseeds", "population_final", "population_mean", "capacity_fraction", "death_causes", "nutrition_types", "bite_deaths", "lifespan_median")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
