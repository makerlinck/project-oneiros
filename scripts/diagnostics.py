"""M1 diagnostics on the design-value world (SANDBOX §9, PRINCIPLES §5).

Calibration tools only: the scripted forager lives here, is not a brain-registry
entry and never appears in an experiment arm.

Arms:
  fixed    -- the engine's fixed arm (B0, plasticity/prediction off)
  control  -- same with reproduction.randomize_genome = true
  forager  -- scripted probe: nearest bush of the diet-appropriate species, stay
              inside, bite only when the engine's nearest bite target is that
              bush, bite force 0.5, and set the metabolic knob from the real core
              temperature b (b > 0.3 -> 0, b < -0.5 -> 1, linear between).

Outputs JSON with population trajectory, births grouped by origin (reproduction
vs reseed), lifespans for both groups and for matured individuals, death causes
with the decaying loss vector, missing nutrients, same-lineage bite deaths, and
per-cohort foraging statistics for reproduction-born cohorts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
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
from life_sandbox.core.systems import reproduction as reproduction_system
from life_sandbox.core.systems.metabolism import R_DIG
from life_sandbox.analysis import probe
from life_sandbox.core.struct import update
from life_sandbox.world.init import diet_from_genome, initial_state

CHUNK = 128
A_MATURE = 100
NUTRIENT_LABELS = ("plant", "animal", "toxin")
KNOB = 0.5
KNOB_MODE = "zero"
BITE_FORCE = 1.0
AVOID_TOXIN = False
MIN_STOCK = 0.0
AVOID_HOT = 0.0
STATIONARY = False
PIN_BUSH_STOCK = False
SKIP_BITE_RATE = 0.0
HUNGER_THRESHOLD = 0.8
STOCK_MIN = 0.05
PINNED_BUSH_SLOTS = jnp.zeros((0,), dtype=jnp.int32)
PINNED_POS = None
PINNED_THETA = None
STATIONARY_META = None


def _wrap(angle):
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _entity_arrays(state):
    """Positions, radii, alive masks and prototype tags of every entity slot."""
    creature = state.entities.creature
    bush = state.entities.bush
    carcass = state.entities.carcass
    neutral = state.entities.neutral
    positions = jnp.concatenate(
        [
            creature.transform.pos,
            bush.transform.pos,
            carcass.transform.pos,
            neutral.transform.pos,
        ],
        axis=0,
    )
    radii = jnp.concatenate(
        [
            creature.shape.radius,
            bush.shape.radius,
            carcass.shape.radius,
            neutral.shape.radius,
        ]
    )
    alive = jnp.concatenate([creature.alive, bush.alive, carcass.alive, neutral.alive])
    kinds = jnp.concatenate(
        [
            jnp.full((creature.alive.shape[0],), C.CREATURE, dtype=jnp.int32),
            jnp.full((bush.alive.shape[0],), C.BUSH, dtype=jnp.int32),
            jnp.full((carcass.alive.shape[0],), C.CARCASS, dtype=jnp.int32),
            jnp.full((neutral.alive.shape[0],), C.NEUTRAL, dtype=jnp.int32),
        ]
    )
    local = jnp.concatenate(
        [
            jnp.arange(creature.alive.shape[0], dtype=jnp.int32),
            jnp.arange(bush.alive.shape[0], dtype=jnp.int32),
            jnp.arange(carcass.alive.shape[0], dtype=jnp.int32),
            jnp.arange(neutral.alive.shape[0], dtype=jnp.int32),
        ]
    )
    return positions, radii, alive, kinds, local


def forager_think(state, cfg):
    """Scripted competent forager (calibration tool; reads true body state)."""
    creature = state.entities.creature
    bush = state.entities.bush
    pos = creature.transform.pos
    theta = creature.transform.theta
    self_radius = creature.shape.radius
    n_creature = pos.shape[0]

    # Probe v2 (2026-11-06 review): a species is eligible only if its animal
    # supply covers the animal demand with a 1.3x margin at full digestion.
    # Selection lives in analysis/probe.py so it can be unit tested.
    diet = diet_from_genome(creature.genome.g, cfg)  # (N, 3)
    delta = bush.transform.pos[None, :, :] - pos[:, None, :]
    distance0 = jnp.where(bush.alive[None, :], jnp.linalg.norm(delta, axis=-1), jnp.inf)
    bush_slot = probe.select_bush_slot(
        diet,
        state.comp,
        bush.edible.kind,
        bush.alive,
        bush.edible.stock,
        distance0,
        creature.body.N[:, C.TOXIN],
        stationary=STATIONARY,
        stock_min=STOCK_MIN,
    )
    target = bush.transform.pos[bush_slot]
    target_radius = bush.shape.radius[bush_slot]
    to_target = target - pos
    desired = jnp.arctan2(to_target[:, 1], to_target[:, 0])
    error = _wrap(desired - theta)
    reach = self_radius + target_radius + 0.2
    in_range = jnp.linalg.norm(to_target, axis=1) <= reach
    facing = jnp.abs(error) <= jnp.deg2rad(30.0)
    if STATIONARY:
        facing = jnp.ones_like(facing)
    stomach = jnp.sum(creature.body.G, axis=1)
    hungry = stomach < HUNGER_THRESHOLD * 0.3

    # Bite only if the engine's nearest bite target is the chosen bush: the
    # engine picks the nearest entity within +-30 degrees and reach, so the
    # probe must reproduce that selection exactly.
    ent_pos, ent_radius, ent_alive, ent_kind, ent_local = _entity_arrays(state)
    delta_all = ent_pos[None, :, :] - pos[:, None, :]
    distance_all = jnp.linalg.norm(delta_all, axis=-1)
    angle_all = jnp.arctan2(delta_all[:, :, 1], delta_all[:, :, 0])
    error_all = _wrap(angle_all - theta[:, None])
    reach_all = self_radius[:, None] + ent_radius[None, :] + 0.2
    self_mask = (ent_kind[None, :] == C.CREATURE) & (
        ent_local[None, :] == jnp.arange(n_creature, dtype=jnp.int32)[:, None]
    )
    eligible = (
        ent_alive[None, :]
        & (distance_all <= reach_all)
        & (jnp.abs(error_all) <= jnp.deg2rad(30.0))
        & ~self_mask
    )
    candidate = jnp.where(eligible, distance_all, jnp.inf)
    nearest_entity = jnp.argmin(candidate, axis=1)
    bite_target_is_bush = (
        (ent_kind[nearest_entity] == C.BUSH) & (ent_local[nearest_entity] == bush_slot)
    )
    bite_ok = hungry & in_range & facing & bite_target_is_bush
    if SKIP_BITE_RATE > 0.0:
        slot = jnp.arange(n_creature, dtype=jnp.int32)
        selector = (state.tick * 1103515245 + slot * 12345) % 100
        skip = selector < int(SKIP_BITE_RATE * 100)
        bite_ok = bite_ok & ~skip

    if KNOB_MODE == "zero":
        knob = jnp.zeros_like(theta)
    elif KNOB_MODE == "true-b":
        body_temperature = creature.body.b
        knob = jnp.where(
            body_temperature > 0.3,
            0.0,
            jnp.where(
                body_temperature < -0.5,
                1.0,
                jnp.clip((0.3 - body_temperature) / 0.8, 0.0, 1.0),
            ),
        )
    else:
        knob = jnp.full_like(theta, KNOB)

    zeros = jnp.zeros_like(theta)
    if STATIONARY:
        accel = zeros
        turn = zeros
    else:
        accel = jnp.where(in_range, -0.2, 1.0)
        turn = jnp.clip(error / 0.5, -1.0, 1.0)
    outputs = jnp.stack(
        [
            accel,
            turn,
            knob,
            jnp.where(bite_ok, BITE_FORCE, 0.0),
            zeros,
            zeros,
        ],
        axis=1,
    )
    outputs = jnp.where(creature.alive[:, None], outputs, 0.0)
    motor = update(creature.motor, last_out=outputs)
    return update(state, entities=update(state.entities, creature=update(creature, motor=motor)))


def forager_think_system(state, cfg):
    return forager_think(state, cfg), events_mod.empty(cfg)


_FORAGER_SYSTEMS = tuple(
    (name, forager_think_system if name == "think" else system) for name, system in SYSTEMS
)


def forager_step(state, cfg):
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


def stationary_setup(states, cfg):
    """Place each founder at the centre of its own bush (diagnostic only)."""
    global PINNED_BUSH_SLOTS, PINNED_POS, PINNED_THETA, STATIONARY_META
    slots = jnp.arange(int(cfg.world.founders), dtype=jnp.int32)
    PINNED_BUSH_SLOTS = slots
    creature = states.entities.creature
    meta_all = []
    for world in range(int(states.tick.shape[0])):
        wstate = _world(states, world)
        w_creature = wstate.entities.creature
        bush = wstate.entities.bush
        kinds = np.asarray(bush.edible.kind)
        alive = np.asarray(bush.alive)
        rho = np.asarray(diet_from_genome(w_creature.genome.g[: int(cfg.world.founders)], cfg))
        comp = np.asarray(wstate.comp)
        coverage = comp[None, :, :2] / np.maximum(rho[:, None, :2], 1e-6)
        best_species = np.argmax(coverage.min(axis=-1), axis=1)
        chosen = np.zeros(int(cfg.world.founders), dtype=np.int64)
        for i, species in enumerate(best_species):
            candidates = np.nonzero(alive & (kinds == species))[0]
            chosen[i] = int(candidates[0]) if candidates.size else int(np.nonzero(alive)[0][0])
        bush_pos_arr = np.asarray(bush.transform.pos)[chosen]
        bush_pos = jnp.asarray(bush_pos_arr)
        theta = np.asarray(reproduction_system.theta_repro(w_creature, cfg))[: int(cfg.world.founders)]
        scaled_t = cfg.world.temperature_scale * (
            -2.0 + 4.0 * bush_pos_arr[:, 1] / cfg.world.height
        ) - 0.5
        meta_all.append(
            {
                "rho": rho,
                "theta": theta,
                "species": np.asarray(kinds)[chosen],
                "temperature_geo_shade": scaled_t,
            }
        )
        transform = w_creature.transform
        transform = update(
            transform,
            pos=transform.pos.at[slots].set(bush_pos),
            theta=transform.theta.at[slots].set(0.0),
            v=transform.v.at[slots].set(0.0),
        )
        w_creature = update(w_creature, transform=transform)
        creature = jax.tree.map(
            lambda stack, leaf: stack.at[world].set(leaf), creature, w_creature
        )
    PINNED_POS = creature.transform.pos
    PINNED_THETA = creature.transform.theta
    STATIONARY_META = meta_all
    return update(states, entities=update(states.entities, creature=creature))


def pin_stationary(states):
    """Keep the stationary founders at their pinned positions (stacked worlds)."""
    if PINNED_POS is None:
        return states
    creature = states.entities.creature
    n = PINNED_BUSH_SLOTS.size
    pinned = (jnp.arange(creature.alive.shape[1])[None, :] < n) & creature.alive
    transform = creature.transform
    transform = update(
        transform,
        pos=jnp.where(pinned[:, :, None], PINNED_POS, transform.pos),
        theta=jnp.where(pinned, PINNED_THETA, transform.theta),
        v=jnp.where(pinned, 0.0, transform.v),
    )
    creature = update(creature, transform=transform)
    states = update(states, entities=update(states.entities, creature=creature))
    if PIN_BUSH_STOCK:
        bush = states.entities.bush
        stock = jnp.where(bush.alive, 1.0, bush.edible.stock)
        bush = update(bush, edible=update(bush.edible, stock=stock))
        states = update(states, entities=update(states.entities, bush=bush))
    return states


def _scan_chunk(step_fn, states, cfg, size: int, collect: bool):
    batched = jax.vmap(lambda state: step_fn(state, cfg))

    def body(carry, _):
        current, totals, stats = carry
        current, buffer = batched(current)
        if STATIONARY:
            current = pin_stationary(current)
        totals = totals + jax.vmap(_event_counts)(buffer)
        creature = current.entities.creature
        alive = creature.alive
        mature = alive & (creature.body.mu >= 1.0)
        juvenile = alive & ~mature
        digested = R_DIG * jnp.sum(creature.body.G, axis=-1)
        stats = stats + jnp.stack(
            [
                jnp.sum(jnp.where(juvenile, digested, 0.0), axis=1),
                jnp.sum(jnp.where(mature, digested, 0.0), axis=1),
                jnp.sum(juvenile, axis=1).astype(jnp.float32),
                jnp.sum(mature, axis=1).astype(jnp.float32),
            ],
            axis=1,
        )
        if collect:
            return (current, totals, stats), (buffer, current.ledger)
        return (current, totals, stats), None

    initial = jnp.zeros((states.tick.shape[0], len(events_mod.EVENT_KINDS)), dtype=jnp.int32)
    zero_stats = jnp.zeros((states.tick.shape[0], 4), dtype=jnp.float32)
    if collect:
        (states, totals, maturity), (buffers, ledgers) = jax.lax.scan(
            body, (states, initial, zero_stats), None, length=size
        )
        return states, totals, buffers, ledgers, maturity
    (states, totals, _maturity), _ = jax.lax.scan(
        body, (states, initial, zero_stats), None, length=size
    )
    return states, totals, None, None, None


def _flatten(buffers, world: int) -> dict:
    counts = np.asarray(buffers.count)[:, world]
    capacity = buffers.kind.shape[2]
    mask = np.arange(capacity)[None, :] < counts[:, None]
    columns = {}
    for name in (
        "kind",
        "uid",
        "tick",
        "cause",
        "nutrient",
        "target_kind",
        "target_uid",
        "same_lineage",
        "reason",
        "loss",
        "parent_uid",
        "absorbed",
    ):
        columns[name] = np.asarray(getattr(buffers, name))[:, world][mask]
    return columns


def run_worlds(
    step_fn,
    cfg,
    seeds,
    ticks: int,
    collect: bool,
    chunk: int = CHUNK,
    track: bool = False,
    initial_hook=None,
) -> dict:
    states = _stack([initial_state(cfg, seed=seed) for seed in seeds])
    if initial_hook is not None:
        states = initial_hook(states, cfg)
    founder_uids = np.asarray(
        states.entities.creature.identity.uid[:, : int(cfg.world.founders)]
    ).tolist()
    worlds = [
        {
            "births": {},        # uid -> (tick, reseed: bool, parent_uid)
            "deaths": [],        # (tick, uid, cause, nutrient, same_lineage, loss[5])
            "bites": Counter(),
            "reseeds": 0,
            "reseed_ticks": [],
            "birth_fail": Counter(),
            "birth_fail_parents": {},
            "birth_fail_nutrients": Counter(),
            "births_by_parent": Counter(),
            "bite_targets": Counter(),
            "bites_living": Counter(),
            "meat_animal": Counter(),
            "generation": {},
        }
        for _ in seeds
    ]
    for world, data in enumerate(worlds):
        for uid in founder_uids[world]:
            data["generation"][int(uid)] = 0
    tracker = [[] for _ in seeds] if track else None
    dropped_events = 0
    maturity_totals = np.zeros((len(seeds), 4), dtype=np.float64)
    totals = np.zeros((len(seeds), len(events_mod.EVENT_KINDS)), dtype=np.int64)
    ledger = {
        "inflow": np.zeros((len(seeds), C.N_INFLOW_CATEGORIES, C.N_NUTRIENTS), dtype=np.float64),
        "outflow": np.zeros((len(seeds), C.N_OUTFLOW_CATEGORIES, C.N_NUTRIENTS), dtype=np.float64),
        "transfer_from": np.zeros((len(seeds), C.N_NUTRIENTS), dtype=np.float64),
        "transfer_to": np.zeros((len(seeds), C.N_NUTRIENTS), dtype=np.float64),
        "digested": np.zeros((len(seeds), C.N_NUTRIENTS), dtype=np.float64),
    }
    populations = []
    jitted = jax.jit(lambda current: _scan_chunk(step_fn, current, cfg, chunk, collect))
    started = time.perf_counter()
    for _ in range(ticks // chunk):
        states, counts, buffers, ledgers, maturity = jitted(states)
        totals += np.asarray(counts)
        maturity_totals += np.asarray(maturity, dtype=np.float64)
        dropped_events += int(np.asarray(buffers.dropped).sum())
        for name in ledger:
            ledger[name] += np.asarray(getattr(ledgers, name), dtype=np.float64).sum(axis=0)
        if collect:
            for world in range(len(seeds)):
                columns = _flatten(buffers, world)
                births = columns["kind"] == events_mod.kind_code("birth")
                reseed_birth = births & (columns["reason"] == events_mod.reason_code("reseed"))
                for tick, uid, reseed, parent in zip(
                    columns["tick"][births].tolist(),
                    columns["uid"][births].tolist(),
                    reseed_birth[births].tolist(),
                    columns["parent_uid"][births].tolist(),
                ):
                    worlds[world]["births"][int(uid)] = (int(tick), bool(reseed), int(parent))
                    if int(parent) >= 0:
                        worlds[world]["births_by_parent"][int(parent)] += 1
                        worlds[world]["generation"][int(uid)] = (
                            worlds[world]["generation"].get(int(parent), 0) + 1
                        )
                    else:
                        worlds[world]["generation"][int(uid)] = 0
                fails = columns["kind"] == events_mod.kind_code("birth_fail")
                for uid, reason, nutrient in zip(
                    columns["uid"][fails].tolist(),
                    columns["reason"][fails].tolist(),
                    columns["nutrient"][fails].tolist(),
                ):
                    name = events_mod.REASONS[int(reason)]
                    worlds[world]["birth_fail"][name] += 1
                    worlds[world]["birth_fail_parents"].setdefault(name, set()).add(int(uid))
                    if name == "no_reserve" and 0 <= int(nutrient) < 3:
                        worlds[world]["birth_fail_nutrients"][NUTRIENT_LABELS[int(nutrient)]] += 1
                deaths = columns["kind"] == events_mod.kind_code("death")
                for tick, uid, cause, nutrient, same, loss in zip(
                    columns["tick"][deaths].tolist(),
                    columns["uid"][deaths].tolist(),
                    columns["cause"][deaths].tolist(),
                    columns["nutrient"][deaths].tolist(),
                    columns["same_lineage"][deaths].tolist(),
                    columns["loss"][deaths].tolist(),
                ):
                    worlds[world]["deaths"].append(
                        (int(tick), int(uid), int(cause), int(nutrient), bool(same), tuple(loss))
                    )
                bites = (columns["kind"] == events_mod.kind_code("bite")) & (
                    columns["target_kind"] == C.BUSH
                )
                for uid, target in zip(
                    columns["uid"][bites].tolist(), columns["target_uid"][bites].tolist()
                ):
                    worlds[world]["bites"][int(uid)] += 1
                    worlds[world]["bite_targets"][int(target)] += 1
                living = (columns["kind"] == events_mod.kind_code("bite")) & (
                    (columns["target_kind"] == C.CREATURE)
                    | (columns["target_kind"] == C.CARCASS)
                )
                for uid, target_kind, absorbed in zip(
                    columns["uid"][living].tolist(),
                    columns["target_kind"][living].tolist(),
                    columns["absorbed"][living].tolist(),
                ):
                    if int(target_kind) == C.CREATURE:
                        worlds[world]["bites_living"][int(uid)] += 1
                    worlds[world]["meat_animal"][int(uid)] += float(
                        np.asarray(absorbed)[C.ANIMAL]
                    )
                reseed_events = columns["kind"] == events_mod.kind_code("reseed")
                worlds[world]["reseeds"] += int(np.sum(reseed_events))
                worlds[world]["reseed_ticks"].extend(columns["tick"][reseed_events].tolist())
        if track:
            for world in range(len(seeds)):
                wstate = _world(states, world)
                creat = wstate.entities.creature
                bush = wstate.entities.bush
                diet = np.asarray(diet_from_genome(creat.genome.g, cfg))
                nutrients = np.asarray(creat.body.N)
                ratio = nutrients / np.maximum(diet, 1e-6)
                deficit = diet - nutrients
                sum_n = np.asarray(jnp.sum(creat.body.N, axis=1))
                theta = np.asarray(reproduction_system.theta_repro(creat, cfg))
                mass = 0.3 + 0.7 * np.asarray(creat.body.mu)

                tracker[world].append(
                    {
                        "tick": int(states.tick[world]),
                        "uid": np.asarray(creat.identity.uid),
                        "alive": np.asarray(creat.alive),
                        "sum_n": sum_n,
                        "theta": theta,
                        "mass": mass,
                        "limiting": np.argmin(ratio, axis=1),
                        "limiting_absolute": np.argmax(deficit, axis=1),
                        "stock": np.asarray(bush.edible.stock),
                        "bush_alive": np.asarray(bush.alive),
                        "bush_kind": np.asarray(bush.edible.kind),
                        "pos": np.asarray(creat.transform.pos),
                        "rho_animal": diet[:, C.ANIMAL],
                    }
                )
        populations.append([int(population(_world(states, i))) for i in range(len(seeds))])
    wall = time.perf_counter() - started
    return {
        "wall_seconds": round(wall, 2),
        "populations": populations,
        "counts": totals,
        "ledger": ledger,
        "worlds": worlds,
        "states": states,
        "tracker": tracker,
        "dropped_events": dropped_events,
        "founder_uids": founder_uids,
        "maturity": maturity_totals,
        "ticks": ticks,
    }


def _lifespan_stats(ages: list[int]) -> dict:
    if not ages:
        return {"count": 0, "median": None, "matured": 0, "matured_median": None}
    matured = [age for age in ages if age >= A_MATURE]
    return {
        "count": len(ages),
        "median": float(np.median(ages)),
        "matured": len(matured),
        "matured_median": float(np.median(matured)) if matured else None,
    }


def summarize(cfg, result, seeds, cohort_len: int, chunk: int = CHUNK) -> dict:
    pop = np.asarray(result["populations"])
    capacity = int(cfg.capacities.creature)
    causes: Counter = Counter()
    nutrient_counts = [0, 0, 0]
    bite_deaths = {"same_lineage": 0, "other": 0}
    loss_by_cause: dict[str, list] = {}
    ages_reproduction: list[int] = []
    ages_reseed: list[int] = []
    per_world = []
    cohorts: dict[int, dict] = {}
    for world, data in enumerate(result["worlds"]):
        for tick, uid, cause, nutrient, same, loss in data["deaths"]:
            name = events_mod.DEATH_CAUSES[cause]
            causes[name] += 1
            loss_by_cause.setdefault(name, []).append(loss)
            if cause == events_mod.cause_code("nutrition") and 0 <= nutrient < C.N_NUTRIENTS:
                nutrient_counts[int(nutrient)] += 1
            if cause == events_mod.cause_code("bite"):
                bite_deaths["same_lineage" if same else "other"] += 1
            born = data["births"].get(uid)
            if born is not None:
                born_tick, reseed_born, _parent = born
                age = tick - born_tick
                (ages_reseed if reseed_born else ages_reproduction).append(age)
                if not reseed_born:
                    key = int(born_tick // cohort_len)
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
        for uid, (born_tick, reseed_born, _parent) in data["births"].items():
            if reseed_born:
                continue
            key = int(born_tick // cohort_len)
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
            bites = data["bites"].get(uid, 0)
            entry["bites"] += bites
            if bites:
                entry["biters"] += 1
        born_groups = Counter(reseed for _, reseed, _parent in data["births"].values())
        per_world.append(
            {
                "seed": seeds[world],
                "final_population": int(pop[-1, world]),
                "min_population": int(pop[:, world].min()),
                "max_population": int(pop[:, world].max()),
                "births_reproduction": int(born_groups.get(False, 0)),
                "births_reseed": int(born_groups.get(True, 0)),
                "deaths": len(data["deaths"]),
                "reseeds": data["reseeds"],
            }
        )
    cohort_rows = []
    for key in sorted(cohorts):
        entry = cohorts[key]
        born = max(entry["born"], 1)
        ticks = entry["lifetime_ticks"]
        cohort_rows.append(
            {
                "cohort": key,
                "born": entry["born"],
                "died": entry["died"],
                "biters": entry["biters"],
                "bites_per_individual": entry["bites"] / born,
                "fraction_biters": entry["biters"] / born,
                "bites_per_lifetime_tick": (entry["bites"] / ticks) if ticks else None,
                "mean_lifespan": float(np.mean(entry["lifespans"])) if entry["lifespans"] else None,
            }
        )
    loss_summary = {
        name: [float(np.mean([row[index] for row in rows])) for index in range(C.N_DAMAGE_SOURCES)]
        for name, rows in loss_by_cause.items()
    }
    event_counts = {
        name: int(value)
        for name, value in zip(events_mod.EVENT_KINDS, np.asarray(result["counts"]).sum(axis=0).tolist())
        if value
    }
    ledger_totals = {
        "inflow": np.asarray(result["ledger"]["inflow"]).sum(axis=0).tolist(),
        "outflow": np.asarray(result["ledger"]["outflow"]).sum(axis=0).tolist(),
        "transfer_from": np.asarray(result["ledger"]["transfer_from"]).sum(axis=0).tolist(),
        "transfer_to": np.asarray(result["ledger"]["transfer_to"]).sum(axis=0).tolist(),
        "digested": np.asarray(result["ledger"]["digested"]).sum(axis=0).tolist(),
    }
    birth_fail_counts: Counter = Counter()
    birth_fail_parents: dict = {}
    birth_fail_nutrients: Counter = Counter()
    births_by_parent: dict = {}
    births_total = 0
    for world, data in enumerate(result["worlds"]):
        birth_fail_counts.update(data["birth_fail"])
        birth_fail_nutrients.update(data["birth_fail_nutrients"])
        for reason, parents in data["birth_fail_parents"].items():
            birth_fail_parents.setdefault(reason, set()).update(
                (world, int(parent)) for parent in parents
            )
        for parent, count in data["births_by_parent"].items():
            births_by_parent[(world, int(parent))] = int(count)
            births_total += int(count)
    n_worlds = len(result["worlds"])
    founders_total = int(cfg.world.founders) * n_worlds
    individuals = founders_total + births_total
    founder_counts, repro_counts = [], []
    first_birth_ages, founder_first, repro_first = [], [], []
    for (world, parent), count in births_by_parent.items():
        data = result["worlds"][world]
        first_tick = min(
            tick for uid, (tick, _reseed, pid) in data["births"].items() if pid == parent
        )
        born = data["births"].get(parent)
        if born is None:
            founder_counts.append(count)
            founder_first.append(first_tick)
        else:
            repro_counts.append(count)
            repro_first.append(first_tick - born[0])
    first_birth_ages = founder_first + repro_first

    def _group_stats(counts, individuals_in_group):
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

    r0 = {
        "births_total": births_total,
        "individuals_ever": individuals,
        "founders": _group_stats(founder_counts, founders_total),
        "reproduction_born": _group_stats(repro_counts, births_total),
        "mean_r0_all_individuals": births_total / max(individuals, 1),
        "completed_lifetimes": {},
    }
    dead_founders = dead_repro = 0
    completed_births = {"founders": 0, "reproduction_born": 0}
    for world, data in enumerate(result["worlds"]):
        dead_uids = {row[1] for row in data["deaths"]}
        for row in data["deaths"]:
            if row[1] in data["births"]:
                dead_repro += 1
            else:
                dead_founders += 1
        for parent, count in data["births_by_parent"].items():
            if int(parent) in dead_uids:
                key = "reproduction_born" if parent in data["births"] else "founders"
                completed_births[key] += int(count)

    def _completed(dead, births):
        return {
            "dead_individuals": int(dead),
            "births": int(births),
            "mean_r0": (births / dead) if dead else None,
        }

    r0["completed_lifetimes"] = {
        "founders": _completed(dead_founders, completed_births["founders"]),
        "reproduction_born": _completed(dead_repro, completed_births["reproduction_born"]),
        "all": _completed(
            dead_founders + dead_repro,
            completed_births["founders"] + completed_births["reproduction_born"],
        ),
        "note": "only individuals with a death record inside the run; alive-at-end excluded",
    }
    r0["reproducing_parents"] = len(founder_counts) + len(repro_counts)
    r0["mean_r0_reproducing"] = (
        float(np.mean(founder_counts + repro_counts))
        if founder_counts or repro_counts
        else None
    )
    r0["max_r0"] = (
        int(max(founder_counts + repro_counts)) if founder_counts or repro_counts else 0
    )
    r0["fraction_reproducing_with_2plus"] = (
        float(np.mean([value >= 2 for value in founder_counts + repro_counts]))
        if founder_counts or repro_counts
        else None
    )
    r0["fraction_reproducing_with_3plus"] = (
        float(np.mean([value >= 3 for value in founder_counts + repro_counts]))
        if founder_counts or repro_counts
        else None
    )
    ticks = int(result.get("ticks", 0))
    cutoff = ticks - 2000  # age cap = 5 * a_sen
    trunc_n = {"founders": 0, "reproduction_born": 0}
    trunc_births = {"founders": 0, "reproduction_born": 0}
    if cutoff >= 0:
        for world, data in enumerate(result["worlds"]):
            trunc_n["founders"] += int(cfg.world.founders)
            for _uid, (tick, reseed, _parent) in data["births"].items():
                if not reseed and tick <= cutoff:
                    trunc_n["reproduction_born"] += 1
            for parent, count in data["births_by_parent"].items():
                born = data["births"].get(parent)
                born_tick = 0 if born is None else int(born[0])
                if born_tick <= cutoff:
                    key = "reproduction_born" if born is not None else "founders"
                    trunc_births[key] += int(count)

    def _trunc(group):
        return {
            "individuals": trunc_n[group],
            "births": trunc_births[group],
            "mean_r0": (
                trunc_births[group] / trunc_n[group] if trunc_n[group] else None
            ),
        }

    r0["cohort_truncated"] = {
        "cutoff_tick": cutoff,
        "rule": "only individuals born at or before tick_end - 2000 (age cap); all such lives are complete",
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
    track_summary = None
    if result.get("tracker"):
        max_ratio, ever_above, above_fractions, limiting_counts = [], 0, [], Counter()
        absolute_limiting = Counter()
        stock_means, low_stock = [], []
        for samples in result["tracker"]:
            stats = {}
            for sample in samples:
                alive = sample["alive"]
                stock = sample["stock"][sample["bush_alive"]]
                if stock.size:
                    stock_means.append(float(np.mean(stock)))
                    low_stock.append(float(np.mean(stock < 0.1)))
                for index in np.nonzero(alive)[0]:
                    uid = int(sample["uid"][index])
                    ratio = float(sample["sum_n"][index] / max(float(sample["theta"][index]), 1e-6))
                    entry = stats.setdefault(
                        uid,
                        {
                            "max": 0.0,
                            "above": 0,
                            "n": 0,
                            "lim": Counter(),
                            "abs_lim": Counter(),
                        },
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
        track_summary = {
            "individuals_tracked": len(max_ratio),
            "median_max_sumN_over_theta": float(np.median(max_ratio)) if max_ratio else None,
            "fraction_ever_above_theta": ever_above / max(len(max_ratio), 1),
            "mean_fraction_time_above_theta": (
                float(np.mean(above_fractions)) if above_fractions else None
            ),
            "limiting_nutrient_most_common": {
                NUTRIENT_LABELS[key]: value for key, value in limiting_counts.items()
            },
            "limiting_nutrient_absolute": {
                NUTRIENT_LABELS[key]: value for key, value in absolute_limiting.items()
            },
            "bush_stock_mean": float(np.mean(stock_means)) if stock_means else None,
            "bush_stock_below_0_1_fraction": float(np.mean(low_stock)) if low_stock else None,
        }
    individuals_export = None
    if STATIONARY and STATIONARY_META is not None:
        individuals_export = []
        for world, meta in enumerate(STATIONARY_META):
            data = result["worlds"][world]
            uids = result["founder_uids"][world]
            for index in range(len(meta["rho"])):
                uid = int(uids[index])
                birth_ticks = sorted(
                    tick for _uid, (tick, _reseed, parent) in data["births"].items() if parent == uid
                )
                death = next((row for row in data["deaths"] if row[1] == uid), None)
                individuals_export.append(
                    {
                        "world": world,
                        "uid": uid,
                        "rho": [float(value) for value in meta["rho"][index]],
                        "theta": float(meta["theta"][index]),
                        "species": int(meta["species"][index]),
                        "temperature_geo_shade": float(meta["temperature_geo_shade"][index]),
                        "births": len(birth_ticks),
                        "birth_ages": [int(tick) for tick in birth_ticks],
                        "death_age": int(death[0]) if death is not None else None,
                        "death_cause": (
                            events_mod.DEATH_CAUSES[death[2]] if death is not None else None
                        ),
                        "death_nutrient": (
                            NUTRIENT_LABELS[int(death[3])]
                            if death is not None and 0 <= int(death[3]) < 3
                            else None
                        ),
                    }
                )
    individual_ticks = float(np.sum(pop)) * chunk
    harvest = float(np.asarray(result["ledger"]["inflow"])[:, C.INFLOW_BITES, :].sum())
    demand = float(np.asarray(result["ledger"]["outflow"])[:, C.OUTFLOW_METABOLISM, :].sum())
    maturity = np.asarray(result["maturity"], dtype=np.float64).sum(axis=0)
    juvenile_ticks, mature_ticks = float(maturity[2]), float(maturity[3])
    intake_by_maturity = {
        "juvenile": {
            "individual_ticks": juvenile_ticks,
            "digested_mass": float(maturity[0]),
            "digested_per_individual_tick": (
                float(maturity[0]) / juvenile_ticks if juvenile_ticks else None
            ),
        },
        "mature": {
            "individual_ticks": mature_ticks,
            "digested_mass": float(maturity[1]),
            "digested_per_individual_tick": (
                float(maturity[1]) / mature_ticks if mature_ticks else None
            ),
        },
        "note": "digested = R_DIG * sum(G) accumulated per tick; equals intake over a lifetime",
    }
    species_labels = {0: "cold", 1: "temperate", 2: "nut"}
    stock_by_species = {}
    end_rho = []
    end_gen = Counter()
    if result.get("tracker"):
        for kind_id, label in species_labels.items():
            means, lows = [], []
            for samples in result["tracker"]:
                for sample in samples:
                    mask = sample["bush_alive"] & (sample["bush_kind"] == kind_id)
                    if np.any(mask):
                        values = sample["stock"][mask]
                        means.append(float(np.mean(values)))
                        lows.append(float(np.mean(values < 0.1)))
            stock_by_species[label] = {
                "mean": float(np.mean(means)) if means else None,
                "below_0_1_fraction": float(np.mean(lows)) if lows else None,
            }
        for world, samples in enumerate(result["tracker"]):
            if not samples:
                continue
            last = samples[-1]
            end_rho.extend(
                float(value) for value in last["rho_animal"][last["alive"]]
            )
            generation = result["worlds"][world]["generation"]
            for uid in last["uid"][last["alive"]].tolist():
                end_gen[generation.get(int(uid), -1)] += 1
    generation_summary = {
        "individuals_by_generation": {},
        "births_by_generation": {},
        "r0_by_generation": {},
        "end_population_by_generation": dict(end_gen),
    }
    gen_individuals, gen_births = Counter(), Counter()
    for data in result["worlds"]:
        for generation in data["generation"].values():
            gen_individuals[int(generation)] += 1
        for parent, count in data["births_by_parent"].items():
            gen = data["generation"].get(int(parent), 0)
            gen_births[int(gen)] += int(count)
    for generation in sorted(set(gen_individuals) | set(gen_births)):
        individuals = gen_individuals.get(generation, 0)
        births = gen_births.get(generation, 0)
        generation_summary["individuals_by_generation"][str(generation)] = individuals
        generation_summary["births_by_generation"][str(generation)] = births
        generation_summary["r0_by_generation"][str(generation)] = (
            births / individuals if individuals else None
        )
    bites_by_species = {"cold": 0, "temperate": 0, "nut": 0, "unknown": 0}
    states = result.get("states")
    if states is not None:
        bush_uid = np.asarray(states.entities.bush.uid)
        bush_kind = np.asarray(states.entities.bush.edible.kind)
        for world, data in enumerate(result["worlds"]):
            mapping = {
                int(uid): species_labels.get(int(kind), "unknown")
                for uid, kind in zip(bush_uid[world].tolist(), bush_kind[world].tolist())
            }
            for target, count in data["bite_targets"].items():
                bites_by_species[mapping.get(int(target), "unknown")] += int(count)
    rho_stats = {}
    if end_rho:
        values = np.asarray(end_rho, dtype=np.float64)
        rho_stats = {
            "individuals": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "fraction_below_0_2": float(np.mean(values < 0.2)),
            "fraction_below_0_34": float(np.mean(values < 0.34)),
        }
    budget_buckets: dict[int, dict] = {}
    for data in result["worlds"]:
        for tick in data["reseed_ticks"]:
            bucket = int(tick) // 1000
            entry = budget_buckets.setdefault(bucket, {"rescues": 0, "births": 0, "deaths": 0})
            entry["rescues"] += 1
        for _uid, (tick, reseed, _parent) in data["births"].items():
            if not reseed:
                bucket = int(tick) // 1000
                budget_buckets.setdefault(bucket, {"rescues": 0, "births": 0, "deaths": 0})
                budget_buckets[bucket]["births"] += 1
        for row in data["deaths"]:
            bucket = int(row[0]) // 1000
            budget_buckets.setdefault(bucket, {"rescues": 0, "births": 0, "deaths": 0})
            budget_buckets[bucket]["deaths"] += 1
    per_1000 = [
        {"start_tick": bucket * 1000, **budget_buckets[bucket]}
        for bucket in sorted(budget_buckets)
    ]
    post_ramp = []
    ramp_ticks = int(cfg.world.ramp_ticks)
    for world, data in enumerate(result["worlds"]):
        post_rescues = sum(1 for tick in data["reseed_ticks"] if tick >= ramp_ticks)
        post_ramp.append(
            {
                "world": world,
                "seed": int(seeds[world]),
                "post_ramp_rescues": int(post_rescues),
                "final_population": int(pop[-1, world]),
            }
        )
    generation_bites = {}
    for data in result["worlds"]:
        for uid, gen in data["generation"].items():
            key = str(int(gen))
            entry = generation_bites.setdefault(key, {"bites_living": 0, "meat_animal": 0.0})
            entry["bites_living"] += int(data["bites_living"].get(int(uid), 0))
            entry["meat_animal"] += float(data["meat_animal"].get(int(uid), 0.0))
    return {
        "per_1000": per_1000,
        "post_ramp": post_ramp,
        "generation_bites": generation_bites,
        "stock_by_species": stock_by_species,
        "end_population_rho_animal": rho_stats,
        "generation": generation_summary,
        "bites_by_species": bites_by_species,
        "intake_per_individual_tick": harvest / max(individual_ticks, 1.0),
        "demand_per_individual_tick": demand / max(individual_ticks, 1.0),
        "individual_ticks": individual_ticks,
        "intake_by_maturity": intake_by_maturity,
        "individuals_export": individuals_export,
        "r0": r0,
        "first_birth_age": {
            "count": len(first_birth_ages),
            "median": float(np.median(first_birth_ages)) if first_birth_ages else None,
            "founders_median": float(np.median(founder_first)) if founder_first else None,
            "reproduction_median": float(np.median(repro_first)) if repro_first else None,
        },
        "birth_fail": {
            "counts": dict(birth_fail_counts),
            "unique_parents": {key: len(value) for key, value in birth_fail_parents.items()},
            "no_reserve_nutrients": dict(birth_fail_nutrients),
        },
        "track": track_summary,
        "dropped_events": int(result.get("dropped_events", 0)),
        "event_counts": event_counts,
        "ledger_totals": ledger_totals,
        "population_rows": pop.tolist(),
        "per_world": per_world,
        "population_final": pop[-1].tolist(),
        "population_min": pop.min(axis=0).tolist(),
        "population_mean": float(pop.mean()),
        "capacity_fraction": float(np.mean(pop >= capacity)),
        "births_reproduction": int(sum(w["births_reproduction"] for w in per_world)),
        "births_reseed": int(sum(w["births_reseed"] for w in per_world)),
        "deaths": int(sum(len(data["deaths"]) for data in result["worlds"])),
        "reseeds": int(sum(data["reseeds"] for data in result["worlds"])),
        "reseeds_after_ramp": int(
            sum(
                1
                for data in result["worlds"]
                for tick in data["reseed_ticks"]
                if tick >= int(cfg.world.ramp_ticks)
            )
        ),
        "death_causes": dict(causes),
        "death_loss_mean": loss_summary,
        "nutrition_types": {
            "plant": nutrient_counts[C.PLANT],
            "animal": nutrient_counts[C.ANIMAL],
            "toxin": nutrient_counts[C.TOXIN],
        },
        "bite_deaths": bite_deaths,
        "lifespan_reproduction": _lifespan_stats(ages_reproduction),
        "lifespan_reseed": _lifespan_stats(ages_reseed),
        "cohorts_reproduction": cohort_rows,
    }


def git_provenance() -> dict:
    """Commit and dirty flag for the report (provenance)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


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
    if args.bush_count is not None:
        cfg = cfg.replaced(capacities=replace(cfg.capacities, bush=int(args.bush_count)))
    if args.bush_regen_rate is not None:
        cfg = cfg.replaced(
            world=replace(cfg.world, bush_regen_rate=float(args.bush_regen_rate))
        )
    if args.nut_regen_multiplier is not None:
        cfg = cfg.replaced(
            world=replace(
                cfg.world, bush_regen_nut_multiplier=float(args.nut_regen_multiplier)
            )
        )
    if args.founder_logit_std is not None:
        cfg = cfg.replaced(
            body=replace(cfg.body, founder_logit_std=float(args.founder_logit_std))
        )
    if args.reseed_until_tick >= 0:
        cfg = cfg.replaced(
            world=replace(cfg.world, reseed_until_tick=int(args.reseed_until_tick))
        )
    if args.ramp_metabolism is not None:
        cfg = cfg.replaced(
            world=replace(cfg.world, ramp_metabolism_multiplier=float(args.ramp_metabolism))
        )
    if args.temperature_scale is not None:
        cfg = cfg.replaced(
            world=replace(
                cfg.world,
                temperature_scale=float(args.temperature_scale),
                ramp_ticks=int(args.ramp_ticks),
            )
        )
    if args.base_metabolism is not None:
        cfg = cfg.replaced(body=replace(cfg.body, base_metabolism=float(args.base_metabolism)))
    if args.tissue_from_b:
        cfg = cfg.replaced(
            reproduction=replace(cfg.reproduction, tissue_from_reserves=False)
        )
    if args.founders is not None:
        cfg = cfg.replaced(world=replace(cfg.world, founders=int(args.founders)))
    if args.no_reseed:
        cfg = cfg.replaced(world=replace(cfg.world, reseed_enabled=False))
    if args.founder_animal_logit is not None:
        cfg = cfg.replaced(
            body=replace(
                cfg.body, founder_animal_logit_mean=float(args.founder_animal_logit)
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
    parser.add_argument("--knob-mode", choices=("constant", "true-b"), default="true-b")
    parser.add_argument("--avoid-toxin", action="store_true")
    parser.add_argument("--min-stock", type=float, default=0.0)
    parser.add_argument("--avoid-hot", type=float, default=0.0)
    parser.add_argument("--founder-animal-logit", type=float, default=None)
    parser.add_argument("--no-reseed", action="store_true")
    parser.add_argument("--base-metabolism", type=float, default=None)
    parser.add_argument("--founders", type=int, default=None)
    parser.add_argument("--tissue-from-b", action="store_true")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--stationary", action="store_true")
    parser.add_argument("--pin-bush-stock", action="store_true")
    parser.add_argument("--skip-bite-rate", type=float, default=0.0)
    parser.add_argument("--hunger-threshold", type=float, default=0.8)
    parser.add_argument("--stock-min", type=float, default=0.05)
    parser.add_argument("--ramp-metabolism", type=float, default=None)
    parser.add_argument("--bush-count", type=int, default=None)
    parser.add_argument("--bush-regen-rate", type=float, default=None)
    parser.add_argument("--nut-regen-multiplier", type=float, default=None)
    parser.add_argument("--founder-logit-std", type=float, default=None)
    parser.add_argument("--reseed-until-tick", type=int, default=-1)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    global KNOB, KNOB_MODE, AVOID_TOXIN, MIN_STOCK, AVOID_HOT
    global STATIONARY, PIN_BUSH_STOCK, SKIP_BITE_RATE, HUNGER_THRESHOLD, STOCK_MIN
    HUNGER_THRESHOLD = float(args.hunger_threshold)
    STOCK_MIN = float(args.stock_min)
    KNOB = float(args.knob)
    KNOB_MODE = args.knob_mode
    AVOID_TOXIN = bool(args.avoid_toxin)
    MIN_STOCK = float(args.min_stock)
    AVOID_HOT = float(args.avoid_hot)
    STATIONARY = bool(args.stationary)
    PIN_BUSH_STOCK = bool(args.pin_bush_stock)
    SKIP_BITE_RATE = float(args.skip_bite_rate)
    cfg = design_config(args)
    seeds = [9300 + i for i in range(args.worlds)]
    step_fn = forager_step if args.arm == "forager" else engine_step
    hook = stationary_setup if STATIONARY else None
    result = run_worlds(
        step_fn,
        cfg,
        seeds,
        args.ticks,
        collect=True,
        chunk=args.chunk,
        track=args.track,
        initial_hook=hook,
    )
    summary = summarize(cfg, result, seeds, args.cohort, chunk=args.chunk)
    summary["arm"] = args.arm
    summary["probe"] = {
        "knob_mode": KNOB_MODE,
        "bite_force": BITE_FORCE,
        "avoid_toxin": AVOID_TOXIN,
        "min_stock": MIN_STOCK,
        "avoid_hot": AVOID_HOT,
        "stationary": STATIONARY,
        "pin_bush_stock": PIN_BUSH_STOCK,
        "skip_bite_rate": SKIP_BITE_RATE,
        "hunger_threshold": HUNGER_THRESHOLD,
        "stock_min": STOCK_MIN,
    }
    summary["founder_animal_logit_mean"] = cfg.body.founder_animal_logit_mean
    summary["design"] = {
        "base_metabolism": cfg.body.base_metabolism,
        "bush_capacity": cfg.capacities.bush,
        "bush_regen_rate": cfg.world.bush_regen_rate,
        "bush_regen_base": cfg.world.bush_regen_base,
        "bush_regen_nut_multiplier": cfg.world.bush_regen_nut_multiplier,
        "temperature_scale": cfg.world.temperature_scale,
        "founder_reserves": cfg.body.founder_reserves,
        "founder_animal_logit_mean": cfg.body.founder_animal_logit_mean,
        "founder_logit_std": cfg.body.founder_logit_std,
        "ramp_ticks": cfg.world.ramp_ticks,
        "ramp_metabolism_multiplier": cfg.world.ramp_metabolism_multiplier,
        "ramp_berry_multiplier": cfg.world.ramp_berry_multiplier,
        "ramp_temperature_multiplier": cfg.world.ramp_temperature_multiplier,
        "reseed_until_tick": cfg.world.reseed_until_tick,
        "chunk": args.chunk,
    }
    summary["seeds"] = seeds
    summary["ticks"] = args.ticks
    summary["provenance"] = git_provenance()
    summary["wall_seconds"] = result["wall_seconds"]
    out = Path(args.out or ("runs/diagnostics/" + args.arm + ".json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    keys = (
        "arm",
        "births_reproduction",
        "births_reseed",
        "deaths",
        "reseeds",
        "population_mean",
        "r0",
        "birth_fail",
        "first_birth_age",
        "track",
        "dropped_events",
        "death_causes",
        "death_loss_mean",
        "lifespan_reproduction",
        "lifespan_reseed",
    )
    print(json.dumps({key: summary[key] for key in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
