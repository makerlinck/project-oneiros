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
from life_sandbox.core import ramp as ramp_mod
from life_sandbox.core.state import population
from life_sandbox.core.step import SYSTEMS, _event_counts, step as engine_step
from life_sandbox.core.systems import reproduction as reproduction_system
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


def _prefer(mask, distance):
    """Zero out non-preferred bushes, falling back to the full distance row."""
    masked = jnp.where(mask, distance, jnp.inf)
    any_preferred = jnp.any(mask, axis=1, keepdims=True)
    return jnp.where(any_preferred, masked, distance)


def forager_think(state, cfg):
    """Scripted competent forager (calibration tool; reads true body state)."""
    creature = state.entities.creature
    bush = state.entities.bush
    pos = creature.transform.pos
    theta = creature.transform.theta
    self_radius = creature.shape.radius
    n_creature = pos.shape[0]

    # Probe v2 (2026-11-06 review): a species is eligible only if its animal
    # supply covers the animal demand with a 1.3x margin at full digestion
    # (0.0135/tick * absorption 0.9*sqrt(rho_A) * c_A >= 1.3*rho_A).
    diet = diet_from_genome(creature.genome.g, cfg)  # (N, 3)
    rho_animal = diet[:, C.ANIMAL]
    animal_share = state.comp[C.ANIMAL][bush.edible.kind]  # per bush slot
    # Per-tick metabolic demand (ramp-scaled base metabolism): the animal intake
    # at full digestion must cover 1.3x of the animal share of that demand.
    c_tot = cfg.body.base_metabolism * ramp_mod.metabolism_multiplier(state.tick, cfg)
    supplied = 0.0135 * 0.9 * jnp.sqrt(rho_animal)[:, None] * animal_share[None, :]
    eligible = bush.alive[None, :] & (supplied >= 1.3 * rho_animal[:, None] * c_tot)

    delta = bush.transform.pos[None, :, :] - pos[:, None, :]
    distance0 = jnp.where(bush.alive[None, :], jnp.linalg.norm(delta, axis=-1), jnp.inf)
    stocked = bush.edible.stock >= 0.3

    def _pick(candidates, distance0, stocked):
        base = jnp.where(candidates, distance0, jnp.inf)
        stocked_base = jnp.where(candidates & stocked[None, :], distance0, jnp.inf)
        picked = jnp.where(
            jnp.any(jnp.isfinite(stocked_base), axis=1, keepdims=True), stocked_base, base
        )
        return jnp.where(jnp.any(jnp.isfinite(picked), axis=1, keepdims=True), picked, distance0)

    if STATIONARY:
        pick = distance0
    else:
        pick = _pick(eligible, distance0, stocked)
    slot0 = jnp.argmin(pick, axis=1)
    toxic_share = state.comp[C.TOXIN][bush.edible.kind]
    own_toxic = toxic_share[bush.edible.kind[slot0]] > 0.05
    needs_toxin = creature.body.N[:, C.TOXIN] >= 0.5 * diet[:, C.TOXIN]
    switch = own_toxic & needs_toxin
    non_toxic = eligible & (toxic_share[None, :] <= 0.05)
    alternative = _pick(non_toxic, distance0, stocked)
    if not STATIONARY:
        pick = jnp.where(switch[:, None], alternative, pick)
    distance = jnp.where(jnp.isfinite(pick), pick, 1e6)
    bush_slot = jnp.argmin(distance, axis=1)
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
        current, totals = carry
        current, buffer = batched(current)
        if STATIONARY:
            current = pin_stationary(current)
        totals = totals + jax.vmap(_event_counts)(buffer)
        if collect:
            return (current, totals), (buffer, current.ledger)
        return (current, totals), None

    initial = jnp.zeros((states.tick.shape[0], len(events_mod.EVENT_KINDS)), dtype=jnp.int32)
    if collect:
        (states, totals), (buffers, ledgers) = jax.lax.scan(
            body, (states, initial), None, length=size
        )
        return states, totals, buffers, ledgers
    (states, totals), _ = jax.lax.scan(body, (states, initial), None, length=size)
    return states, totals, None, None


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
        "same_lineage",
        "reason",
        "loss",
        "parent_uid",
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
            "birth_fail": Counter(),
            "birth_fail_parents": {},
            "birth_fail_nutrients": Counter(),
            "births_by_parent": Counter(),
        }
        for _ in seeds
    ]
    tracker = [[] for _ in seeds] if track else None
    dropped_events = 0
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
        states, counts, buffers, ledgers = jitted(states)
        totals += np.asarray(counts)
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
                for uid in columns["uid"][bites].tolist():
                    worlds[world]["bites"][int(uid)] += 1
                worlds[world]["reseeds"] += int(
                    np.sum(columns["kind"] == events_mod.kind_code("reseed"))
                )
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


def summarize(cfg, result, seeds, cohort_len: int) -> dict:
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
        "reproducing_parents": len(founder_counts) + len(repro_counts),
        "mean_r0_reproducing": (
            float(np.mean(founder_counts + repro_counts))
            if founder_counts or repro_counts
            else None
        ),
        "max_r0": int(max(founder_counts + repro_counts)) if founder_counts or repro_counts else 0,
        "fraction_reproducing_with_2plus": (
            float(np.mean([value >= 2 for value in founder_counts + repro_counts]))
            if founder_counts or repro_counts
            else None
        ),
        "fraction_reproducing_with_3plus": (
            float(np.mean([value >= 3 for value in founder_counts + repro_counts]))
            if founder_counts or repro_counts
            else None
        ),
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
    individual_ticks = float(np.sum(pop)) * CHUNK
    harvest = float(np.asarray(result["ledger"]["inflow"])[:, C.INFLOW_BITES, :].sum())
    demand = float(np.asarray(result["ledger"]["outflow"])[:, C.OUTFLOW_METABOLISM, :].sum())
    return {
        "intake_per_individual_tick": harvest / max(individual_ticks, 1.0),
        "demand_per_individual_tick": demand / max(individual_ticks, 1.0),
        "individual_ticks": individual_ticks,
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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    global KNOB, KNOB_MODE, AVOID_TOXIN, MIN_STOCK, AVOID_HOT
    global STATIONARY, PIN_BUSH_STOCK, SKIP_BITE_RATE, HUNGER_THRESHOLD
    HUNGER_THRESHOLD = float(args.hunger_threshold)
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
        step_fn, cfg, seeds, args.ticks, collect=True, track=args.track, initial_hook=hook
    )
    summary = summarize(cfg, result, seeds, args.cohort)
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
    }
    summary["founder_animal_logit_mean"] = cfg.body.founder_animal_logit_mean
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
