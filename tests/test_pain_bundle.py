"""Pain bundle tests (SANDBOX.md §5.6, §13 test_pain_bundle).

The nine carriers are evolved through sensing.pain_step (the production code)
inside a jax.lax.scan, so the tests exercise the real recurrence instead of a
reimplementation.  Checks: every resonant source peaks at its documented period,
amplitude scales with the source strength, zero strength leaves only the 0.02
base noise, the AR(2) gain gives unit stationary variance, and the shared
weights mix the carriers into the four channels.

The period estimate averages periodograms over eight independent carriers (one
per creature) and a long run, because the heavily damped sources have broad
spectral peaks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from life_sandbox.config import Capacities, Config, RunConfig, WorldConfig
from life_sandbox.core import rng
from life_sandbox.core.components import PAIN_SOURCES
from life_sandbox.core.state import empty
from life_sandbox.core.struct import update
from life_sandbox.core.systems import sensing

LONG_TICKS = 60_000
SHORT_TICKS = 6_000
BASE_KEY = rng.world_key(11)

EXPECTED_PERIODS = {
    "toxin": 5.0,
    "distension": 8.0,
    "heat": 12.0,
    "cold": 20.0,
    "hunger_plant": 40.0,
    "hunger_animal": 70.0,
    "hunger_toxin": 110.0,
    "sleepiness": 30.0,
}


def make_cfg(creatures: int = 1) -> Config:
    return Config.default().replaced(
        world=WorldConfig(width=16.0, height=16.0, cell=1.0),
        capacities=Capacities(creature=creatures, bush=1, carcass=1, neutral=1),
        run=RunConfig(seed=11, ticks=0),
    )


def base_state(cfg: Config):
    state = empty(cfg, rng.world_key(11))
    arena = state.entities.creature
    for slot in range(cfg.capacities.creature):
        arena = update(
            arena,
            alive=arena.alive.at[slot].set(True),
            transform=update(
                arena.transform,
                pos=arena.transform.pos.at[slot].set(
                    jnp.asarray([2.0 + slot, 8.0], jnp.float32)
                ),
            ),
            shape=update(arena.shape, radius=arena.shape.radius.at[slot].set(0.5)),
        )
    state = update(state, entities=update(state.entities, creature=arena))
    return update(state, pain_weights=jnp.zeros((9, 4), jnp.float32))


def set_body(state, slot=0, **values):
    arena = state.entities.creature
    body = arena.body
    for name, value in values.items():
        body = update(body, **{name: getattr(body, name).at[slot].set(value)})
    return update(state, entities=update(state.entities, creature=update(arena, body=body)))


def set_acute(state, a_inj, slot=0):
    arena = state.entities.creature
    pain = update(arena.pain, a_inj=arena.pain.a_inj.at[slot].set(a_inj))
    return update(state, entities=update(state.entities, creature=update(arena, pain=pain)))


def set_weights(state, weights):
    return update(state, pain_weights=jnp.asarray(weights, jnp.float32))


def run_bundle(state, cfg: Config, ticks: int):
    """Scan pain_step; returns signal [T, N, 4] and carriers [T, N, 9]."""
    x0 = state.entities.creature.pain.x

    def body_fn(carry_x, tick):
        current = update(
            state,
            entities=update(
                state.entities,
                creature=update(
                    state.entities.creature,
                    pain=update(state.entities.creature.pain, x=carry_x),
                ),
            ),
        )
        key = rng.system_key(BASE_KEY, tick, "sense")
        signal, new_x, _ = sensing.pain_step(current, cfg, key)
        return new_x, (signal, new_x[:, :, 0])

    _, (signals, carriers) = jax.lax.scan(body_fn, x0, jnp.arange(ticks, dtype=jnp.int32))
    return np.asarray(signals), np.asarray(carriers)


def estimate_period(carriers: np.ndarray, index: int) -> float:
    """Average the periodograms of all creatures, then take the peak bin."""
    ticks = carriers.shape[0]
    window = np.hanning(ticks)
    frequencies = np.fft.rfftfreq(ticks, d=1.0)
    power = np.zeros_like(frequencies)
    for creature in range(carriers.shape[1]):
        series = carriers[:, creature, index] - carriers[:, creature, index].mean()
        power += np.abs(np.fft.rfft(series * window)) ** 2
    power[0] = 0.0
    return float(1.0 / frequencies[int(np.argmax(power))])


def test_pain_sources_have_the_documented_order():
    assert PAIN_SOURCES == (
        "acute",
        "toxin",
        "distension",
        "heat",
        "cold",
        "hunger_plant",
        "hunger_animal",
        "hunger_toxin",
        "sleepiness",
    )
    assert sensing.PAIN_PERIODS[0] is None
    assert sensing.PAIN_PERIODS[1:] == tuple(EXPECTED_PERIODS[name] for name in PAIN_SOURCES[1:])


def test_ar2_gain_gives_unit_stationary_variance():
    for name in PAIN_SOURCES[1:]:
        index = PAIN_SOURCES.index(name)
        phi1, phi2, gain = sensing.ar2_coefficients(
            float(sensing.PAIN_PERIODS[index]), float(sensing.PAIN_DAMPING[index])
        )
        variance = gain**2 * (1.0 - phi2) / ((1.0 + phi2) * ((1.0 - phi2) ** 2 - phi1**2))
        assert variance == pytest.approx(1.0, rel=1e-9), name


def test_strengths_follow_the_documented_formulas():
    cfg = make_cfg()
    state = base_state(cfg)
    # Genome logits stay zero, so rho = (1/3, 1/3, 1/3).
    state = set_body(
        state,
        G=jnp.asarray([0.27, 0.0, 0.0], jnp.float32),
        N=jnp.asarray([0.0, 0.0, 0.0], jnp.float32),
        X=jnp.asarray(0.3, jnp.float32),
        b=jnp.asarray(0.8, jnp.float32),
        S=jnp.asarray(0.25, jnp.float32),
    )
    state = set_acute(state, 0.4)
    strengths = np.asarray(sensing.pain_strengths(state, cfg)[0])
    expected = {
        "acute": 0.4,
        "toxin": 0.6,
        "distension": 0.5,
        "heat": 0.3,
        "cold": 0.0,
        "hunger_plant": 1.0,
        "hunger_animal": 1.0,
        "hunger_toxin": 1.0,
        "sleepiness": 0.25,
    }
    assert strengths.tolist() == pytest.approx(
        [expected[name] for name in PAIN_SOURCES], abs=1e-6
    )


def test_carriers_peak_at_their_documented_periods():
    cfg = make_cfg(creatures=8)
    _, carriers = run_bundle(base_state(cfg), cfg, LONG_TICKS)
    for name in PAIN_SOURCES[1:]:
        index = PAIN_SOURCES.index(name)
        measured = estimate_period(carriers, index)
        assert measured == pytest.approx(EXPECTED_PERIODS[name], rel=0.06), name


def test_carrier_variance_is_unit_for_fast_sources():
    cfg = make_cfg(creatures=8)
    _, carriers = run_bundle(base_state(cfg), cfg, LONG_TICKS)
    for name in ("toxin", "distension", "heat", "cold", "sleepiness"):
        index = PAIN_SOURCES.index(name)
        variance = float(np.var(carriers[:, 0, index]))
        assert variance == pytest.approx(1.0, rel=0.25), name


def test_amplitude_scales_with_strength():
    cfg = make_cfg()
    weights = np.zeros((9, 4), np.float32)
    weights[PAIN_SOURCES.index("toxin"), 0] = 1.0
    weak_state = set_weights(
        set_body(base_state(cfg), X=jnp.asarray(0.1, jnp.float32)), weights
    )
    strong_state = set_weights(
        set_body(base_state(cfg), X=jnp.asarray(0.4, jnp.float32)), weights
    )
    weak, _ = run_bundle(weak_state, cfg, SHORT_TICKS)
    strong, _ = run_bundle(strong_state, cfg, SHORT_TICKS)
    weak_std = float(np.std(weak[:, 0, 0]))
    strong_std = float(np.std(strong[:, 0, 0]))
    # Strength is 2 * X, so the strong run is four times louder (the same 0.02
    # base noise floor is negligible against 0.2 and 0.8).
    assert strong_std / weak_std == pytest.approx(4.0, rel=0.1)
    assert weak_std == pytest.approx(0.2, rel=0.1)
    assert strong_std == pytest.approx(0.8, rel=0.1)


def test_zero_strength_leaves_only_the_base_noise():
    cfg = make_cfg()
    rho = 1.0 / 3.0
    state = set_body(
        base_state(cfg),
        G=jnp.zeros(3, jnp.float32),
        N=jnp.full(3, rho, jnp.float32),
        X=jnp.asarray(0.0, jnp.float32),
        b=jnp.asarray(0.0, jnp.float32),
        S=jnp.asarray(0.0, jnp.float32),
    )
    state = set_weights(state, np.ones((9, 4), np.float32))
    strengths = np.asarray(sensing.pain_strengths(state, cfg)[0])
    assert np.allclose(strengths, 0.0, atol=1e-7)

    signals, _ = run_bundle(state, cfg, SHORT_TICKS)
    for channel in range(4):
        std = float(np.std(signals[:, 0, channel]))
        assert std == pytest.approx(sensing.DEFAULT_NOISE.pain, rel=0.15)
        assert abs(float(np.mean(signals[:, 0, channel]))) < 0.005


def test_weights_mix_each_source_into_shared_channels():
    cfg = make_cfg()
    weights = np.zeros((9, 4), np.float32)
    weights[PAIN_SOURCES.index("sleepiness"), 2] = 1.0
    state = set_weights(
        set_body(base_state(cfg), S=jnp.asarray(1.0, jnp.float32)), weights
    )
    signals, carriers = run_bundle(state, cfg, SHORT_TICKS)
    source = carriers[:, 0, PAIN_SOURCES.index("sleepiness")]
    mixed = signals[:, 0, 2]
    assert float(np.corrcoef(source, mixed)[0, 1]) > 0.999
    # Channels without a weight carry only the base noise.
    for channel in (0, 1, 3):
        assert float(np.std(signals[:, 0, channel])) == pytest.approx(
            sensing.DEFAULT_NOISE.pain, rel=0.2
        )


def test_acute_is_a_smoothed_impulse_train():
    cfg = make_cfg()
    weights = np.zeros((9, 4), np.float32)
    weights[PAIN_SOURCES.index("acute"), 0] = 1.0
    firing = set_weights(base_state(cfg), weights)
    firing = set_acute(firing, jnp.asarray(1.0, jnp.float32))
    signals, carriers = run_bundle(firing, cfg, SHORT_TICKS)
    acute = carriers[:, 0, PAIN_SOURCES.index("acute")]
    # Every tick fires +-1 with probability one; the first-order filter with
    # phi = 0.5 leaves a stationary variance of 1/3.
    assert float(np.var(acute)) == pytest.approx(1.0 / 3.0, rel=0.2)
    assert float(np.std(signals[:, 0, 0])) == pytest.approx(np.sqrt(1.0 / 3.0), rel=0.2)

    silent = set_weights(base_state(cfg), weights)
    _, silent_carriers = run_bundle(silent, cfg, SHORT_TICKS)
    assert np.allclose(silent_carriers[:, 0, PAIN_SOURCES.index("acute")], 0.0, atol=1e-7)


def test_bundle_is_deterministic_for_the_same_key():
    cfg = make_cfg()
    state = base_state(cfg)
    first, _ = run_bundle(state, cfg, ticks=64)
    second, _ = run_bundle(state, cfg, ticks=64)
    assert np.array_equal(first, second)


def test_apply_advances_the_carrier_state():
    cfg = make_cfg()
    state = base_state(cfg)
    out, _ = sensing.apply(state, cfg)
    assert out.entities.creature.pain.x.shape == (1, 9, 2)
    assert float(jnp.abs(out.entities.creature.pain.x).sum()) > 0.0
