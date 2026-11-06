"""Batch invariance (SANDBOX §2, §13).

One world must produce the same results alone and as a copy inside a vmap batch;
if that ever fails, the batch size becomes part of the reproduction scope.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from life_sandbox.core.hashing import state_hash
from life_sandbox.core.step import run, run_batch
from life_sandbox.world.init import initial_state


def _world(states, index: int):
    return jax.tree.map(lambda value: value[index], states)


def test_single_world_matches_batch(tiny_cfg):
    state = initial_state(tiny_cfg)
    single = run(state, tiny_cfg, 6)
    batch1 = run_batch(jax.tree.map(lambda value: value[None], state), tiny_cfg, 6)
    batch2 = run_batch(jax.tree.map(lambda value: jnp.stack([value, value]), state), tiny_cfg, 6)

    reference = state_hash(single.state)
    assert state_hash(_world(batch1.states, 0)) == reference
    assert state_hash(_world(batch2.states, 0)) == reference
    assert state_hash(_world(batch2.states, 1)) == reference
    assert single.counts == batch1.counts_dict(0) == batch2.counts_dict(0)
    assert np.array_equal(single.inflow, batch1.inflow[0])
    assert np.array_equal(single.outflow, batch2.outflow[0])
    assert np.array_equal(single.transfer_from, batch2.transfer_from[0])


def test_eager_batch_matches_compiled_batch(tiny_cfg):
    # C6/SANDBOX §2: bitwise equality is required within one execution mode only.
    # The eager reference and the fused jit path may differ by float32 ULPs, so
    # this check compares counts and the ledger totals with the §3.5 tolerance.
    state = initial_state(tiny_cfg)
    compiled = run_batch(jax.tree.map(lambda value: value[None], state), tiny_cfg, 4)
    eager = run_batch(
        jax.tree.map(lambda value: value[None], state), tiny_cfg, 4, compile=False
    )
    assert compiled.counts_dict(0) == eager.counts_dict(0)
    assert np.allclose(compiled.inflow, eager.inflow, rtol=1e-6, atol=1e-9)
    assert np.allclose(compiled.outflow, eager.outflow, rtol=1e-6, atol=1e-9)
