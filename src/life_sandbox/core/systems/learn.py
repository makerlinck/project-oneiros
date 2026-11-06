"""learn system (SANDBOX.md 3.3 step 5, 6.1, 6.2).

Contract: apply(state, cfg) -> (WorldState, EventBuffer).

Only cfg.brain.kind == "plastic_rnn" learns; the M0 bootstrap path has no
brain state to update (SANDBOX 9), so it returns the state unchanged.

Pipeline for every slot (all inputs are the values think just wrote):
  * e_t = BrainState.e (the e_t think used, stored verbatim);
  * prediction delta rule, capability switch cfg.learning.prediction:
        W_pred^fast += eta_p * e_t (outer) h_(t-1)
    (with the switch off, think has already silenced W_err and zeroed s_hat,
    and this system skips the delta rule, i.e. the SANDBOX 6.2 zeroed W_pred)
  * modulated Hebbian rule for W_in, W_rec, W_out, switch
    cfg.learning.plasticity:
        W_in : pre s_t = state.slice,               post h_t = BrainState.h
        W_rec: pre h_(t-1) = BrainState.h_prev,     post h_t
        W_out: pre h_t,                             post out_t = Motor.last_out
    eta_j/lambda_j are per post-synaptic unit; cfg.learning.eta_granularity
    decides whether they are used per unit, per group or shared
    (plasticity.aggregate_rates).  m~_j is recomputed from h_t and W_mod, a pure
    function of the state, because BrainState has no m slot.
  * new fast weights are written back to BrainState.w_fast (W_slow stays
    unchanged while awake; sleep consolidation is ARCHITECTURES 3.4, phase 2b).

The prediction s_hat_(t+1) was already computed by think with the weights of
this tick, exactly in the SANDBOX 3.3 order (step 4 gives s_hat, step 5 updates
W_pred^fast).  Every branch depends on static config only (no Python if on a
traced value), so the system runs under jit / lax.scan.  This system writes no
events.
"""

from __future__ import annotations

import jax.numpy as jnp

from life_sandbox.brain import genome
from life_sandbox.config import Config
from life_sandbox.core import events
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update
from life_sandbox.core.systems import plasticity, prediction
from life_sandbox.core.systems.think import KIND_PLASTIC_RNN


def _modulation(slow, h):
    """Per-post-unit modulation m~_j (SANDBOX 6.1), (N, H + N_OUT)."""
    m = jnp.tanh(jnp.einsum("nmh,nh->nm", slow.w_mod, h))
    return jnp.einsum("npm,nm->np", slow.g, m)


def apply(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    if cfg.brain.kind != KIND_PLASTIC_RNN:
        return state, events.empty(cfg)

    creature = state.entities.creature
    lay = genome.layout(cfg, slice_dim=int(state.slice.shape[1]))
    slow = genome.decode_slow(cfg, creature.genome.g, lay)
    plastic = genome.decode_plastic(cfg, creature.genome.g, lay)
    fast = genome.split_fast(creature.brain.w_fast, lay)

    brain = creature.brain
    s = state.slice
    h = brain.h
    h_prev = brain.h_prev
    e = brain.e
    out = creature.motor.last_out

    if cfg.learning.prediction:
        w_pred = prediction.delta_update(fast.w_pred, e, h_prev, plastic.eta_p)
    else:
        w_pred = fast.w_pred

    if cfg.learning.plasticity:
        eta, lam = plasticity.aggregate_rates(
            plastic.eta,
            plastic.lam,
            lay.h,
            cfg.learning.eta_granularity,
        )
        mod_tilde = _modulation(slow, h)
        hidden = lay.h
        coefs = plastic.hebb
        fast = genome.FastWeights(
            w_in=plasticity.hebbian_update(
                fast.w_in,
                s,
                h,
                eta[:, :hidden],
                lam[:, :hidden],
                mod_tilde[:, :hidden],
                coefs[:, 0, 0],
                coefs[:, 0, 1],
                coefs[:, 0, 2],
                coefs[:, 0, 3],
            ),
            w_rec=plasticity.hebbian_update(
                fast.w_rec,
                h_prev,
                h,
                eta[:, :hidden],
                lam[:, :hidden],
                mod_tilde[:, :hidden],
                coefs[:, 1, 0],
                coefs[:, 1, 1],
                coefs[:, 1, 2],
                coefs[:, 1, 3],
            ),
            w_out=plasticity.hebbian_update(
                fast.w_out,
                h,
                out,
                eta[:, hidden:],
                lam[:, hidden:],
                mod_tilde[:, hidden:],
                coefs[:, 2, 0],
                coefs[:, 2, 1],
                coefs[:, 2, 2],
                coefs[:, 2, 3],
            ),
            w_pred=w_pred,
        )
    else:
        fast = genome.FastWeights(
            w_in=fast.w_in,
            w_rec=fast.w_rec,
            w_out=fast.w_out,
            w_pred=w_pred,
        )

    brain = update(brain, w_fast=genome.join_fast(fast, lay))
    creature = update(creature, brain=brain)
    return update(state, entities=update(state.entities, creature=creature)), events.empty(cfg)
