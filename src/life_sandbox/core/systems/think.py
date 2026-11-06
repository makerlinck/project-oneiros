"""think system (SANDBOX.md 3.3 step 4, 6.1; M0 random drive per 9).

Contract: apply(state, cfg) -> (WorldState, EventBuffer).

The architecture is selected by cfg.brain.kind (SANDBOX 6.4), never by an arm
name.  Two paths exist in M0:

* kind == "bootstrap": SANDBOX 9 says M0 drives creatures with random outputs,
  so one call to brain/base.bootstrap_outputs writes Motor.last_out and nothing
  else.  The brain state stays at its initialization (h, s_hat, w_fast zeros),
  in particular s_hat stays zero.  Randomness comes from
  rng.system_key(state.key, state.tick, "think") only.
* kind == "plastic_rnn": the default brain B0 (SANDBOX 6.1):
    1. read the slice s_t written by sense, the previous hidden state h_(t-1)
       and the prediction s_hat_t of the current slice (made on the previous
       tick);
    2. form the prediction error e_t = s_t - s_hat_t (prediction.error);
    3. B0 forward pass (brain/base.forward): h_t, the six body outputs, m_t,
       m~_j and s_hat_(t+1) = W_pred h_t with the weights at the start of the
       tick;
    4. write BrainState.h = h_t, BrainState.s_hat = s_hat_(t+1),
       Motor.last_out = out_t.
  Two extra BrainState slots carry what learn needs so think and learn use
  bitwise the same e_t (SANDBOX 3.3 step 5: "learn uses e_t = s_t - s_hat_t"):
      BrainState.e      <- e_t       (D slots; learn cannot recompute it
                                      because s_hat has already been overwritten
                                      with s_hat_(t+1))
      BrainState.h_prev <- h_(t-1)   (H slots; W_rec pre and the W_pred delta
                                      rule need it)
  Decision recorded here: SANDBOX 3.2 names the slots "pre and post activities
  of the previous step"; this implementation uses them as the prediction error
  and the previous hidden state, which is what the M0 rules actually need.  m_t
  is not stored (BrainState has no slot for it); learn recomputes it from h_t
  and W_mod, a pure function of the state.

Other cfg.brain.kind values are candidate architectures of SANDBOX 6.4 that are
not implemented in M0; they raise NotImplementedError at trace time (the kind is
static configuration, never a traced value).

All slots are updated, alive or not: rows are independent, dead rows are
overwritten on slot reuse, and masking would only add a branch without changing
any living creature.  The slice width is taken from state.slice.shape[1]
(cfg.learning.readable_leak can widen it, SANDBOX 5.6) so jit/lax.scan see the
array that is actually present.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from life_sandbox.brain import base, genome
from life_sandbox.config import Config
from life_sandbox.core import events, rng
from life_sandbox.core.state import WorldState
from life_sandbox.core.struct import update
from life_sandbox.core.systems import prediction

KIND_BOOTSTRAP = "bootstrap"
KIND_PLASTIC_RNN = "plastic_rnn"


def _bootstrap(state: WorldState, cfg: Config) -> WorldState:
    """M0 random drive: only Motor.last_out is written (SANDBOX 9)."""
    creature = state.entities.creature
    key = rng.system_key(state.key, state.tick, "think")
    n = creature.motor.last_out.shape[0]
    out = base.bootstrap_outputs(key, n)
    creature = update(creature, motor=update(creature.motor, last_out=out))
    return update(state, entities=update(state.entities, creature=creature))


def _plastic_rnn(state: WorldState, cfg: Config) -> WorldState:
    creature = state.entities.creature
    lay = genome.layout(cfg, slice_dim=int(state.slice.shape[1]))
    slow = genome.decode_slow(cfg, creature.genome.g, lay)
    fast = genome.split_fast(creature.brain.w_fast, lay)

    s = state.slice
    h_prev = creature.brain.h
    e = prediction.error(s, creature.brain.s_hat)

    key = rng.system_key(state.key, state.tick, "think")
    noise = jax.random.normal(key, h_prev.shape, dtype=jnp.float32)

    result = base.forward(
        slow,
        fast,
        s,
        e,
        h_prev,
        creature.body.S,
        noise,
        prediction_enabled=cfg.learning.prediction,
    )

    brain = update(
        creature.brain,
        h=result.h,
        s_hat=result.prediction,
        e=e,
        h_prev=h_prev,
    )
    motor = update(creature.motor, last_out=result.out)
    creature = update(creature, brain=brain, motor=motor)
    return update(state, entities=update(state.entities, creature=creature))


def apply(state: WorldState, cfg: Config) -> tuple[WorldState, events.EventBuffer]:
    kind = cfg.brain.kind
    if kind == KIND_BOOTSTRAP:
        return _bootstrap(state, cfg), events.empty(cfg)
    if kind == KIND_PLASTIC_RNN:
        return _plastic_rnn(state, cfg), events.empty(cfg)
    raise NotImplementedError(
        "brain kind " + repr(kind) + " is a candidate architecture not implemented in M0"
    )
