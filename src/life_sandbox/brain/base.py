"""Default brain B0 and the M0 bootstrap policy (SANDBOX.md 6.1, 9).

Two separate things live here:

* bootstrap_outputs -- the M0 placeholder only.  SANDBOX.md 9 lets M0
  "drive creatures with random outputs" while no brain exists.  It is NOT a
  candidate brain and no capability switch selects it: it just keeps the engine
  runnable without a decoded genome.  The default brain landed in this module,
  so think runs B0 directly; the helper is kept for the M0 random-policy check
  and is tested on its own.
* forward -- the default brain B0 of SANDBOX.md 6.1: a modulated plastic
  recurrent network.  Every weight is W_slow + W_fast: W_slow is decoded from
  the genome (brain/genome.py), W_fast is BrainState.w_fast.

Output layout inside W_out: rows 0..4 are the five movement outputs
(u_acc, u_turn, u_bite, u_grab, u_drop, all tanh) and row 5 is the metabolic
knob k (sigmoid).  out is reordered into the canonical Motor.last_out order of
components.OUT_* (u_acc, u_turn, k, u_bite, u_grab, u_drop).

This module writes no state and holds no event buffer: think/learn own the
WorldState writes, so the forward pass is a pure function of arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from life_sandbox.core import components as C

# Hidden-unit noise scale, SANDBOX.md 6.1: sigma_n = 0.3, scaled by sleep
# pressure S (Body.S).  It is a form constant, not an experiment factor.
SIGMA_N: float = 0.3

# Rows 0..4 of W_out are the movement outputs; row 5 is the metabolic knob.
N_MOVEMENT: int = C.N_OUTPUTS - 1


@dataclass(frozen=True)
class SlowWeights:
    """Genome-decoded (non-plastic while awake) B0 parameters.

    Shapes use a leading slot/creature axis of length N: (N, ...).
    """

    w_in: jax.Array  # (N, H, D)  slice -> hidden
    w_err: jax.Array  # (N, H, D)  prediction error -> hidden
    w_rec: jax.Array  # (N, H, H)  hidden -> hidden (post, pre)
    w_out: jax.Array  # (N, N_OUT, H)  hidden -> outputs (rows 0..4 move, row 5 = k)
    w_mod: jax.Array  # (N, M, H)  hidden -> modulators
    g: jax.Array  # (N, H + N_OUT, M)  modulator mixing per post-synaptic unit
    w_pred: jax.Array  # (N, D, H)  hidden -> next slice
    b: jax.Array  # (N, H)
    b_out: jax.Array  # (N, N_OUT)


@dataclass(frozen=True)
class FastWeights:
    """Awake-plastic weights (BrainState.w_fast), same layout as their slow twins."""

    w_in: jax.Array  # (N, H, D)
    w_rec: jax.Array  # (N, H, H)
    w_out: jax.Array  # (N, N_OUT, H)
    w_pred: jax.Array  # (N, D, H)


@dataclass(frozen=True)
class ForwardResult:
    """One B0 forward pass (SANDBOX.md 6.1)."""

    h: jax.Array  # (N, H) internal state h_t
    out: jax.Array  # (N, 6) body outputs in Motor.last_out order
    m: jax.Array  # (N, M) modulator activity m_t
    m_tilde: jax.Array  # (N, H + N_OUT) per-post-unit modulation m~_j
    prediction: jax.Array  # (N, D) s_hat_(t+1) = W_pred h_t


def bootstrap_outputs(key: jax.Array, n: int) -> jax.Array:
    """M0 random-drive policy: six body outputs, SANDBOX.md 9.

    Returns float32 of shape (n, 6) in Motor.last_out order: u_acc, u_turn,
    u_bite, u_grab, u_drop are uniform in [-1, 1); k is uniform in [0, 1).

    This is NOT a candidate brain.  It exists so that M0 can run the engine
    before the default brain B0 was wired in; every random draw comes from the
    supplied key (never Python randomness), so a run stays deterministic.
    """
    key_move, key_knob = jax.random.split(key, 2)
    outputs = jax.random.uniform(
        key_move,
        (int(n), C.N_OUTPUTS),
        dtype=jnp.float32,
        minval=-1.0,
        maxval=1.0,
    )
    knob = jax.random.uniform(key_knob, (int(n),), dtype=jnp.float32, minval=0.0, maxval=1.0)
    return outputs.at[:, C.OUT_K].set(knob)


def forward(
    slow: SlowWeights,
    fast: FastWeights,
    s: jax.Array,
    e: jax.Array,
    h_prev: jax.Array,
    sleep_pressure: jax.Array,
    noise: jax.Array,
    *,
    prediction_enabled: bool,
) -> ForwardResult:
    """B0 forward pass, SANDBOX.md 6.1.

    h_t = tanh(W_in s_t + W_err e_t + W_rec h_(t-1) + b + sigma_n S xi)
    out_t = [tanh(W_out h_t + b_out) for the 5 movement rows, sigmoid for k]
    m_t = tanh(W_mod h_t);  m~_j = sum_k G[j, k] m_t[k]
    s_hat_(t+1) = W_pred h_t

    'prediction_enabled' is the capability switch for the error units: when it
    is off the W_err pathway is silenced and s_hat is zero, which is the
    SANDBOX.md 6.2 "W_pred, W_err set to zero" arm expressed as a switch.

    'noise' is a standard-normal (N, H) array drawn by the caller from the
    think key; it is multiplied by sleep pressure, so an awake creature (S=0)
    gets the same h regardless of the draw and the random stream is unchanged.
    """
    w_in = slow.w_in + fast.w_in
    w_rec = slow.w_rec + fast.w_rec
    w_out = slow.w_out + fast.w_out

    drive = jnp.einsum("nhd,nd->nh", w_in, s)
    if prediction_enabled:
        drive = drive + jnp.einsum("nhd,nd->nh", slow.w_err, e)
    drive = drive + jnp.einsum("nhk,nk->nh", w_rec, h_prev)
    drive = drive + slow.b
    drive = drive + SIGMA_N * sleep_pressure[:, None] * noise
    h = jnp.tanh(drive)

    out_pre = jnp.einsum("noh,nh->no", w_out, h) + slow.b_out
    movement = jnp.tanh(out_pre[:, :N_MOVEMENT])
    knob = jax.nn.sigmoid(out_pre[:, N_MOVEMENT])
    out = jnp.stack(
        [
            movement[:, C.OUT_ACC],
            movement[:, C.OUT_TURN],
            knob,
            movement[:, C.OUT_BITE - 1],
            movement[:, C.OUT_GRAB - 1],
            movement[:, C.OUT_DROP - 1],
        ],
        axis=1,
    )

    m = jnp.tanh(jnp.einsum("nmh,nh->nm", slow.w_mod, h))
    m_tilde = jnp.einsum("npm,nm->np", slow.g, m)

    if prediction_enabled:
        prediction = jnp.einsum("ndh,nh->nd", slow.w_pred + fast.w_pred, h)
    else:
        prediction = jnp.zeros_like(s)

    return ForwardResult(h=h, out=out, m=m, m_tilde=m_tilde, prediction=prediction)
