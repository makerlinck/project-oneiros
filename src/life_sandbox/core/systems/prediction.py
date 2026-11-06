"""Prediction delta rule and prediction error (SANDBOX.md 6.1).

Helper module called by think.py / learn.py.

    e_t = s_t - s_hat_t                              (prediction error)
    W_pred^fast += eta_p * e_t (outer) h_(t-1)       (local delta rule)

s_hat_t is the prediction of the current slice produced on the previous tick
(SANDBOX.md 3.3: "think gives s_hat_(t+1)", "learn uses e_t = s_t - s_hat_t").
The error is the only architectural assumption (PRINCIPLES.md 3) and it also
feeds the hidden units through W_err, so think and learn must use bitwise the
same e_t -- think stores it in BrainState.e and learn reuses it.

The delta rule has no explicit clip in SANDBOX.md 6.1.  Documented local choice:
W_pred^fast is clipped to [-1, 1] like every other fast weight, so a long run
cannot drive W = W_slow + W_fast out of the same bounded box.  This is a local
implementation decision, not a semantic change to the rule.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Same bound as plasticity.W_FAST_CLIP (SANDBOX.md 6.1).
W_FAST_CLIP: float = 1.0


def error(s: jax.Array, s_hat_prev: jax.Array) -> jax.Array:
    """Prediction error e_t = s_t - s_hat_t, both (N, D)."""
    return s - s_hat_prev


def delta_update(
    weight: jax.Array,
    e: jax.Array,
    h_prev: jax.Array,
    eta_p: jax.Array,
) -> jax.Array:
    """Local delta rule for W_pred^fast (SANDBOX.md 6.1).

    Shapes: weight (N, D, H), e (N, D), h_prev (N, H), eta_p (N,).
    Returns W_pred^fast + eta_p * e (outer) h_(t-1), clipped to [-1, 1].
    """
    delta = eta_p[:, None, None] * e[:, :, None] * h_prev[:, None, :]
    return jnp.clip(weight + delta, -W_FAST_CLIP, W_FAST_CLIP)
