"""Modulated Hebbian rule (SANDBOX.md 6.1, 6.2).

Helper module called by learn.py.  For every plastic weight group the rule is

    dW_ij = eta_j * m~_j * (A_grp*pre_i*post_j + B_grp*pre_i + C_grp*post_j + D_grp)
            - lambda_j * W_ij
    W_fast <- clip(W_fast + dW, -1, 1)

with pre/post per group:

    W_in   pre s_t        post h_t
    W_rec  pre h_(t-1)    post h_t
    W_out  pre h_t        post out_t

A_grp ... D_grp are per-group genome scalars; eta_j and lambda_j are per
post-synaptic unit (SANDBOX 6.1) and m~_j is the per-post-unit modulation from
G in brain/base.py.  Fast weights change while awake; consolidation
W_slow += rho_j W_fast happens during sleep (ARCHITECTURES.md 3.4, phase 2b,
out of scope here).

Only arrays are touched here: no Python randomness, no state writes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Fast weights are bounded (SANDBOX.md 6.1); W = W_slow + W_fast therefore stays
# inside a bounded box, which keeps the recurrent net away from saturation blow-up.
W_FAST_CLIP: float = 1.0

# eta_granularity capability switch values (SANDBOX.md 6.2).
ETA_GRANULARITY_PER_UNIT = "per_unit"
ETA_GRANULARITY_PER_GROUP = "per_group"
ETA_GRANULARITY_SHARED = "shared"


def hebbian_update(
    weight: jax.Array,
    pre: jax.Array,
    post: jax.Array,
    eta: jax.Array,
    lam: jax.Array,
    modulation: jax.Array,
    coef_a: jax.Array,
    coef_b: jax.Array,
    coef_c: jax.Array,
    coef_d: jax.Array,
) -> jax.Array:
    """One modulated-Hebbian step for one weight group (SANDBOX.md 6.1).

    Shapes: weight (N, Q, P), pre (N, P), post (N, Q), eta/lam/modulation
    (N, Q), coef_* (N,) per-creature group coefficients.  Q is the number of
    post-synaptic units, P the number of pre-synaptic units.  The result is the
    updated fast weight, clipped to [-1, 1].
    """
    pre_term = pre[:, None, :]
    post_term = post[:, :, None]
    rule = (
        coef_a[:, None, None] * pre_term * post_term
        + coef_b[:, None, None] * pre_term
        + coef_c[:, None, None] * post_term
        + coef_d[:, None, None]
    )
    delta = eta[:, :, None] * modulation[:, :, None] * rule - lam[:, :, None] * weight
    return jnp.clip(weight + delta, -W_FAST_CLIP, W_FAST_CLIP)


def aggregate_rates(
    eta: jax.Array,
    lam: jax.Array,
    hidden_units: int,
    granularity: str,
) -> tuple[jax.Array, jax.Array]:
    """Apply the eta_granularity capability switch (SANDBOX.md 6.2).

    Rates are decoded per post-synaptic unit (rows 0..H-1 hidden units, rows
    H..H+N_OUT-1 output units).  This capability switch only changes how they
    are combined, never the genome layout:

    * per_unit  : use every row as decoded (H2 needs the per-unit distribution).
    * per_group : replace the hidden block and the output block each by their
                  mean.  W_in and W_rec share the hidden post units in this
                  layout, so both receive the hidden-block mean; W_out receives
                  the output-block mean.
    * shared    : one mean over all post units per creature.

    'eta' and 'lam' have shape (N, POST); hidden_units = H.
    """
    if granularity == ETA_GRANULARITY_PER_UNIT:
        return eta, lam
    post = eta.shape[1]
    if hidden_units <= 0 or hidden_units > post:
        raise ValueError("hidden_units must be within the post-unit axis")
    if granularity == ETA_GRANULARITY_PER_GROUP:
        return _per_group(eta, hidden_units), _per_group(lam, hidden_units)
    if granularity == ETA_GRANULARITY_SHARED:
        eta_shared = jnp.broadcast_to(
            jnp.mean(eta, axis=1, keepdims=True), eta.shape
        ).astype(eta.dtype)
        lam_shared = jnp.broadcast_to(
            jnp.mean(lam, axis=1, keepdims=True), lam.shape
        ).astype(lam.dtype)
        return eta_shared, lam_shared
    raise ValueError(
        "unknown eta granularity " + repr(granularity) + "; expected per_unit, per_group or shared"
    )


def _per_group(rates: jax.Array, hidden_units: int) -> jax.Array:
    hidden = jnp.mean(rates[:, :hidden_units], axis=1, keepdims=True)
    outputs = jnp.mean(rates[:, hidden_units:], axis=1, keepdims=True)
    n, post = rates.shape
    return jnp.concatenate(
        [
            jnp.broadcast_to(hidden, (n, hidden_units)),
            jnp.broadcast_to(outputs, (n, post - hidden_units)),
        ],
        axis=1,
    ).astype(rates.dtype)
