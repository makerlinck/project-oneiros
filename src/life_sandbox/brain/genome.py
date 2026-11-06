"""Genome decode layout for the default brain B0 (SANDBOX.md 6.1, 6.3).

The shared header lives in core/genome.py: diet logits [0:3], colour [3:6],
odour [6:6+C_o], theta_repro at 6+C_o.  Brain-specific blocks start at
core.genome.brain_base(odor_channels) = 7+C_o (same file).

Genome width and fast-weight width are derived from the layout, never from
configuration: GenomeLayout.genome_dim == GenomeLayout.total and
GenomeLayout.plastic_dim == H*D + H*H + N_OUT*H + D*H.  Changing H, D, M or the
slice width therefore moves every downstream array automatically (SANDBOX 6.3).

Block order (absolute indices into Genome.g, every block float32):

    block       size        row-major layout
    w_in        H*D         [hidden j, slice i]        W_slow
    w_err       H*D         [hidden j, slice i]        W_slow (error units)
    w_rec       H*H         [post j, pre k]            W_slow
    w_out       N_OUT*H     [output o, hidden j]       W_slow; rows 0..4 movement
                                                       (u_acc,u_turn,u_bite,u_grab,
                                                       u_drop, tanh), row 5 k (sigmoid)
    w_mod       M*H         [modulator k, hidden j]    W_slow
    g           POST*M      [post-synaptic unit j, modulator k]  G in 6.1
    w_pred      D*H         [slice i, hidden j]        W_slow
    b           H           hidden bias
    b_out       N_OUT       output bias (rows as w_out)
    log_eta     POST        log learning rate per post-synaptic unit
    log_lambda  POST        log decay rate per post-synaptic unit
    hebb        N_GRP*4     A, B, C, D per Hebbian weight group
    log_eta_p   1           log prediction (delta-rule) learning rate

POST = H + N_OUT.  SANDBOX.md 6.1 writes the modulation and the rate per
"post-synaptic unit j": W_in/W_rec have H hidden post units, W_out has the
N_OUT output post units, so one contiguous POST-sized block covers all three
groups (rows 0..H-1 hidden, rows H..H+N_OUT-1 outputs).  This is the documented
resolution of the ambiguity, and it deviates from the SANDBOX.md 6.5 size table,
which estimates G as H*M = 128 because it does not spell out the W_out post
units; the formula in 6.1 needs one row per post-synaptic unit, i.e.
(H + N_OUT)*M = 152 at the default size.  No information is added: the extra
rows are part of the same evolved G matrix.

Hebbian groups, in block order: 0 = W_in (pre s_t, post h_t), 1 = W_rec
(pre h_(t-1), post h_t), 2 = W_out (pre h_t, post out_t).  A-D are per group
per SANDBOX.md 6.1 (A_grp ... D_grp).

D is the slice width.  cfg.learning.readable_leak opens the leak arm of
SANDBOX.md 5.6, which appends 5 designer dimensions to the slice; the
caller (think/learn) passes the actual width of state.slice so a jitted system
never disagrees with the traced array.  Widening D moves the later block offsets
and grows the fast weights, so such an arm needs a wider layout (out of M0
scope; readable_leak is rejected by Config.validate until M3).

The sleep-consolidation rate rho_j (ARCHITECTURES.md 3.4, phase 2b) is not part
of the M0 genome; it will be appended as a new block when consolidation lands,
which keeps every existing offset stable.

eta/lambda/eta_p are stored as logs (SANDBOX.md 6.3) and decoded with exp(); a
genome entry near 0 therefore decodes to a rate near 1.  The founder init and
the M1/M2 calibration own the distribution of those log entries, not this
decoder.

W_fast (BrainState.w_fast, layout.plastic_dim slots) holds only the four
plastic matrices, in this order: w_in, w_rec, w_out, w_pred.  Their sizes sum
to layout.plastic_dim (H*D + H*H + N_OUT*H + D*H).

mutation_sigmas(cfg) returns the per-block mutation scale used by reproduction
(SANDBOX.md 6.3): genome blocks mutate with g' = g + sigma_block * N(0, 1).

initial_sigmas(cfg) returns the per-slot standard deviation world/init draws
founder genomes with (C4 weight-init table, moved before M1): fan-in scaling
1/sqrt(D) for W_in/W_err and 1/sqrt(H) for W_rec/W_out/W_mod/W_pred, g scaled by
1/sqrt(M); b is 0.1 and b_out 1.0 (output units start with distinct
preferences); the plasticity meta-parameters and Hebbian A-D stay 0.1 until C4's
plasticity init is fixed before M2; shared header slots default to 0.1 because
world/init overwrites diet/colour/odour/theta_repro with the founder conventions
of SANDBOX 5.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp

from life_sandbox.brain.base import FastWeights, SlowWeights
from life_sandbox.core import components as C
from life_sandbox.core import genome as core_genome

# Hebbian weight groups in block order (SANDBOX.md 6.1: W_in, W_rec, W_out).
HEBB_GROUPS: tuple[str, ...] = ("w_in", "w_rec", "w_out")
N_HEBB_GROUPS: int = len(HEBB_GROUPS)
# A, B, C, D of the modulated Hebbian rule (SANDBOX.md 6.1).
N_HEBB_COEFS: int = 4

# Mutation scales of SANDBOX.md 6.3.  Weight-like blocks use 0.02; the log
# plasticities use 0.1 (they are logs); the shared header uses the 5.1/6.3
# values (0.05 for colour/odour/diet, 0.02 for theta_repro).
SIGMA_WEIGHT: float = 0.02
SIGMA_LOG_RATE: float = 0.1
SIGMA_HEADER_TRAIT: float = 0.05
SIGMA_THETA_REPRO: float = 0.02

# Initial genome scales (C4 weight-init table, moved before M1; SANDBOX 6.3).
# Fan-in scaled: a hidden unit sums D (or H) zero-mean inputs, so weight std
# 1/sqrt(fan_in) keeps its drive O(1) and tanh away from saturation.
INIT_BIAS_STD: float = 0.1
INIT_OUTPUT_BIAS_STD: float = 1.0  # output units start with distinct preferences
INIT_LOG_RATE_STD: float = 0.1  # C4 plasticity init lands before M2
INIT_HEBB_COEF_STD: float = 0.1  # A-D of SANDBOX 6.1; C4 before M2
INIT_HEADER_STD: float = 0.1  # world/init fills the shared header itself

# Slice dimensions appended when cfg.learning.readable_leak is on (SANDBOX 5.6).
LEAK_DIM: int = 5


@dataclass(frozen=True)
class GenomeLayout:
    """Absolute genome slices of every brain block for one config."""

    h: int
    d: int
    m: int
    n_out: int
    post: int
    base: int
    w_in: slice
    w_err: slice
    w_rec: slice
    w_out: slice
    w_mod: slice
    g: slice
    w_pred: slice
    b: slice
    b_out: slice
    log_eta: slice
    log_lambda: slice
    hebb: slice
    log_eta_p: slice
    total: int
    plastic_dim: int

    @property
    def genome_dim(self) -> int:
        """Total genome width implied by this layout (shared header included)."""
        return self.total

    def block_sizes(self) -> dict[str, int]:
        """Size in genome slots of every brain block (for tests and diagnostics)."""
        names = (
            "w_in",
            "w_err",
            "w_rec",
            "w_out",
            "w_mod",
            "g",
            "w_pred",
            "b",
            "b_out",
            "log_eta",
            "log_lambda",
            "hebb",
            "log_eta_p",
        )
        return {name: getattr(self, name).stop - getattr(self, name).start for name in names}


@dataclass(frozen=True)
class PlasticParams:
    """Decoded plasticity meta-parameters (SANDBOX.md 6.1, 6.3)."""

    eta: jax.Array  # (N, POST) learning rate per post-synaptic unit
    lam: jax.Array  # (N, POST) decay rate per post-synaptic unit
    hebb: jax.Array  # (N, N_HEBB_GROUPS, 4) A, B, C, D per weight group
    eta_p: jax.Array  # (N,) prediction delta-rule learning rate


def config_slice_dim(cfg) -> int:
    """Slice width the config asks for, including the readable_leak switch."""
    return int(cfg.sensors.slice_dim) + (LEAK_DIM if cfg.learning.readable_leak else 0)


def layout(cfg, slice_dim: int | None = None) -> GenomeLayout:
    """Resolve the brain genome layout; its widths are the derived truth.

    'slice_dim' overrides the config-derived slice width; think/learn pass
    state.slice.shape[1] so the decoded W_in/W_err/w_pred always match the array
    they multiply, even under jit.  Genome and fast-weight widths are NOT
    configured: they follow from H, D, M and N_OUT (SANDBOX.md §6.3), so a
    change of D moves every array automatically.

    GenomeLayout.genome_dim (== total) is the genome width and
    GenomeLayout.plastic_dim the BrainState.w_fast width for this config.

    The arithmetic is cached on static config values (lru_cache): think/learn
    call this on every tick, and under lax.scan the call happens once per trace,
    so rebuilding a layout must stay a pure function of static integers.
    """
    return _layout_cached(
        int(cfg.brain.h),
        int(config_slice_dim(cfg) if slice_dim is None else slice_dim),
        int(cfg.brain.modulators),
        C.N_OUTPUTS,
        core_genome.brain_base(int(cfg.sensors.odor_channels)),
    )


@lru_cache(maxsize=32)
def _layout_cached(
    h: int,
    d: int,
    m: int,
    n_out: int,
    base: int,
) -> GenomeLayout:
    """Build one layout from static integers (see layout())."""
    if h <= 0 or d <= 0 or m <= 0 or n_out <= 0:
        raise ValueError("brain genome needs positive H, D, M and output count")
    post = h + n_out

    cursor = base

    def take(size: int) -> slice:
        nonlocal cursor
        if size < 0:
            raise ValueError("negative brain genome block size")
        block = slice(cursor, cursor + size)
        cursor += size
        return block

    w_in = take(h * d)
    w_err = take(h * d)
    w_rec = take(h * h)
    w_out = take(n_out * h)
    w_mod = take(m * h)
    g = take(post * m)
    w_pred = take(d * h)
    b = take(h)
    b_out = take(n_out)
    log_eta = take(post)
    log_lambda = take(post)
    hebb = take(N_HEBB_GROUPS * N_HEBB_COEFS)
    log_eta_p = take(1)
    total = cursor

    # Fast-weight width: W_in + W_rec + W_out + W_pred (SANDBOX.md §6.5).
    layout_plastic_dim = h * d + h * h + n_out * h + d * h
    return GenomeLayout(
        h=h,
        d=d,
        m=m,
        n_out=n_out,
        post=post,
        base=base,
        w_in=w_in,
        w_err=w_err,
        w_rec=w_rec,
        w_out=w_out,
        w_mod=w_mod,
        g=g,
        w_pred=w_pred,
        b=b,
        b_out=b_out,
        log_eta=log_eta,
        log_lambda=log_lambda,
        hebb=hebb,
        log_eta_p=log_eta_p,
        total=total,
        plastic_dim=layout_plastic_dim,
    )


def _check_genome(cfg, genome: jax.Array, lay: GenomeLayout | None = None) -> None:
    if genome.ndim != 2:
        raise ValueError("genome must be (N, genome_dim), got ndim " + str(genome.ndim))
    expected = layout(cfg).total if lay is None else lay.total
    if genome.shape[1] != expected:
        raise ValueError(
            "genome width " + str(genome.shape[1]) + " does not match the B0 layout " + str(expected)
        )


def decode_slow(
    cfg,
    genome: jax.Array,
    lay: GenomeLayout | None = None,
) -> SlowWeights:
    """Decode W_slow and the biases of B0 from the genome (SANDBOX.md 6.1).

    'genome' has shape (N, layout.genome_dim); every block is read row-major
    with the layout documented at the top of this module.  The values are used
    as-is: founder init and mutation own the scale (SANDBOX.md 6.3).
    """
    lay = layout(cfg) if lay is None else lay
    _check_genome(cfg, genome, lay)
    n = genome.shape[0]
    return SlowWeights(
        w_in=genome[:, lay.w_in].reshape(n, lay.h, lay.d),
        w_err=genome[:, lay.w_err].reshape(n, lay.h, lay.d),
        w_rec=genome[:, lay.w_rec].reshape(n, lay.h, lay.h),
        w_out=genome[:, lay.w_out].reshape(n, lay.n_out, lay.h),
        w_mod=genome[:, lay.w_mod].reshape(n, lay.m, lay.h),
        g=genome[:, lay.g].reshape(n, lay.post, lay.m),
        w_pred=genome[:, lay.w_pred].reshape(n, lay.d, lay.h),
        b=genome[:, lay.b],
        b_out=genome[:, lay.b_out],
    )


def decode_plastic(
    cfg,
    genome: jax.Array,
    lay: GenomeLayout | None = None,
) -> PlasticParams:
    """Decode the plasticity meta-parameters (SANDBOX.md 6.1, 6.3).

    eta_j = exp(log_eta_j), lambda_j = exp(log_lambda_j) and
    eta_p = exp(log_eta_p), per SANDBOX.md 6.3 ("stored as logs").
    """
    lay = layout(cfg) if lay is None else lay
    _check_genome(cfg, genome, lay)
    n = genome.shape[0]
    return PlasticParams(
        eta=jnp.exp(genome[:, lay.log_eta]),
        lam=jnp.exp(genome[:, lay.log_lambda]),
        hebb=genome[:, lay.hebb].reshape(n, N_HEBB_GROUPS, N_HEBB_COEFS),
        eta_p=jnp.exp(genome[:, lay.log_eta_p].reshape(n)),
    )


def split_fast(w_fast: jax.Array, lay: GenomeLayout) -> FastWeights:
    """Split BrainState.w_fast into the four plastic matrices.

    Flat order: w_in, w_rec, w_out, w_pred (documented at the top of the file).
    """
    n = w_fast.shape[0]
    size_in = lay.h * lay.d
    size_rec = lay.h * lay.h
    size_out = lay.n_out * lay.h
    size_pred = lay.d * lay.h
    expected = size_in + size_rec + size_out + size_pred
    if w_fast.shape[1] != expected:
        raise ValueError(
            "w_fast width " + str(w_fast.shape[1]) + " does not match layout " + str(expected)
        )
    offset = 0
    w_in = w_fast[:, offset : offset + size_in].reshape(n, lay.h, lay.d)
    offset += size_in
    w_rec = w_fast[:, offset : offset + size_rec].reshape(n, lay.h, lay.h)
    offset += size_rec
    w_out = w_fast[:, offset : offset + size_out].reshape(n, lay.n_out, lay.h)
    offset += size_out
    w_pred = w_fast[:, offset : offset + size_pred].reshape(n, lay.d, lay.h)
    return FastWeights(w_in=w_in, w_rec=w_rec, w_out=w_out, w_pred=w_pred)


def join_fast(fast: FastWeights, lay: GenomeLayout) -> jax.Array:
    """Flatten the four plastic matrices back into BrainState.w_fast."""
    n = fast.w_in.shape[0]
    return jnp.concatenate(
        [
            fast.w_in.reshape(n, lay.h * lay.d),
            fast.w_rec.reshape(n, lay.h * lay.h),
            fast.w_out.reshape(n, lay.n_out * lay.h),
            fast.w_pred.reshape(n, lay.d * lay.h),
        ],
        axis=1,
    )


def mutation_sigmas(cfg) -> jax.Array:
    """Per-slot mutation scale for the whole genome (SANDBOX.md 6.3).

    Returns float32[layout(cfg).genome_dim] -- exactly the derived genome width,
    so mutation and genome arrays never disagree.  Mutation is
    g' = g + sigma * N(0, 1) with one sigma per block (SANDBOX.md 6.3): shared
    header traits 0.05 except theta_repro 0.02; weight-like brain blocks 0.02;
    log eta/lambda/eta_p 0.1.  When a block is appended later (e.g. log rho in
    phase 2b) the layout and this array grow together and the new block picks
    its own scale.
    """
    lay = layout(cfg)
    sigma = jnp.full((int(lay.total),), SIGMA_WEIGHT, dtype=jnp.float32)
    # Shared header (core/genome.py owns the offsets).
    odor_channels = int(cfg.sensors.odor_channels)
    theta = core_genome.theta_repro_index(odor_channels)
    sigma = sigma.at[core_genome.diet_logits_slice()].set(SIGMA_HEADER_TRAIT)
    sigma = sigma.at[core_genome.color_slice()].set(SIGMA_HEADER_TRAIT)
    sigma = sigma.at[core_genome.odor_slice(odor_channels)].set(SIGMA_HEADER_TRAIT)
    sigma = sigma.at[theta].set(SIGMA_THETA_REPRO)
    # Brain blocks.  Weight-like blocks (w_in, w_err, w_rec, w_out, w_mod, g,
    # w_pred, b, b_out, hebb A-D) keep the default 0.02; the log rates differ.
    sigma = sigma.at[lay.log_eta].set(SIGMA_LOG_RATE)
    sigma = sigma.at[lay.log_lambda].set(SIGMA_LOG_RATE)
    sigma = sigma.at[lay.log_eta_p].set(SIGMA_LOG_RATE)
    return sigma


def initial_sigmas(cfg) -> jax.Array:
    """Per-slot standard deviation of a founder genome (C4, SANDBOX.md 6.3).

    Returns float32[layout(cfg).genome_dim].  world/init draws founder genomes
    as sigma_slot * N(0, 1), so the scale is per genome slot and follows the
    block layout.  Fan-in scaling (SANDBOX 6.3 weight init):

        w_in, w_err            std = 1/sqrt(D)   (fan-in = slice width)
        w_rec, w_out, w_mod,
        w_pred                 std = 1/sqrt(H)   (fan-in = hidden width)
        g                      std = 1/sqrt(M)   (modulator mixing)
        b                      std = 0.1
        b_out                  std = 1.0         (distinct output preferences)

    The plasticity meta-parameters (log eta/lambda/eta_p) and the Hebbian A-D
    coefficients stay at 0.1 until C4's plasticity initialisation is fixed
    before M2.  The shared header (diet/colour/odour/theta_repro) defaults to
    0.1; world/init overwrites it with the founder conventions of SANDBOX 5.1,
    so this function does not encode them.
    """
    lay = layout(cfg)
    sigma = jnp.full((int(lay.total),), INIT_HEADER_STD, dtype=jnp.float32)
    w_in_std = jnp.float32(1.0 / math.sqrt(lay.d))
    w_hidden_std = jnp.float32(1.0 / math.sqrt(lay.h))
    g_std = jnp.float32(1.0 / math.sqrt(lay.m))
    sigma = sigma.at[lay.w_in].set(w_in_std)
    sigma = sigma.at[lay.w_err].set(w_in_std)
    sigma = sigma.at[lay.w_rec].set(w_hidden_std)
    sigma = sigma.at[lay.w_out].set(w_hidden_std)
    sigma = sigma.at[lay.w_mod].set(w_hidden_std)
    sigma = sigma.at[lay.w_pred].set(w_hidden_std)
    sigma = sigma.at[lay.g].set(g_std)
    sigma = sigma.at[lay.b].set(INIT_BIAS_STD)
    sigma = sigma.at[lay.b_out].set(INIT_OUTPUT_BIAS_STD)
    sigma = sigma.at[lay.log_eta].set(INIT_LOG_RATE_STD)
    sigma = sigma.at[lay.log_lambda].set(INIT_LOG_RATE_STD)
    sigma = sigma.at[lay.log_eta_p].set(INIT_LOG_RATE_STD)
    sigma = sigma.at[lay.hebb].set(INIT_HEBB_COEF_STD)
    return sigma
