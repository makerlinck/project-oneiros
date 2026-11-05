"""World constants that are not experiment factors (SANDBOX.md §4, §5.1).

Values are initial calibration values (§4.2, §5.1); M1 is allowed to change
them, the prereg records the calibrated ones.
"""

from __future__ import annotations

NUM_BUSH_KINDS: int = 3
BUSHES_PER_KIND: int = 20

# Species odour spectra in the shared 4-channel bundle (§4.3).  The agent never
# sees the species label; it only smells the vector.
BUSH_ODOR: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)
ROT_ODOR: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)

# Berry composition (plant, animal, toxin) before the toxin schedule (§4.2).
INITIAL_COMP: tuple[tuple[float, ...], ...] = (
    (0.85, 0.15, 0.0),
    (0.75, 0.25, 0.0),
    (0.60, 0.40, 0.0),
)

# Body tissue composition is fixed (§5.1).
FLESH_COMP: tuple[float, float, float] = (0.3, 0.7, 0.0)

# Spatial temperature preference of each bush species (§4.2).
BUSH_T_PREF: tuple[float, ...] = (-1.5, 0.0, 1.2)
BUSH_T_SIGMA: float = 0.5

# Founder diet logits: plant/animal from N(0,1), toxin from N(-3,1) (§5.1).
FOUNDER_TOXIN_LOGIT_MEAN: float = -3.0
FOUNDER_LOGIT_STD: float = 1.0

# Genome init scale for founders (§6.3).
GENOME_INIT_STD: float = 0.1

# Bush appearance (§4.2): green mixed with the berry colour by stock.
BUSH_RADIUS: float = 1.0
BUSH_GREEN: tuple[float, float, float] = (0.2, 0.6, 0.2)
BERRY_RGB: tuple[tuple[float, float, float], ...] = (
    (0.85, 0.20, 0.20),
    (0.25, 0.30, 0.85),
    (0.60, 0.40, 0.20),
)

# Founder body defaults.  Founders start as adults (mu = 1, mass = 1) with half
# reserves and half tissue; these are calibration values for M1.
FOUNDER_RADIUS: float = 0.5
FOUNDER_MASS: float = 1.0
FOUNDER_INTEGRITY: float = 1.0
FOUNDER_RESERVE_FRACTION: float = 0.5
FOUNDER_TISSUE_FRACTION: float = 0.5

# Neutral objects (§4.5).
NEUTRAL_RADIUS: float = 0.4
NEUTRAL_MASS: float = 0.2
NEUTRAL_POSITIONS_RGB_MIN: float = 0.5
