"""
Utilities for sampling orbital parameters (inclination / eccentricity).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from scipy.stats import beta

import numpy as np


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def sample_cos_i(n: int = 1) -> np.ndarray:
    """
    Sample cos(inclination) uniformly from [0, 1].
    """
    return np.random.uniform(0, 1, n)


def sample_i(n: int = 1) -> np.ndarray:
    """
    Sample `n` inclinations such that cos(i) is uniform in [0, 1].
    The resulting values are returned in *degree*.
    """
    return np.rad2deg(np.arccos(sample_cos_i(n)))


def sample_e(n: int = 1, a: float = 0.867, b: float = 3.03) -> np.ndarray:
    """
    Sample `n` eccentricities `e` from a beta distribution with shape
    parameters `a` and `b`. The default values for `a` and `b` are
    taken from:

        Guimond and Cowan (2019): arXiv:1903.06184

    We are truncating `e` to [0, 1] to exclude hyperbolic orbits.
    """

    return np.clip(beta.rvs(a=a, b=b, size=n), 0, 1)


# -----------------------------------------------------------------------------
# MAIN CODE (TESTS)
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------------
    # Test the sampling of inclinations
    # -------------------------------------------------------------------------

    n = 100_000

    # Sample inclinations i uniformly in cos(i)
    inclinations = sample_i(n)

    # Plot a histogram of the inclinations
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(
        inclinations,
        bins=np.linspace(0, 90, 90),
        histtype="step",
        lw=2,
        density=True,
    )
    ax.set_xlim(0, 90)
    ax.set_xlabel("Inclination (deg)")
    ax.set_ylabel("Density")

    fig.tight_layout()
    plt.show()

    n = 100_000

    # -------------------------------------------------------------------------
    # Test the sampling of eccentricities
    # -------------------------------------------------------------------------

    # Sample eccentricities from beta distribution
    eccentricities = sample_e(n)

    # Plot a histogram of the inclinations
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(
        eccentricities,
        bins=np.linspace(0, 1, 100),
        histtype="step",
        lw=2,
        density=True,
        zorder=1,
    )

    # Plot the beta PDF
    x = np.linspace(
        beta.ppf(0.001, 0.867, 3.03),
        beta.ppf(0.999, 0.867, 3.03),
        100
    )
    ax.plot(x, beta.pdf(x, 0.867, 3.03), "gray", lw=1, ls="--", zorder=0)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Eccentricity")
    ax.set_ylabel("Density")

    plt.show()
