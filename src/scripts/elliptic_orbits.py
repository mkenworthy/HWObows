"""
Utility functions for computing beta_min / beta_max.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from scipy.spatial.transform import Rotation

import astropy.units as u
import numpy as np


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def xyz_position(
    phases: np.ndarray,
    sma: u.Quantity[u.AU],
    e: float,
    aperi: u.Quantity[u.rad],
    lan: u.Quantity[u.rad],
    inc: u.Quantity[u.rad],
) -> np.ndarray:
    """
    Take the Keplerian orbital elements and compute the position of the
    planet in x, y, z coordinates.

    TODO: What coordinate system is this?

    For an explanation / illustration of the orbital elements, see:
        https://en.m.wikipedia.org/wiki/Orbital_elements

    TODO: This is Sophia's code, I just copied it here and started to
        document it. No guarantees that it is correct yet.

    Args:
        phases: Orbital phase, must be in the range [0, 1). Basically,
            this is a normalized time variable.
        sma: Semi-major axis of the orbit.
        e: Eccentricity. Must obey 0 < e < 1 for ellipctic orbits.
        aperi: Argument of the periapsis
        lan: Longitude of the ascending node.
        inc: Orbital inclination (where 0° = face-on, 90° = edge-on).

    Returns:
        A numpy array of shape `(len(phases), 3)` containing the
        position of the planet for all `phases`. The columns are
        x, y, z coordinates, respectively.
    """

    # Compute the radius for each phase (and ensure that it is in AU)
    r = (sma * (1 - e**2) / (1 + e * np.cos(2 * np.pi * phases))).to(u.AU)

    # TODO: Where does the following come from?

    # Compute the x, y, z coordinates for each phase
    x = -r * np.cos(2 * np.pi * phases)
    y = r * np.sin(2 * np.pi * phases)
    xyz = np.array([x, y, np.zeros(len(r))]).T

    # Rotate the coordinates to the correct orientation
    rotation_matrix = Rotation.from_euler(
        "ZXZ",
        np.array(
            [
                -lan.to(u.rad).value,
                np.pi / 2 - inc.to(u.rad).value,
                -aperi.to(u.rad).value,
            ]
        ).T,
    )
    xyz = rotation_matrix.apply(xyz)

    return xyz * r.unit


def beta_lim(
    iwa: u.Quantity[u.rad],
    sma: u.Quantity[u.AU],
    e: float,
    aperi: u.Quantity[u.rad],
    lan: u.Quantity[u.rad],
    inc: u.Quantity[u.rad],
    dist: u.Quantity[u.pc]
) -> tuple[u.Quantity[u.deg], u.Quantity[u.deg]]:
    """
    Compute the minimum and maximum beta values for a given IWA.

    TODO: This is Sophia's code, I just copied it here and started to
        document it. No guarantees that it is correct yet.

    Args:
        iwa:
        sma:
        e:
        aperi:
        lan:
        inc:
        dist:

    Returns:
        A tuple containing the minimum and maximum beta values.
    """

    # Define a fine grid of orbital phases and compute x, y, z for each phase
    phases = np.arange(0, 1, 0.00001)
    xyz = xyz_position(phases, sma, e, aperi, lan, inc)

    # TODO: How are x, y, z defined? Where does all this come from?

    # Compute separations in mas
    a = (xyz[:, 1] / dist).to(u.mas, equivalencies=u.dimensionless_angles())
    d = (xyz[:, 2] / dist).to(u.mas, equivalencies=u.dimensionless_angles())

    # Define a mask for the points on the orbit that are outside the IWA,
    # that is, the points that are not occulted by the coronagraph.
    mask = np.sqrt(a**2 + d**2) > iwa

    # Compute beta for all points on the orbit
    beta = np.arccos(
        (xyz[:, 0] / np.linalg.norm(xyz, axis=1))
        .to(u.dimensionless_unscaled)
        .value
    )

    # Compute beta_min and beta_max by applying the mask
    beta_min = np.rad2deg(np.min(beta[mask])) * u.deg
    beta_max = np.rad2deg(np.max(beta[mask])) * u.deg

    return beta_min, beta_max


# -----------------------------------------------------------------------------
# TEST ZONE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    sma = 1 * u.au
    iwa = 300 * u.mas
    e = 0.3
    lan = np.pi / 6 * u.rad
    aperi = 0 * u.rad
    inc = np.pi / 6 * u.rad
    dist = 1.3 * u.pc

    beta_min, beta_max = beta_lim(iwa, sma, e, aperi, lan, inc, dist)
    print(f"beta_min = {beta_min:.2f}")
    print(f"beta_max = {beta_max:.2f}")
