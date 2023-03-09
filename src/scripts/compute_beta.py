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

def get_xyz_positions(
    orbital_phases: np.ndarray,
    semimajor_axis: u.Quantity[u.AU],
    eccentricity: u.Quantity[u.one],
    arg_periapsis: u.Quantity[u.rad],
    lon_asc_node: u.Quantity[u.rad],
    inclination: u.Quantity[u.rad],
) -> np.ndarray:
    """
    Take the Keplerian orbital elements and compute the position of the
    planet in x, y, z coordinates, where x is the direction of the
    line of sight, and y and z are in the plane of the orbit.

    For an explanation / illustration of the orbital elements, see:
        https://en.m.wikipedia.org/wiki/Orbital_elements

    Args:
        orbital_phases: Orbital phase, must be in the range [0, 1).
            Basically, this is a normalized time variable.
        semimajor_axis: Semi-major axis of the orbit.
        eccentricity: Eccentricity; 0 < e < 1 for elliptic orbits.
        arg_periapsis: Argument of the periapsis
        lon_asc_node: Longitude of the ascending node.
        inclination: Orbital inclination; 0° = face-on, 90° = edge-on.

    Returns:
        A numpy array of shape `(len(phases), 3)` containing the
        position of the planet for all `phases`. The columns are
        x, y, z coordinates, respectively.
    """

    # Compute the radius for each phase (and ensure that it is in AU)
    r = (
        semimajor_axis * (1 - eccentricity**2) /
        (1 + eccentricity * np.cos(2 * np.pi * orbital_phases))
    ).to(u.AU)

    # Compute the x, y, z coordinates for each phase
    x = -r * np.cos(2 * np.pi * orbital_phases)
    y = r * np.sin(2 * np.pi * orbital_phases)
    xyz = np.array([x, y, np.zeros(len(r))]).T

    # Rotate the coordinates to the correct orientation
    rotation_matrix = Rotation.from_euler(
        "ZXZ",
        np.array(
            [
                -lon_asc_node.to(u.rad).value,
                np.pi / 2 - inclination.to(u.rad).value,
                -arg_periapsis.to(u.rad).value,
            ]
        ).T,
    )
    xyz = rotation_matrix.apply(xyz)

    return xyz * r.unit


def get_beta_min_and_beta_max(
    iwas: list[u.Quantity[u.mas]],
    semimajor_axis: u.Quantity[u.AU],
    eccentricity: u.Quantity[u.one],
    arg_periapsis: u.Quantity[u.rad],
    lon_asc_node: u.Quantity[u.rad],
    inclination: u.Quantity[u.rad],
    distance: u.Quantity[u.pc]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the minimum and maximum beta values for each given IWA.

    Args:
        iwas: List of inner working angles.
        semimajor_axis: Semi-major axis of the orbit.
        eccentricity: Eccentricity; 0 < e < 1 for ellipctic orbits.
        arg_periapsis: Argument of the periapsis.
        lon_asc_node: Longitude of the ascending node.
        inclination: Orbital inclination; 0° = face-on, 90° = edge-on.
        distance: Distance to the target star.

    Returns:
        A 2-tuple `(beta_min, beta_max)`, where `beta_min` and
        `beta_max` are numpy arrays of shape `(len(iwas),)`.
    """

    # If the semi-major axis is given in mas, convert it to AU
    if semimajor_axis.unit == u.mas:
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            semimajor_axis = (semimajor_axis * distance).to(u.au)

    # Define a grid of orbital phases, compute x, y, z for each phase
    orbital_phases = np.arange(0, 1, 0.00001)
    xyz = get_xyz_positions(
        orbital_phases=orbital_phases,
        semimajor_axis=semimajor_axis,
        eccentricity=eccentricity,
        arg_periapsis=arg_periapsis,
        lon_asc_node=lon_asc_node,
        inclination=inclination,
    )

    # Compute on-sky separation for all points on the orbit
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        a = (xyz[:, 1] / distance).to(u.mas)
        d = (xyz[:, 2] / distance).to(u.mas)
    separations = np.sqrt(a**2 + d**2)

    # Compute beta for all points on the orbit
    beta = np.arccos(
        (xyz[:, 0] / np.linalg.norm(xyz, axis=1))
        .to(u.dimensionless_unscaled)
        .value
    )

    # Hold beta_min and beta_max for each IWA
    beta_min = np.zeros(len(iwas)) * u.deg
    beta_max = np.zeros(len(iwas)) * u.deg

    # Loop over all IWA values is more efficient than computing than calling
    # this function multiple times for each IWA, because we can re-use `xyz`
    for i, iwa in enumerate(iwas):

        # Define a mask for the points on the orbit that are outside the IWA,
        # that is, the points that are not occulted by the coronagraph.
        mask = separations > iwa

        # Compute beta_min and beta_max by applying the mask
        try:
            beta_min[i] = np.rad2deg(np.min(beta[mask])) * u.deg
        except ValueError:
            beta_min[i] = 180 * u.deg
        try:
            beta_max[i] = np.rad2deg(np.max(beta[mask])) * u.deg
        except ValueError:
            beta_max[i] = 0 * u.deg

    return beta_min, beta_max


# -----------------------------------------------------------------------------
# DEMO USAGE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Parameters for HD 4391 (from target list)
    semimajor_axis = 63.9 * u.mas
    distance = 15 * u.pc

    # Instrument parameters
    wavelength = 600 * u.nm
    diameter = 6 * u.m
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        iwas = (np.array([1, 2, 3]) * wavelength / diameter).to(u.mas)

    # Orbital parameters (need to be sampled randomly)
    eccentricity = 0.3 * u.one
    lon_asc_node = np.pi / 6 * u.rad
    arg_periapsis = 0 * u.rad
    inclination = np.pi / 2 * u.rad

    # Compute beta_min and beta_max
    beta_min, beta_max = get_beta_min_and_beta_max(
        iwas=iwas,
        semimajor_axis=semimajor_axis,
        eccentricity=eccentricity,
        arg_periapsis=arg_periapsis,
        lon_asc_node=lon_asc_node,
        inclination=inclination,
        distance=distance,
    )

    print(f"iwas     = {np.around(iwas, 2)}")
    print(f"beta_min = {np.around(beta_min, 2)}")
    print(f"beta_max = {np.around(beta_max, 2)}")
