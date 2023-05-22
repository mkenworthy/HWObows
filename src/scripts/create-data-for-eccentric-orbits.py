"""
Simulate 1,000 eccentric orbits for every planet in the target list.
"""

from warnings import catch_warnings, filterwarnings

from astropy.io import ascii
from joblib import Parallel, delayed
from tqdm import tqdm

import astropy.units as u
import numpy as np

from utils.constants import RANDOM_SEED
from utils.phases import kep3d, xyztoscatter
from utils.samplers import sample_e, sample_i
from utils import paths


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

# noinspection PyUnresolvedReferences
def simulate_orbit(
    a: float,
    e: float,
    i: float,
    w: float,
    anode: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate an eccentric orbit and return the minimum and maximum
    scattering angles for each IWA.

    Args:
        a: The semi-major axis of the orbit (in mas).
        e: The eccentricity of the orbit.
        i: The inclination of the orbit (in degrees).
        w: The argument of periapsis of the orbit (in degrees).
        anode: The longitude of the ascending node of the orbit
            (in degrees).

    Returns:
        scatmin: The minimum scattering angle for each IWA (in degrees).
        scatmax: The maximum scattering angle for each IWA (in degrees).
    """

    P = 1.0 * u.year
    tperi = 2050.0 * u.year

    # Define the epochs to sample
    epochs = np.linspace(tperi, tperi + P, n_epochs, endpoint=True)

    # Calculate the orbit
    _, _, Xsa, Ysa, Zsa, _, _, _ = kep3d(epochs, P, tperi, a, e, i, w, anode)

    # Calculate current projected separation
    rho = np.sqrt(Xsa * Xsa + Ysa * Ysa)

    # Calculate scattering angle
    scang = xyztoscatter(Xsa, Ysa, Zsa)

    # For each IWA, set points on orbit that are behind the coronagraph to True
    notvisible = rho[:, np.newaxis] < iwa

    # Define dummy array in preparation for max and min
    ttt = scang[:, np.newaxis] * np.ones(iwa.size)

    # For each IWA column, set the points behind the mask as NaN
    ttt[notvisible] = np.nan

    # Get the largest and smallest scattering angles
    # If the entire orbit is all behind the coronagraph, return NaNs
    with catch_warnings():
        filterwarnings("ignore", "All-NaN axis encountered")
        scatmax = np.nanmax(ttt, axis=0).to(u.deg)
        scatmin = np.nanmin(ttt, axis=0).to(u.deg)

    return scatmin, scatmax


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\nSIMULATE ECCENTRIC ORBITS (IN PARALLEL)\n")

    # Fix the random seed to ensure reproducibility
    np.random.seed(RANDOM_SEED)

    # Define number of planet orbits to simulate for each star and number of
    # epochs to sample along each orbit ("resolution of the orbit")
    n_orbits = 1_000
    n_epochs = 1_000

    print(f"Number of orbits to simulate: {n_orbits:,}")
    print(f"Number of epochs to sample:   {n_epochs:,}\n")

    # Select values for the inner working angle (IWA) to simulate (in mas)
    # We simulate IWAs of 20 to 120 mas in steps of 1 mas (101 values) so that
    # we can use them both for Figure 5 and Figure 8.
    iwa = np.arange(20, 121)
    with np.printoptions(threshold=1):
        print(f"IWA values (in mas): {iwa}\n")

    # Read in the stellar target list
    print("Reading in stellar target list...", end=" ")
    file_name = "2646_NASA_ExEP_Target_List_HWO_Table.csv"
    targets = ascii.read(paths.data / file_name, header_start=1, data_start=2)
    n_stars = len(targets)
    print(f"Done! ({n_stars} stars)\n")

    # Set up output arrays
    betamin = np.zeros((n_orbits, n_stars, iwa.size))
    betamax = np.zeros((n_orbits, n_stars, iwa.size))

    # Loop over all the stars in the target list
    print("Simulating orbits for stars in target list:")
    for idx_star, target in tqdm(enumerate(targets), total=n_stars, ncols=80):

        # Habitable zone distance (in mas)
        hz = target["st_eei_angsep"]

        # Draw orbital parameters from the prior
        esamp = sample_e(n_orbits)
        isamp = sample_i(n_orbits) * u.deg
        anodesamp = np.random.random_sample((n_orbits,)) * 360 * u.deg
        wsamp = np.random.random_sample((n_orbits,)) * 360 * u.deg

        # Simulate the orbits in parallel
        # This returns a list of length `n_planets`, each element of which is
        # a 2-tuple `(scatmin, scatmax)`. Both `scatmin` and `scatmax` are
        # arrays of length `iwa.size`.
        results = Parallel(n_jobs=-1)(
            delayed(simulate_orbit)(a, e, i, w, anode)
            for a, e, i, w, anode in zip(
                np.ones(n_orbits) * hz, esamp, isamp, wsamp, anodesamp
            )
        )

        # Unpack the results into the output arrays
        for idx_orbit, (scatmin, scatmax) in enumerate(results):
            betamin[idx_orbit, idx_star, :] = scatmin
            betamax[idx_orbit, idx_star, :] = scatmax

    file_name = "eccentric-orbits.npz"
    print(f"\nSaving data to file {file_name}...", end=" ")
    np.savez_compressed(
        file=paths.data / file_name,
        iwa=iwa,
        betamax=betamax,
        betamin=betamin,
    )
    print("Done!\n")
