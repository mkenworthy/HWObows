"""
Compute phi_max (up to an offset of 90 degrees) at an IWA of 3 lambda/D
for each system in the target list assuming an edge-on, circular orbit.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from warnings import catch_warnings, filterwarnings

import astropy.units as u
import pandas as pd
import numpy as np

from utils import paths
from utils.constants import DIAMETER, WAVELENGTH


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\nCREATE DATA FOR CIRCULAR ORBITS\n")

    # Load the target list and select the EEIDmas column as the habitable zones
    print("Loading target list...", end=" ", flush=True)
    df = pd.read_csv(paths.data / "2646_NASA_ExEP_Target_List_HWO_Table.csv")
    hz = np.array(df["EEIDmas"].values[1:], dtype=float) * u.mas
    print("Done!")

    # Compute phi_max for 3 lambda/D (up to an offset of 90 degrees)
    # This is essentially Eq. (4) from the paper, where i=90° (i.e., cos(i)=0).
    # Note 1: We use astropy.units to handle all conversions instead of using
    #   the small angle approximation with the "magic number" 206265.
    # Note 2: We filter out the warning that occurs when the argument of arccos
    #   is larger than 1, in which case we set phi_max to NaN. (This happens
    #   when the entire orbit is behind the coronagraph.)
    print("Computing phi_max for IWA = 3 lambda/D...", end=" ", flush=True)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        with catch_warnings():
            filterwarnings("ignore", "invalid value encountered in arccos")
            phi_max = np.degrees(np.arccos(3 * WAVELENGTH / DIAMETER / hz))
    print("Done!")

    # Save the results as a CSV file
    print("Saving results to CSV file...", end=" ", flush=True)
    # noinspection PyUnresolvedReferences
    np.savetxt(
        fname=paths.data / "phi_max_3_lambda_over_d.csv",
        X=phi_max.value,
        delimiter=",",
    )
    print("Done!\n")
