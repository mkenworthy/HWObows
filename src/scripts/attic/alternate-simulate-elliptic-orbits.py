"""
Script to run Monte Carlo simulations of elliptic orbits.

This is an alternative version of the data generation script that was
NOT actually used in the end. It is kept here for reference.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import time

from tqdm.auto import tqdm

import astropy.units as u
import h5py
import numpy as np
import pandas as pd

from compute_beta import matts_approach
from paths import data as data_dir
from samplers import sample_e, sample_i


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    script_start = time.time()
    print("\nSIMULATE ELLIPTIC ORBITS\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hd-name", type=str, required=True)
    parser.add_argument("--n-simulations", type=int, default=10_000)
    parser.add_argument("--wavelength", type=float, default=600)
    parser.add_argument("--diameter", type=float, default=6)
    parser.add_argument("--random-seed", type=float, default=42)
    args = parser.parse_args()
    args.hd_name = args.hd_name.strip()

    # Set random seed
    np.random.seed(args.random_seed)

    # Load target list and select target star
    file_path = data_dir / "2646_NASA_ExEP_Target_List_HWO_Table.csv"
    target_list = pd.read_csv(file_path, header=1)
    idx = target_list["hd_name"].str.strip() == args.hd_name.strip()
    target_star = target_list[idx]
    if not len(target_star) == 1:
        raise ValueError(f'Failed to find "{args.hd_name}" in target list!')

    # Get stellar parameters from target list
    semimajor_axis = float(target_star.st_eei_orbsep) * u.au
    distance = float(target_star.sy_dist) * u.pc

    print("Target star:     ", args.hd_name)
    print("Semi-major axis: ", np.around(semimajor_axis, 2))
    print("Stellar distance:", np.around(distance, 2), "\n")

    # Compute inner working angles
    wavelength = args.wavelength * u.nm
    diameter = args.diameter * u.m
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        iwas = (np.array([1, 2, 3]) * wavelength / diameter).to(u.mas)

    print("IWAs:", np.around(iwas, 2), "\n", flush=True)

    # Set up array to store results
    results = {
        "eccentricity": np.full(args.n_simulations, np.nan),
        "inclination": np.full(args.n_simulations, np.nan),
        "arg_periapsis": np.full(args.n_simulations, np.nan),
        "lon_asc_node": np.full(args.n_simulations, np.nan),
        "beta_min": np.full((args.n_simulations, iwas.size), np.nan),
        "beta_max": np.full((args.n_simulations, iwas.size), np.nan),
    }

    # Compute beta_min and beta_max
    print("Simulating orbits:", flush=True)
    for n in tqdm(list(range(args.n_simulations)), ncols=80):

        # Randomly sample parameters for elliptic orbit
        eccentricity = float(sample_e()) * u.one
        inclination = (float(sample_i()) * u.deg).to(u.rad)
        arg_periapsis = float(np.random.uniform(0, 2 * np.pi)) * u.rad
        lon_asc_node = float(np.random.uniform(0, 2 * np.pi)) * u.rad

        # Compute beta_min and beta_max
        beta_min, beta_max = matts_approach(
            iwas=iwas,
            semimajor_axis=semimajor_axis,
            eccentricity=eccentricity,
            arg_periapsis=arg_periapsis,
            lon_asc_node=lon_asc_node,
            inclination=inclination,
            distance=distance,
        )

        # Store results
        results["eccentricity"][n] = eccentricity.value
        results["inclination"][n] = inclination.value
        results["arg_periapsis"][n] = arg_periapsis.value
        results["lon_asc_node"][n] = lon_asc_node.value
        results["beta_min"][n] = beta_min
        results["beta_max"][n] = beta_max

    # Create output directory
    output_dir = data_dir / "elliptic-orbits"
    output_dir.mkdir(exist_ok=True)

    # Save results to file
    print("\nSaving results to file...", end=" ", flush=True)
    file_name = args.hd_name.lower().replace(" ", "-") + ".hdf"
    with h5py.File(output_dir / file_name, "w") as f:
        f.attrs["hd_name"] = args.hd_name
        f.attrs["wavelength"] = args.wavelength
        f.attrs["diameter"] = args.diameter
        f.attrs["random_seed"] = args.random_seed
        f.attrs["iwas"] = iwas.value
        f.attrs["semimajor_axis"] = semimajor_axis.value
        f.attrs["distance"] = distance.value
        for key, value in results.items():
            f.create_dataset(name=key, data=value)
    print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
