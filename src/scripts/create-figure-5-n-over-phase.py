"""
Create Figure 5: Expected number of accessible planets over phase angle
for different inner working angles (both circular and eccentric).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from warnings import catch_warnings, filterwarnings

import time

from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from tabulate import tabulate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import paths
from utils.constants import (
    lambda_over_d_in_mas,
    ETA_EARTH,
    RANDOM_SEED,
    PHASE_ANGLES_OF_FEATURES,
)
from utils.plotting import set_fontsize, CBF_COLORS
from utils.samplers import sample_i


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print("\n" + 80 * "-")
    print("\nCREATE FIGURE 5: N_SYSTEMS OVER PHASE ANGLE\n")

    # Start timer
    script_start = time.time()

    # Get IWA values (in mas) for 1, 2, 3, 4 lambda / D at 600 nm and round
    # to the nearest integer (because `create-data-for-eccentric-orbits.py`
    # only computes the beta values for integer values of the IWA in mas).
    # This should be [21, 41, 62, 83] mas.
    n_lambda_over_d_in_mas = np.around(
        lambda_over_d_in_mas().value * np.array([1, 2, 3, 4]), 0
    )

    # -------------------------------------------------------------------------
    # Read in the data
    # -------------------------------------------------------------------------

    print("Reading in the data...", end=" ", flush=True)

    # Read in the target list; select habitable zones (in mas)
    df = pd.read_csv(paths.data / "2646_NASA_ExEP_Target_List_HWO_Table.csv")
    hz = np.array(df["EEIDmas"].values[1:], dtype=float)  # in mas

    # Read in the maximum and minimum scattering angles as calculated by the
    # dynamical simulations. This set of files only has 4 IWAs.
    iwa = np.load(paths.data / "eccentric-orbits.npz")
    phase_max = iwa["betamax"]  # in degrees
    phase_min = iwa["betamin"]  # in degrees
    iwa_list = iwa["iwa"]  # in mas

    # Note: We only use phase_max for plotting.
    # For a given system, phase_max and phase_min might differ (for eccentric
    # orbits), but the distributions mirror each other (about 90 degrees) when
    # you average over enough orbits.

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Run simulations for different inclinations (circular orbits)
    # -------------------------------------------------------------------------

    print("Simulating different inclinations...", end=" ", flush=True)

    # Fix the random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # The number of random inclinations to generate per system
    n_sims = 1_000

    # Beta is an intermediate parameter used here.
    # The maximum scattering angle is equal to 90+beta degrees
    # The minimum scattering angle is equal to 90-beta degrees
    beta = np.zeros(shape=(n_sims, len(hz), len(iwa_list)))

    # Loop over all targets to run the simulations
    for i, target_hz in enumerate(hz):

        # Generate random inclinations such that cos(i) is uniform over [0, 1]
        inclinations = sample_i(n_sims)

        # For each target, the absolute maximum scattering angle accessible
        # will be when the _projected_ semi-minor axis of the orbit just
        # touches the coronagraph inner working angle.
        for j, iwa in enumerate(iwa_list):

            with catch_warnings():
                filterwarnings("ignore", "invalid value encountered in arccos")
                beta_max0 = np.degrees(np.arccos(iwa / target_hz))

            # If beta_max is nan then the planet is not detectable at this IWA
            if np.isnan(beta_max0):
                beta[:, i, j] = np.nan

            # If the inclination is greater than or equal to betamax (i.e., if
            # it is touching the coronagraph edge or more), then beta = betamax
            else:
                beta[:, i, j] = np.clip(inclinations, 0, beta_max0)

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------

    print("Creating the plot...", end=" ", flush=True)

    # Define locations and rotations for labeling the IWA
    label_locations = [[140, 115], [150, 55], [120, 115], [110, 92]]
    label_rotations = [-12, -25, -15, -10]

    # Define bins for np.histogram
    bins = np.linspace(90, 180, 900)

    # Create a new figure
    # These numbers where carefully hand-tuned to ensure that the output has
    # exactly the right width (7 inches) for the paper when saved as a PDF.
    pad_inches = 0.025
    fig, ax = plt.subplots(figsize=(7.05 - 2 * pad_inches, 2 - 2 * pad_inches))

    # Set up plot options
    ax.set_ylabel("Number of accessible systems")
    ax.set_xlabel("Phase angle (in degrees)")
    ax.set_yticks(np.arange(0, 176, 25))
    ax.set_xlim(90, 180)
    ax.set_ylim(0, 175)
    ax.grid(
        ls='--',
        alpha=0.3,
        color='k',
        lw=0.5,
        dash_capstyle='round',
        dashes=(0.05, 2.5),
    )

    # Add a second x-axis at the top
    top_ax = ax.secondary_xaxis(
        location="top",
        functions=(lambda x: 180 - x, lambda x: 180 - x)
    )
    top_ax.set_xlabel("Phase angle (in degrees)")

    # Add a second y-axis on the right
    right_ax = ax.secondary_yaxis(
        location="right",
        functions=(lambda x: x * ETA_EARTH, lambda x: x / ETA_EARTH)
    )
    right_ax.set_yticks(np.arange(0, 176, 6))
    right_ax.set_ylabel(
        rf"Number of planets for $\eta_\oplus$={ETA_EARTH:.2f}"
    )

    # Set up the font size
    set_fontsize(ax, 6)
    set_fontsize(top_ax, 6)
    set_fontsize(right_ax, 6)

    # Manually set up a label for circular vs. eccentric orbits
    handles = [
        Line2D([0], [0], ls="-", color="k", lw=1, label="Circular orbits"),
        Line2D([0], [0], ls="--", color="k", lw=1, label="Eccentric orbits"),
    ]
    legend = ax.legend(
        handles=handles,
        loc='upper right',
        fontsize=6,
    )
    legend.get_frame().set_alpha(0.85)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0)

    # Store results for table
    eccentric = []
    circular = []

    # Define a grid of phase angles on which to evaluate the expected number
    # of accessible planets
    phase_angles = np.linspace(90, 180, 360)

    # Loop over IWA to plot a line for each one
    for i, iwa in enumerate(n_lambda_over_d_in_mas):

        # Get the index that belongs to the current IWA (since we are pulling
        # the data from a file that has all IWA values: 20, 21, ..., 120)
        iwa_idx = np.where(np.isclose(iwa_list, iwa))[0][0]

        # ---------------------------------------------------------------------
        # First the dynamical betas (eccentric orbits)
        # ---------------------------------------------------------------------

        # For each phase angle, compute the expected number of planets that
        # are accessible at that phase angle
        n_accessible = []
        for phase_angle in phase_angles:

            # For each planet and each simulation:
            # Is the current phase angle accessible at the current IWA?
            accessible = (
                (phase_angle >= phase_min[:, :, iwa_idx])
                * (phase_angle <= phase_max[:, :, iwa_idx])
            ).astype(float)

            # Compute expected number of accessible planets at this phase angle
            # by averaging over simulations and summing over targets
            n_expected = np.sum(np.mean(accessible, axis=0))
            n_accessible.append(n_expected)

        # Plot the results
        ax.plot(
            phase_angles,
            n_accessible,
            lw=1,
            ls="--",
            color=CBF_COLORS[i]
        )

        # Get number of accessible planets at the phase angle at which our 4
        # features have their respective peaks and store the results. Keep in
        # mind that we only computed the curve from 90 to 180 degrees, but we
        # can assume that it is symmetric around 90 degrees.
        f = interp1d(phase_angles, n_accessible)
        for key, (_, angle, _) in PHASE_ANGLES_OF_FEATURES.items():
            angle = 180 - angle if angle < 90 else angle
            n = np.around(f(angle), 2)
            eccentric.append({"iwa": iwa, "feature": key, "n": n})

        # ---------------------------------------------------------------------
        # Now the circular case (these ones are actually betas)
        # ---------------------------------------------------------------------

        # For each phase angle, compute the expected number of planets that
        # are accessible at that phase angle
        n_accessible = []
        for phase_angle in phase_angles:

            # For each planet and each simulation:
            # Is the current phase angle accessible at the current IWA?
            accessible = (
                (phase_angle >= beta[:, :, iwa_idx] - 90)
                * (phase_angle <= beta[:, :, iwa_idx] + 90)
            ).astype(float)

            # Compute expected number of accessible planets at this phase angle
            # by averaging over simulations and summing over targets
            n_expected = np.sum(np.mean(accessible, axis=0))
            n_accessible.append(n_expected)

        # Plot the results
        ax.plot(
            phase_angles,
            n_accessible,
            lw=1,
            ls="-",
            color=CBF_COLORS[i]
        )

        # Add the labels for the line
        ax.text(
            x=label_locations[i][0],
            y=label_locations[i][1],
            s=rf"IWA = {iwa:.0f} mas",
            color=CBF_COLORS[i],
            rotation=label_rotations[i],
            fontsize=6,
            ha="center",
            va="center",
            bbox=dict(
                fc='white',
                ec='none',
                alpha=0.85,
                boxstyle='round,pad=0.01',
            ),
        )

        # Get number of accessible planets at the phase angle at which our 4
        # features have their respective peaks and store the results. Keep in
        # mind that we only computed the curve from 90 to 180 degrees, but we
        # can assume that it is symmetric around 90 degrees.
        f = interp1d(phase_angles, n_accessible)
        for key, (_, angle, _) in PHASE_ANGLES_OF_FEATURES.items():
            angle = 180 - angle if angle < 90 else angle
            n = np.around(f(angle), 2)
            circular.append({"iwa": iwa, "feature": key, "n": n})

    print("Done!\n", flush=True)

    # -------------------------------------------------------------------------
    # Print tables with the expected numbers
    # -------------------------------------------------------------------------

    for data, label in [(eccentric, "Eccentric"), (circular, "Circular")]:
        print(f"\n{label} orbits:")
        print(
            tabulate(
                pd.DataFrame(data)
                .groupby(['iwa', 'feature'])
                .mean()
                .round({"n": 0})
                .astype({"n": int})
                .unstack(level=0),
                headers=(
                    ["Feature"]
                    + [f"{iwa:.0f} mas" for iwa in n_lambda_over_d_in_mas]
                ),
                tablefmt="simple",  # latex_booktabs
            )
        )
        print()

    # -------------------------------------------------------------------------
    # Save the figure
    # -------------------------------------------------------------------------

    fig.tight_layout(pad=0)

    print(f"\nSaving plot to PDF...", end=" ", flush=True)

    file_path = paths.figures / "figure-5-n-over-phase.pdf"
    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!")
    print("\n" + 80 * "-" + "\n")
