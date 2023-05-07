"""
Create Figure 5: CDF of number of planets over accessible phase angle
for different inner working angles (both circular and elliptical).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from warnings import catch_warnings, filterwarnings

from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import paths
from utils.constants import ETA_EARTH, RANDOM_SEED
from utils.plotting import set_fontsize, CBF_COLORS
from utils.samplers import sample_i


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\nCREATE FIGURE 5: N_SYSTEMS OVER PHASE ANGLE\n")

    # -------------------------------------------------------------------------
    # Read in the data
    # -------------------------------------------------------------------------

    print("Reading in the data...", end=" ", flush=True)

    # Read in the target list; select habitable zones (in mas)
    df = pd.read_csv(paths.data / "2646_NASA_ExEP_Target_List_HWO_Table.csv")
    hz = np.array(df["EEIDmas"].values[1:], dtype=float)  # in mas

    # Read in the maximum and minimum scattering angles as calculated by the
    # dynamical simulations. This set of files only has 4 IWAs.
    iwa = np.load(paths.data / "iwa_all.npz")
    phase_max = iwa["betamax"]
    phase_min = iwa["betamin"]
    iwa_list = iwa["iwa"]  # in mas

    # Note: We only use phase_max for plotting.
    # For a given system, phase_max and phase_min might differ (for elliptical
    # orbits), but the distributions mirror each other (about 90 degrees) when
    # you average over enough orbits.

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Run simulations for different inclinations
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
    for i in range(len(hz)):

        # Generate random inclinations such that cos(i) is uniform over [0, 1]
        inclinations = sample_i(n_sims)

        # For each target, the absolute maximum scattering angle accessible
        # will be when the _projected_ semi-minor axis of the orbit just
        # touches the coronagraph inner working angle.
        for j, iwa in enumerate(iwa_list):

            with catch_warnings():
                filterwarnings("ignore", "invalid value encountered in arccos")
                beta_max0 = np.degrees(np.arccos(iwa / hz[i]))

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

    # Manually set up a label for circular vs. elliptical orbits
    handles = [
        Line2D([0], [0], ls="-", color="k", lw=1, label="Circular orbits"),
        Line2D([0], [0], ls="--", color="k", lw=1, label="Elliptical orbits"),
    ]
    legend = ax.legend(
        handles=handles,
        loc='upper right',
        fontsize=6,
    )
    legend.get_frame().set_alpha(0.85)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0)

    # Loop over IWA to plot a lin
    for i, iwa in enumerate(iwa_list):

        # ---------------------------------------------------------------------
        # First the dynamical betas
        # ---------------------------------------------------------------------

        # Note: Although we're calling them betas, really they're the maximum
        # scattering angles)

        # Getting data of the histogram bins to build the CDF
        count, bins_count = np.histogram(phase_max[:, :, i], bins=bins)

        # Find the PDF of the histogram using count values
        pdf = count / sum(count)

        # Use np.cumsum to calculate the CDF
        # We can also find using the PDF values by looping and adding
        cdf = np.cumsum(pdf)

        # Really, we want to plot 1-CDF to show the number of systems with a
        # maximum scattering angle less than the bin. We multiply by the number
        # of detected systems (i.e., where the beta_maxes are finite) and then
        # normalize by the number of simulations.
        n_systems = (
            (1 - cdf)
            * len(np.where(np.isfinite(phase_max[:, :, i]))[0])
            / phase_max[:, :, i].shape[0]
        )

        # Plot the result over the phase angle bins
        ax.plot(
            bins_count[1:],
            n_systems,
            color=CBF_COLORS[i],
            linestyle="--",
            linewidth=1,
        )

        # ---------------------------------------------------------------------
        # Now the circular case (these ones are actually betas)
        # ---------------------------------------------------------------------

        # This is the analytical beta_max for a given IWA
        with catch_warnings():
            filterwarnings("ignore", "invalid value encountered in arccos")
            beta_max2 = np.degrees(np.arccos(iwa / hz))

        # We need to add 90 to get the maximum scattering angle
        count, bins_count = np.histogram(
            beta[:, :, i].flatten() + 90, bins=bins
        )
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)

        # Here multiplying by the beta_max takes into account the
        # non-detections and the total number of systems
        n_systems = (1 - cdf) * np.sum(np.isfinite(beta_max2))

        # Plot the result over the phase angle bins
        ax.plot(
            bins_count[1:],
            n_systems,
            color=CBF_COLORS[i],
            linestyle="-",
            linewidth=1,
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

    print("Done!", flush=True)
    print(f"Saving plot to PDF...", end=" ", flush=True)

    # Save the figure
    fig.tight_layout(pad=0)
    file_path = paths.figures / "figure-5-n-over-phase.pdf"
    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )

    print("Done!", flush=True)
