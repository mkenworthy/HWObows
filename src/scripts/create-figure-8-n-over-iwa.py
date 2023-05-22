"""
Create Figure 8: Number of planets accessible as a function of IWA for
different scattering features. This script also creates Table 3, which
contains the number of systems accessible as a function of IWA.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from matplotlib.lines import Line2D
from tabulate import tabulate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.paths import (
    data as data_dir,
    figures as figures_dir,
    tables as tables_dir,
)
from utils.plotting import set_fontsize, CBF_COLORS
from utils.constants import (
    ETA_EARTH,
    LAMBDA_OVER_D_IN_MAS,
    PHASE_ANGLES_OF_FEATURES as SCATTERING_FEATURES
)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\nCREATE FIGURE 8: N_SYSTEMS OVER IWA\n")

    # -------------------------------------------------------------------------
    # Read in the data
    # -------------------------------------------------------------------------

    print("Reading in the data...", end=" ", flush=True)

    # Read in the target list
    file_path = data_dir / "2646_NASA_ExEP_Target_List_HWO_Table.csv"
    df = pd.read_csv(file_path, header=[0, 1])

    # The angular separation of the Earth Equivalent Instellation (EEI) in mas
    hab_zones = np.array(df["EEIDmas"].values, dtype=float)

    # Compute the minimum and maximum scattering angles (pre-computed)
    file_path = data_dir / "eccentric-orbits.npz"
    iwa = np.load(file_path)
    phase_max = iwa["betamax"]
    phase_min = iwa["betamin"]
    iwa_list = iwa["iwa"]

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------

    print("Creating the plot...", end=" ", flush=True)

    # Create a new figure
    pad_inches = 0.025
    fig, axes = plt.subplots(
        ncols=4,
        figsize=(7 - 2 * pad_inches, 2.5 - 2 * pad_inches),
    )

    # Add twin axes for the top and right
    top_axes = [ax.twiny() for ax in axes]
    right_axes = [ax.twinx() for ax in axes]

    # Keep track of the expected number of planets that are accessible at the
    # peak phase angle; we will use this to generate Table 3 automatically
    expected_numbers = []

    # Create a subplot for each scattering feature
    for k, (ax, top_ax, right_ax, feature_name) in enumerate(
        zip(axes, top_axes, right_axes, SCATTERING_FEATURES.keys())
    ):
        # Set the fontsize for all axes
        set_fontsize(ax=ax, fontsize=6)
        set_fontsize(ax=top_ax, fontsize=6)
        set_fontsize(ax=right_ax, fontsize=6)

        # Add the title
        ax.set_title(feature_name, fontsize=8, pad=10)

        for i, ls in enumerate([":", "-", "--"]):

            # TODO: Fix the plot for Rayleigh scattering, where the  start and
            #  end are on different sides of quadrature.
            x = (
                (phase_min <= SCATTERING_FEATURES[feature_name][i])
                * (SCATTERING_FEATURES[feature_name][i] <= phase_max)
            )
            y = np.sum(np.nanmean(x, axis=0), axis=0)

            ax.plot(iwa_list, y, ls=ls, lw=1, color=CBF_COLORS[k])

            if i == 1:
                for iwa in [21, 41, 62, 83]:
                    idx = np.where(iwa_list == iwa)[0][0]
                    expected_numbers.append(
                        {
                            "iwa": iwa,
                            "feature": feature_name,
                            "n": y[idx],
                        }
                    )

        # Add a grid (using the bottom x-axis and left y-axis)
        ax.grid(ls="--", lw=0.5, color="gray", alpha=0.2)

        # Set the options for the bottom x-axis
        ax.set_xlim(15, 125)
        ax.set_xticks(np.arange(20, 120 + 1, 20))
        ax.set_xlabel(r"IWA (in mas)")

        # Set the options for the top x-axis
        top_ax.set_xticks(np.arange(1, 7) * LAMBDA_OVER_D_IN_MAS)
        top_ax.set_xticklabels(np.arange(1, 7))
        top_ax.set_xlim(ax.get_xlim())
        top_ax.set_xlabel(r"IWA (in $\lambda / D$)")

        # Set the options for the left y-axis
        ax.set_ylim(0, 170)
        ax.set_yticks(np.arange(0, 180, 20))
        if k == 0:
            ax.set_yticklabels(np.arange(0, 180, 20))
            ax.set_ylabel("Number of systems")
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        # Set the options for the right y-axis
        right_ax.set_ylim(ax.get_ylim())
        if k == 3:
            right_ax.set_yticks(axes[0].get_yticks())
            right_ax.set_yticklabels(
                ["{:.0f}".format(y * ETA_EARTH) for y in axes[0].get_yticks()]
            )
            right_ax.set_ylabel(
                rf"Number of planets for $\eta_\oplus$={ETA_EARTH:.2f}"
            )
        else:
            right_ax.set_yticks([])

        # Set spacing between ticks and ticklabels
        ax.tick_params(axis="both", which="major", pad=1)
        top_ax.tick_params(axis="x", which="major", pad=1)
        right_ax.tick_params(axis="y", which="major", pad=1)

    # -------------------------------------------------------------------------
    # Manually create a single legend for all subplots
    # -------------------------------------------------------------------------

    handles = [
        Line2D([0], [0], lw=0, label="Legend:"),
        Line2D([0], [0], ls="--", color="k", lw=1, label="Start of feature"),
        Line2D([0], [0], ls="-", color="k", lw=1, label="Peak of feature"),
        Line2D([0], [0], ls=":", color="k", lw=1, label="End of feature"),
    ]
    fig.legend(
        handles=handles,
        ncols=4,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.025),
        frameon=False,
        fontsize=6,
    )

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Print and export table with the expected numbers
    # -------------------------------------------------------------------------

    # Create and format a dataframe in a way that is compatible with tabulate
    df = (
        pd.DataFrame(expected_numbers)
        .groupby(['iwa', 'feature'])
        .mean()
        .round({"n": 0})
        .astype({"n": int})
        .unstack(level=0)
    )

    # Define the column names
    headers = ["Feature"] + [f"{iwa:.0f} mas" for iwa in (21, 41, 62, 83)]

    # Print the table to the terminal
    table = tabulate(df, headers=headers, tablefmt="simple")
    print("\n" + table + "\n")

    # Save the table to a LaTeX file (this will be Table 3)
    print(f"Saving table to *.tex file...", end=" ", flush=True)
    table = tabulate(df, headers=headers, tablefmt="latex_booktabs")
    file_path = tables_dir / "table-3-expected-number.tex"
    with open(file_path, "w") as tex_file:

        # Add warning at the start
        tex_file.write(
            "\n% ATTENTION:\n"
            "% THIS FILE IS GENERATED AUTOMATICALLY BY "
            "`create-figure-8-n-over-iwa.py`\n"
            "% PLEASE DO NOT EDIT MANUALLY\n\n"
        )

        # Add the table line by line; add multicolumn for the top row
        for line in table.split("\n"):
            tex_file.write(line + "\n")
            if "toprule" in line:
                tex_file.write(
                    "             & \multicolumn{4}{c}{Inner Working Angle "
                    "(IWA)} \\\\\n"
                    "\cmidrule(lr){2-5}\n"
                )

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Save the figure
    # -------------------------------------------------------------------------

    # Adjust spacing
    fig.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0.1)

    print(f"Saving plot to PDF...", end=" ", flush=True)

    # Save the figure
    file_name = "figure-8-n-over-iwa.pdf"
    file_path = figures_dir / file_name
    fig.savefig(
        file_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )

    print("Done!", flush=True)
