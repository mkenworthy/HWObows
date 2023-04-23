"""
Create Figure 9: Number of planets accessible as a function of IWA,
for different scattering features.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from warnings import catch_warnings, filterwarnings

from matplotlib.lines import Line2D
from scipy import interpolate as si

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import paths
from utils.plotting import set_fontsize, CBF_COLORS
from utils.constants import (
    ETA_EARTH,
    LAMBDA_OVER_D_IN_MAS,
    SCATTERING_FEATURES,
)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

# Read in the target list
df = pd.read_csv(paths.data / "2646_NASA_ExEP_Target_List_HWO_Table.csv")

# The angular separation of the Earth Equivalent Instellation (EEI) in mas
hab_zones = np.array(df["EEIDmas"].values[1:], dtype=float)

# Compute the minimum and maximum scattering angles (pre-computed)
iwa = np.load(paths.data / "iwa_all3.npz")
phase_max = iwa["betamax"]
phase_min = iwa["betamin"]
iwa_list = iwa["iwa"]

# Create a new figure
pad_inches = 0.025
fig, axes = plt.subplots(
    ncols=4,
    figsize=(7 - 2 * pad_inches, 2.5 - 2 * pad_inches),
)

# Add twin axes for the top and right
top_axes = [ax.twiny() for ax in axes]
right_axes = [ax.twinx() for ax in axes]

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

    # Store values for plotting
    lows = []
    meds = []
    highs = []

    for i in range(len(iwa_list)):

        # This helps identify which systems aren't detected at all.
        # There are easier ways, but this is legacy and works
        with catch_warnings():
            filterwarnings("ignore", "invalid value encountered in arccos")
            beta_max = np.degrees(np.arccos(iwa_list[i] / hab_zones))

        # Get data of the histogram
        count, bins_count = np.histogram(
            phase_max[:, :, i].flatten(), bins=np.linspace(90, 180, 20)
        )

        # Finding the PDF of the histogram using count values
        pdf = count / sum(count)

        # Use numpy np.cumsum to calculate the CDF
        # We can also find using the PDF values by looping and adding
        cdf = np.cumsum(pdf)
        normalized_cdf = (1 - cdf) * np.sum(np.isfinite(beta_max))

        # Deal with angles below 90
        these_angles = np.array(SCATTERING_FEATURES[feature_name])
        for m, angle in enumerate(these_angles):
            if angle < 90:
                these_angles[m] = 90 + (90 - angle)

        # Interpolate the inverse CDFs to the specified angles
        interp_cdf = si.interp1d(
            bins_count[1:],
            normalized_cdf,
            fill_value="extrapolate",
            kind="slinear",
        )(these_angles)

        # Append the values to lists for future plotting
        lows.append(interp_cdf[0])
        meds.append(interp_cdf[1])
        highs.append(interp_cdf[2])

    # Plot the lines
    ax.plot(iwa_list, lows, "--", lw=1, color=CBF_COLORS[k])
    ax.plot(iwa_list, meds, "-", lw=1, color=CBF_COLORS[k])
    ax.plot(iwa_list, highs, ":", lw=1, color=CBF_COLORS[k])

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

# Manually create a single legend for all subplots
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

# Adjust spacing
fig.tight_layout(pad=0)
plt.subplots_adjust(wspace=0.1)

# Save the figure
file_name = "nplanets_vs_iwa.pdf"
file_path = paths.figures / file_name
fig.savefig(
    file_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=pad_inches,
)
