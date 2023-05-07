"""
Create Figure 4: Scatter plot of the target list color-coded by the
degrees from quadrature when assuming edge-on, circular orbits.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.constants import PHASE_ANGLES_OF_FEATURES
from utils.plotting import set_fontsize
from utils import paths


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\nCREATE FIGURE 4: SCATTERPLOT\n")

    # -------------------------------------------------------------------------
    # Load the data
    # -------------------------------------------------------------------------

    print("Loading data...", end=" ", flush=True)

    # Load the target list
    file_path = paths.data / "2646_NASA_ExEP_Target_List_HWO_Table.csv"
    data = pd.read_csv(file_path, header=1)

    # Load the beta values
    file_path = paths.data / "phi_max_3_lambda_over_d.csv"
    beta_values = np.loadtxt(file_path)

    print("Done!")

    # -------------------------------------------------------------------------
    # Create the scatter plot
    # -------------------------------------------------------------------------

    print("Creating scatter plot...", end=" ", flush=True)

    # Define a function to map a quantity (e.g., angular separation)
    # onto a marker size
    def quantity_to_ms(q: float) -> float:
        return 5 * (1 + q / 500)

    # Define a mapping of quantity (e.g., beta value) to a color
    norm = mpl.colors.Normalize(vmin=0, vmax=90)
    cmap = mpl.colormaps["viridis_r"]

    # Define a function to map a quantity (e.g., beta value) onto a color
    def quantity_to_color(q: float) -> Any:
        return cmap(norm(q))

    # Create a new figure
    # These numbers where carefully hand-tuned to ensure that the output has
    # exactly the right width (7 inches) for the paper when saved as a PDF.
    pad_inches = 0.025
    fig, ax = plt.subplots(figsize=(7.02 - 2 * pad_inches, 2 - 2 * pad_inches))

    # Set up the font size
    set_fontsize(ax, 6)

    # Add a a grid to the plot
    ax.grid(
        ls='--',
        alpha=0.3,
        color='k',
        lw=0.5,
        dash_capstyle='round',
        dashes=(0.05, 2.5),
    )

    # Set labels and limits for the axes
    ax.set_xlabel("Stellar distance (pc)")
    ax.set_ylabel("Stellar effective temperature (K)")
    ax.set_xlim(0, 25)
    ax.set_ylim(3500, 7500)

    # Define x, y, marker size, and color (and convert to numpy arrays)
    x = np.array(data.sy_dist.values)
    y = np.array(data.st_teff.values)
    ms = np.array([quantity_to_ms(_) for _ in data.st_eei_angsep.values])
    c = np.array([quantity_to_color(_) for _ in beta_values])

    # Loop over data points and plot them
    for x_i, y_i, ms_i, c_i in zip(x, y, ms, c):
        ax.plot(
            x_i,
            y_i,
            "o",
            ms=ms_i,
            markerfacecolor=c_i,
            markeredgecolor="none",
            zorder=99,
        )

    # Manually construct a color bar and add it to the plot
    divider = make_axes_locatable(ax)
    cbar_ax = divider.new_horizontal(size="6%", pad=0.05)
    cbar = mpl.colorbar.ColorbarBase(
        cbar_ax,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
    )
    fig.add_axes(cbar_ax)

    # Set up additional options for the colorbar
    cbar.ax.set_ylabel("Degrees from quadrature", fontsize=6)
    cbar.ax.set_xlim(0, 1)
    cbar.ax.set_ylim(0, 90)
    cbar.ax.tick_params(labelsize=6)

    # Add labels to the colorbar
    for label, color, x, key in [
        ("Rayleigh", "black", 0.33, "Rayleigh"),
        ("Rainbows", "white", 0.66, "Rainbow"),
        ("Ocean glint", "white", 0.33, "Ocean Glint"),
        ("Glories", "white", 0.66, "Glory"),
    ]:
        peak_phase_angle = PHASE_ANGLES_OF_FEATURES[key][1]
        degrees_from_quadrature = np.abs(90 - peak_phase_angle)
        cbar.ax.text(
            x=x,
            y=degrees_from_quadrature - 5,  # stay away from the edge
            s=label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=6,
            color=color,
        )

    # Manually create a legend for the markersize
    handles = [
        Line2D(
            [0],
            [0],
            linewidth=0,
            label="Angular separation (in mas):",
            markersize=0,
        )
    ]
    handles += [
        Line2D(
            [0],
            [0],
            linewidth=0,
            marker="o",
            label=f"{_:d}",
            markeredgecolor="black",
            markerfacecolor="none",
            markersize=quantity_to_ms(_),
        )
        for _ in [100, 300, 500, 700, 900]
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=6,
        ncol=len(handles),
        edgecolor="none",
        borderpad=1.25,
    )

    print("Done!", flush=True)
    print(f"Saving plot to PDF...", end=" ", flush=True)

    # Save the figure
    fig.tight_layout(pad=0)
    file_path = paths.figures / "figure-4-scatterplot.pdf"
    plt.savefig(
        file_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )

    print("Done!", flush=True)
