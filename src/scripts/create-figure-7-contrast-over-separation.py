"""
Create Figure 7: Contrast over separation for three example targets.
"""

from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from utils import paths
from utils.plotting import set_fontsize


df = pd.read_csv(paths.data / "2646_NASA_ExEP_Target_List_HWO_Table.csv")

hab_zones = np.array(df["EEIDmas"].values[1:], dtype=float)  # in mas
contrasts = np.array(df["Contrast"].values[1:], dtype=float)
names = np.array(df["CommonID"].values[1:], dtype=str)

data_trees = np.loadtxt(
    paths.data / "Reflected_light_curves_Trees_Stam_2019.txt"
)


P_670 = interpolate.interp1d(data_trees[0], data_trees[2])
F_670 = interpolate.interp1d(data_trees[0], data_trees[1])

orbital_phase = np.linspace(0, 2 * np.pi, 200)

inclination = np.radians(90)

alpha = np.arccos(np.cos(orbital_phase) * np.sin(inclination))

# Define a mapping of quantity (e.g., beta value) to a color
norm = mpl.colors.Normalize(vmin=0, vmax=90)
cmap = mpl.cm.get_cmap("viridis_r")


names_array = np.array(["Rigil Kentaurus", "Ran", "Lalande 21185"])
common_names = np.array(
    [r"$\alpha$ Cen A", r"$\epsilon$ Eri", "Lalande 21185"]
)

pad_inches = 0.025
fig, axs = plt.subplots(
    nrows=1, ncols=3, figsize=(9 - 2 * pad_inches, 3.5 - 2 * pad_inches)
)

xlims = np.array([1e3, 2e2, 1e2])

for i, name_one in enumerate(names_array):
    index2 = np.where(names == name_one)

    dtel = 6
    wl1 = 670e-9
    iwa1 = wl1 / dtel * 180 / np.pi * 3600 * 1000

    x2 = np.sin(orbital_phase)
    y2 = np.cos(inclination) * np.cos(orbital_phase)

    r = np.sqrt(x2**2 + y2**2)
    separation = r * hab_zones[index2]
    y = np.ones_like(separation) * contrasts[index2]

    ax = axs[i]

    # Set up the font size
    set_fontsize(ax, 6)

    ax.scatter(hab_zones[index2], contrasts[index2], c="k", alpha=1, s=50)
    ax.plot(separation, F_670(np.degrees(alpha)) / F_670(90) * y, c="k")
    ax.scatter(
        separation,
        F_670(np.degrees(alpha)) / F_670(90) * y * P_670(np.degrees(alpha)),
        c=np.abs(np.degrees(alpha) - 90),
        cmap=cmap,
        vmin=0,
        vmax=90,
    )

    ax.set_title(common_names[i])
    ax.set_yscale("log")

    if i > 0:
        ax.get_yaxis().set_visible(False)

    ax.set_xlim(0, xlims[i])
    ax.set_ylim(8e-12, 3e-8)

    ax.scatter(hab_zones, contrasts, c="k", alpha=0.2)
    ax.vlines(
        (np.arange(3) + 1) * iwa1,
        1e-13,
        1e-1,
        linestyles="--",
        colors="k",
        alpha=0.5,
    )

    ax.set_xlabel("Separation [mas]")
    ax.set_ylabel("Contrast")

    # Manually construct a color bar and add it to the plot
    if i == 2:
        divider = make_axes_locatable(ax)
        cbar_ax = divider.new_horizontal(size="8%", pad=0.05)
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
        cbar.ax.text(
            0.5,
            10,
            "Rayleigh",
            rotation=90,
            ha="center",
            va="center",
            fontsize=6,
        )
        cbar.ax.text(
            0.5, 30, "Glint", rotation=90, ha="center", va="center", fontsize=6
        )
        cbar.ax.text(
            0.5,
            50,
            "Rainbows",
            rotation=90,
            ha="center",
            va="center",
            fontsize=6,
            color="white",
        )
        cbar.ax.text(
            0.5,
            80,
            "Other",
            rotation=90,
            ha="center",
            va="center",
            fontsize=6,
            color="white",
        )

file_name = (
    f"figure-7-contrast-over-separation_i={int(np.degrees(inclination))}.pdf"
)
file_path = paths.figures / file_name

plt.savefig(file_path, bbox_inches="tight")
