"""
Create Figure 7: Contrast over separation for three example targets.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import astropy.units as u
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colorbar import Colorbar
from scipy import interpolate

from utils.constants import SCATTERING_ANGLES_OF_FEATURES, DIAMETER
from utils.paths import (
    figures as figures_dir,
    data as data_dir,
)
from utils.plotting import set_fontsize


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\nCREATE FIGURE 6: CONTRAST OVER SEPARATION\n")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    print("Loading data...", end=" ", flush=True)

    # Read in target list
    file_path = data_dir / "2646_NASA_ExEP_Target_List_HWO_Table.csv"
    targets = pd.read_csv(file_path, header=[0, 1])

    # Select habitable zone values and target names
    hz = targets["EEIDmas"].values.astype(float)  # in mas
    target_names = targets["CommonID"].values.astype(str)

    # Read in data from Trees and Stam (2019) and extract I, Q, P, and phases.
    # Note: We add 2*std to make the values consistent with Figure 2.
    # Note: The index [3] corresponds to the 670 nm band.
    file_path = data_dir / "v7_fc50.h5"
    with h5py.File(file_path, "r") as hdf_file:
        I_data = np.array(hdf_file["I"])[3] + 2 * np.array(hdf_file["Istd"])[3]
        Q_data = (
            np.abs(np.array(hdf_file["Q"]))[3]
            + 2 * np.array(hdf_file["Qstd"])[3]
        )
        P_data = (
            np.array(hdf_file["Pt"])[3] + 2 * np.array(hdf_file["Ptstd"])[3]
        )
        phases = np.array(hdf_file["phase"])

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Compute corrected contrast values
    # -------------------------------------------------------------------------

    print("Computing corrected contrast values...", end=" ", flush=True)

    # Calculate contrast correction factor from Lambertian phase function to
    # Trees phase function
    geometric_albedo = 0.2
    Lambertian_flux_quadrature = (
        1.0
        / np.pi
        * (np.sin(np.pi / 2) + (np.pi - np.pi / 2) * (np.cos(np.pi / 2)))
        * geometric_albedo
    )
    Trees_flux_quadrature = (I_data[89] + I_data[90]) / 2
    contrast_correction_factor = (
        Trees_flux_quadrature / Lambertian_flux_quadrature
    )

    # By multiplying all contrasts by the correction factor, we now use the
    # Trees model instead of a Lambertian phase function
    contrasts = (
        targets["Contrast"].values.astype(float) * contrast_correction_factor
    )

    # Interpolate degree of linear polarization at 670 nanometers
    P_670 = interpolate.interp1d(
        x=phases,
        y=np.nan_to_num(P_data),
        kind="cubic",
    )

    # Interpolate total flux at 670 nanometers, normalized at quadrature.
    # The final contrast will be calculated by multiplying this number with the
    # contrast from the target list, corrected for the Trees phase function.
    F_670 = interpolate.interp1d(
        x=phases,
        y=np.nan_to_num(I_data / Trees_flux_quadrature),
        kind="cubic",
    )

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Define some parameters
    # -------------------------------------------------------------------------

    # Define the circular orbits (defines trajectory of planet in the plot)
    orbital_phase = np.linspace(0, 2 * np.pi, 1000)
    inclination = np.deg2rad(90)
    alpha = np.rad2deg(np.arccos(np.cos(orbital_phase) * np.sin(inclination)))

    # Compute the IWA at 670 nm
    WAVELENGTH = 670 * u.nm
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        IWA = (WAVELENGTH / DIAMETER).to(u.mas)

    # -------------------------------------------------------------------------
    # Prepare the plot
    # -------------------------------------------------------------------------

    # Select targets to plot (name in target list and label in plot)
    names = ["Lalande 21185", "Ran", "Rigil Kentaurus"]
    labels = ["Lalande 21185", r"$\epsilon$ Eri", r"$\alpha$ Cen A"]

    # Define a mapping of phase angle to color
    norm = mpl.colors.Normalize(vmin=0, vmax=180)
    cmap = plt.get_cmap("coolwarm_r")

    # Set up the figure
    pad_inches = 0.025
    fig, axes = plt.subplots(
        ncols=4,
        figsize=(7 - 2 * pad_inches, 2.5 - 2 * pad_inches),
        width_ratios=[1, 1, 1, 0.15],
    )

    # -------------------------------------------------------------------------
    # Create plot for each panel
    # -------------------------------------------------------------------------

    print("Creating plot...", end=" ", flush=True)

    for i, (name_in_target_list, label, ax, xmax) in enumerate(
        zip(names, labels, axes[:-1], [1e2, 2e2, 1e3], strict=True)
    ):
        # Set up the panel for the plot
        set_fontsize(ax, 6)
        ax.set_title(label, fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("Separation (in mas)")
        ax.set_ylabel("Contrast")
        ax.set_xlim(0, xmax)
        ax.set_ylim(1e-12, 1e-7)
        ax.get_yaxis().set_visible(i == 0)

        # Plot vertical lines at 1, 2, and 3 IWA
        for x in np.arange(1, 4) * IWA:
            ax.axvline(x=x.value, ls="--", c="k", alpha=0.5, lw=0.5)

        # Get index of current target in target list
        idx_in_target_list = np.where(target_names == name_in_target_list)[0]

        # Define separation as function of orbital phase
        x = np.sin(orbital_phase)
        y = np.cos(inclination) * np.cos(orbital_phase)
        r = np.sqrt(x**2 + y**2).flatten()
        separation = r * float(hz[idx_in_target_list])

        # Define planet contrast(s) as function of separation
        base_contrast = float(contrasts[idx_in_target_list])
        contrast_total_flux = base_contrast * F_670(alpha).flatten()
        contrast_polarization = contrast_total_flux * P_670(alpha).flatten()

        # Plot habitable zones of all targets
        ax.scatter(
            x=hz,
            y=contrasts,
            s=3,
            fc="#666666",
            ec="none",
            zorder=-1,
        )

        # Plot the total flux contrast orbit as a solid black line; add the
        # contrast at quadrature as a thick black dot
        ax.plot(separation, contrast_total_flux, c="k", lw=1)
        ax.scatter(hz[idx_in_target_list], base_contrast, c="k", s=12)

        # Plot the linear polarization contrast orbit as a multi-colored line
        x = separation
        y = contrast_polarization
        scattering_angle = 180 - alpha
        colors = [cmap(norm(angle)) for angle in scattering_angle]
        for x1, x2, y1, y2, c in zip(x[:-1], x[1:], y[:-1], y[1:], colors):
            ax.plot([x1, x2], [y1, y2], c=c, lw=2.5, solid_capstyle="round")

    # -------------------------------------------------------------------------
    # Add labels for the IWA lines
    # -------------------------------------------------------------------------

    for x in np.arange(1, 4) * IWA:
        axes[0].text(
            x=x.value,
            y=2e-12,
            s=f"{x:.0f}",
            c="k",
            alpha=0.5,
            ha="center",
            va="bottom",
            fontsize=4,
            rotation=90,
            bbox=dict(
                color="white",
                alpha=0.95,
                lw=0,
                boxstyle="square,pad=0.1",
            ),
        )

    # -------------------------------------------------------------------------
    # Manually construct a color bar and add it to the plot
    # -------------------------------------------------------------------------

    # Create a new axis for the colorbar
    cbar_ax = axes[-1]
    cbar = mpl.colorbar.ColorbarBase(
        cbar_ax,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
    )
    fig.add_axes(cbar_ax)

    # Set up additional options for the colorbar
    cbar.ax.set_ylabel("Scattering angle (in degrees)", fontsize=6)
    cbar.ax.set_xlim(0, 1)
    cbar.ax.set_ylim(0, 180)
    cbar.ax.set_yticks(np.arange(0, 181, 30))
    cbar.ax.tick_params(labelsize=6)

    # Add labels to the colorbar
    for label, color, x, key in [
        ("Rayleigh", "black", 0.3, "Rayleigh"),
        ("Rainbows", "black", 0.7, "Rainbow"),
        ("Ocean glint", "white", 0.7, "Ocean Glint"),
        ("Glories", "white", 0.3, "Glory"),
    ]:
        peak_angle = SCATTERING_ANGLES_OF_FEATURES[key][1]
        offset = 10 if key == "Glory" else 0  # shift label for Glories down
        cbar.ax.text(
            x=x,
            y=peak_angle - offset,
            s=label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=6,
            color=color,
        )

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------

    fig.tight_layout(pad=0, w_pad=0.4)

    print(f"Saving plot to PDF...", end=" ", flush=True)
    file_name = "figure-7-contrast-over-separation.pdf"
    file_path = figures_dir / file_name
    fig.savefig(
        file_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    print("Done!", flush=True)
