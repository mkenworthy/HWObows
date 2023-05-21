"""
Create Figure 1: The "Bott plot".
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from matplotlib.patches import Circle, Polygon, Ellipse
from scipy.interpolate import interp1d

import h5py
import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import set_fontsize
from utils.paths import (
    static as static_dir,
    figures as figures_dir,
    data as data_dir,
)
from utils.constants import PHASE_ANGLES_OF_FEATURES


# -----------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# -----------------------------------------------------------------------------

def draw_phase(
    phase: float,
    x_offset: float,
    radius: float,
    ax: plt.Axes,
) -> None:
    """
    Auxiliary function to draw an illustration of the planet phase.
    """

    # Define colors of the illuminated / dark park
    LIGHT = '#EEEEEE'
    DARK = '#333333'

    t = np.linspace(0, np.pi, 360, endpoint=True)

    if phase < 90:
        # Add dark background
        c = Circle((x_offset, 0), radius=radius, fc=DARK, ec='none')
        ax.add_patch(c)

        # Add right half
        x = np.sin(t) * radius + x_offset
        y = np.cos(t) * radius
        xy = np.column_stack([x, y])
        polygon = Polygon(
            xy=xy,
            closed=True,
            fc=LIGHT,
            ec='none',
            clip_on=False,
        )
        ax.add_artist(polygon)

        # Add left half
        x = -np.sin(t) * np.cos(np.deg2rad(phase)) * radius + x_offset
        y = np.cos(t) * radius
        xy = np.column_stack([x, y])
        polygon = Polygon(
            xy=xy,
            closed=True,
            fc=LIGHT,
            ec='none',
            lw=0,
            clip_on=False,
        )
        ax.add_artist(polygon)

    else:
        # Add light background
        c = Circle((x_offset, 0), radius=radius, fc=LIGHT, ec='none')
        ax.add_patch(c)

        # Add right half
        x = -np.sin(t) * np.cos(np.deg2rad(phase)) * radius + x_offset
        y = np.cos(t) * radius
        xy = np.column_stack([x, y])
        polygon = Polygon(
            xy=xy,
            closed=True,
            fc=DARK,
            ec='none',
            clip_on=False,
        )
        ax.add_artist(polygon)

        # Add left half
        x = -np.sin(t) * radius + x_offset
        y = np.cos(t) * radius
        xy = np.column_stack([x, y])
        polygon = Polygon(
            xy=xy,
            closed=True,
            fc=DARK,
            ec='none',
            clip_on=False,
        )
        ax.add_artist(polygon)

    if phase < 90:
        ellipse = Ellipse(
            (x_offset, 0), 0.2, 2 * radius, fc=LIGHT, ec='none', clip_on=False
        )
    elif phase == 90:
        ellipse = Ellipse(
            (x_offset, 0), 0.2, 2 * radius, fc='none', ec='none', clip_on=False
        )
    else:
        ellipse = Ellipse(
            (x_offset, 0), 0.2, 2 * radius, fc=DARK, ec='none', clip_on=False
        )
    ax.add_artist(ellipse)

    # Dark circle that acts as an outer border around everything
    c = Circle(
        (x_offset, 0),
        radius=1.03 * radius,
        ec=DARK,
        fc='none',
        lw=0.5,
        clip_on=False,
    )
    ax.add_patch(c)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    print("\nCREATE FIGURE 1: THE 'BOTT PLOT'\n")

    # -------------------------------------------------------------------------
    # Load data (from Trees & Stam 2019)
    # -------------------------------------------------------------------------

    print("Loading data from HDF...", end=' ', flush=True)

    Q = {}
    I = {}

    file_path = data_dir / 'v7_fc50.h5'
    with h5py.File(file_path, "r") as hdf_file:
        for i in range(5):
            wavelength = 1000 * float(hdf_file['wavs'][i])

            Q_abs = np.abs(np.array(hdf_file['Q'][i]))
            Q_std = np.array(hdf_file['Qstd'][i])
            Q[wavelength] = Q_abs + 2 * Q_std

            I_abs = np.abs(np.array(hdf_file['I'][i]))
            I_std = np.array(hdf_file['Istd'][i])
            I[wavelength] = I_abs + 2 * I_std

    # Interpolate to 600 nm
    Q[600] = (Q[670] - Q[550]) / (670 - 550) * 50 + Q[550]
    I[600] = (I[670] - I[550]) / (670 - 550) * 50 + I[550]

    wavelengths = sorted(Q.keys())

    print("Done!")

    # -------------------------------------------------------------------------
    # Prepare a new figure
    # -------------------------------------------------------------------------

    print("Preparing figure...", end=' ', flush=True)

    # Create a new figure
    # These numbers where carefully hand-tuned to ensure that the output has
    # exactly the right width (7 inches) for the paper when saved as a PDF.
    # Check by running the following in the terminal:
    # >>> identify -verbose figure-1-bott-plot.pdf | grep "Print size"
    pad_inches = 0.025
    fig, axes = plt.subplots(
        nrows=4,
        figsize=(8.15 - 2 * pad_inches, 5.5 - 2 * pad_inches),
        height_ratios=[1, 4.5, 4.5, 1],
    )

    # Set font size for all axes
    for ax in axes:
        set_fontsize(ax, 6)

    print("Done!")

    # -------------------------------------------------------------------------
    # First panel: Phase illustations
    # -------------------------------------------------------------------------

    print("Creating first panel...", end=' ', flush=True)

    ax = axes[0]

    for phase in np.arange(0, 181, 30):
        draw_phase(phase=phase, x_offset=phase, radius=3, ax=axes[0])

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Phase angle (in degrees)', fontsize=6)
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 10))
    ax.set_xticklabels(
        [x if x % 30 == 0 else "" for x in np.arange(0, 181, 10)]
    )
    ax.set_ylim(-5, 5)
    ax.set_yticks([])
    ax.spines[['left', 'right', 'bottom']].set_visible(False)
    ax.text(x=-5, y=0, s='Phase:', va='center', ha='right', fontsize=6)
    ax.tick_params(axis='both', which='major',  labelsize=6)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.yaxis.set_tick_params(width=0)

    print("Done!")

    # -------------------------------------------------------------------------
    # Joint options for second and third panel
    # -------------------------------------------------------------------------

    # Set some options that are the same for both panels
    for ax in (axes[1], axes[2]):
        ax.grid(
            ls='--',
            alpha=0.3,
            color='k',
            lw=0.5,
            dash_capstyle='round',
            dashes=(0.05, 2.5),
        )
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 10))
        ax.spines[['bottom']].set_visible(False)
        ax.tick_params(
            bottom=False,
            top=True,
            labelbottom=False,
            labeltop=False,
            direction='inout',
        )

    # Define grid for interpolating curves to a "smooth" line
    grid = np.linspace(0, 180, 1800)

    # -------------------------------------------------------------------------
    # Second panel: Total flux (I)
    # -------------------------------------------------------------------------

    print("Creating second panel...", end=' ', flush=True)

    ax = axes[1]

    for i, wavelength in enumerate(wavelengths):

        # Prepare color and linestyle
        color = plt.get_cmap('rainbow')(i / (len(wavelengths) - 1))
        ls = '--' if wavelength == 600 else '-'

        # Interpolate to a smooth line
        x = np.linspace(0, 180, 180, endpoint=True)
        y = np.nan_to_num(I[wavelength])
        f = interp1d(x, y, fill_value=0, bounds_error=False, kind='cubic')

        # Plot the curve
        ax.plot(
            grid,
            np.clip(f(grid), 0, None),
            lw=1,
            ls=ls,
            color=color,
        )

    ax.set_ylabel('Total flux (in arbitrary units)', labelpad=10)
    ax.set_ylim(-0.03, 0.43)

    print("Done!")

    # -------------------------------------------------------------------------
    # Third panel: Normalized polarized flux (Q)
    # -------------------------------------------------------------------------

    print("Creating third panel...", end=' ', flush=True)

    ax = axes[2]

    for i, wavelength in enumerate(wavelengths):

        # Prepare label, color, and linestyle
        label = rf'${int(wavelength)}\,\mathrm{{nm}}$'
        color = plt.get_cmap('rainbow')(i / (len(wavelengths) - 1))
        ls = '--' if wavelength == 600 else '-'

        # Interpolate to a smooth line
        x = np.linspace(0, 180, 180, endpoint=True)
        y = np.nan_to_num(Q[wavelength])
        f = interp1d(x, y, fill_value=0, bounds_error=False, kind='cubic')

        # Plot the curve
        ax.plot(
            grid,
            np.clip(f(grid), 0, None),
            lw=1,
            ls=ls,
            color=color,
            zorder=2,
        )

        # Add label
        ax.text(
            x=68,
            y=f(68) + 0.0004,
            s=label,
            color=color,
            fontsize=4,
            ha='center',
            va='bottom',
            bbox=dict(
                fc='white',
                ec='none',
                alpha=0.85,
                boxstyle='round,pad=0.001',
            ),
            zorder=1,
        )

    ax.set_ylabel('Normalised polarized flux', labelpad=10)
    ax.set_ylim(-0.005, 0.065)

    print("Done!")

    # -------------------------------------------------------------------------
    # Fourth panel: Illustrations of scattering phenomena
    # -------------------------------------------------------------------------

    print("Creating fourth panel...", end=' ', flush=True)

    ax = axes[3]

    # Note: axes[2] displays the scattering angle, but under the hood, plotting
    # uses the phase angle (hence we need PHASE_ANGLES_OF_FEATURES here).
    for label, positions, file_name in (
        ("Glories", PHASE_ANGLES_OF_FEATURES['Glory'], 'glory.png'),
        ("Ocean Glint", PHASE_ANGLES_OF_FEATURES['Ocean Glint'], 'glint.jpg'),
        ("Rainbows", PHASE_ANGLES_OF_FEATURES['Rainbow'], 'rainbow.jpg'),
        ("Rayleigh", PHASE_ANGLES_OF_FEATURES['Rayleigh'], 'rayleigh.jpg'),
    ):
        img = plt.imread(static_dir / "bott-plot" / file_name)
        ax.imshow(X=img, extent=[positions[0], positions[2], 0, 10])

        ax.text(
            x=(positions[0] + positions[2]) / 2,
            y=10,
            s=label,
            ha='center',
            va='center',
            fontsize=5,
            color='white',
            bbox=dict(
                facecolor='black', edgecolor='none', boxstyle='round,pad=0.25'
            ),
        )

    ax.set_xlabel('Scattering angle (in degrees)')
    ax.set_xlim(0, 180)
    ax.set_xticks(np.arange(0, 181, 10))
    ax.set_xticklabels(
        [x if x % 30 == 0 else "" for x in np.arange(0, 181, 10)[::-1]]
    )
    ax.set_ylim(0, 10)
    ax.set_yticks([])
    ax.spines[['left', 'right']].set_visible(False)
    ax.text(x=-5, y=5, s='Features:', va='center', ha='right', fontsize=6)
    ax.tick_params(axis='both', which='major')
    ax.tick_params(
        bottom=True,
        top=True,
        labelbottom=True,
        labeltop=False,
        direction='inout',
    )
    ax.yaxis.set_tick_params(width=0)

    print("Done!")

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------

    print("Saving figure...", end=' ', flush=True)

    plt.subplots_adjust(hspace=0)

    file_path = figures_dir / 'figure-1-bott-plot.pdf'
    fig.savefig(
        fname=file_path,
        dpi=600,
        bbox_inches='tight',
        pad_inches=pad_inches,
    )

    print("Done!\n")
