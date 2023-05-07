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
        nrows=3,
        figsize=(8.15 - 2 * pad_inches, 4.2 - 2 * pad_inches),
        height_ratios=[1, 6, 1],
    )

    # Set font size for all axes
    for ax in axes:
        set_fontsize(ax, 6)

    print("Done!")

    # -------------------------------------------------------------------------
    # Top panel: Phase illustations
    # -------------------------------------------------------------------------

    print("Creating top panel...", end=' ', flush=True)

    for phase in np.arange(0, 181, 30):
        draw_phase(phase=phase, x_offset=phase, radius=3, ax=axes[0])

    axes[0].set_aspect('equal', 'box')
    axes[0].set_xlabel('Phase angle (in degrees)', fontsize=6)
    axes[0].set_xlim(0, 180)
    axes[0].set_xticks(np.arange(0, 181, 10))
    axes[0].set_xticklabels(
        [x if x % 30 == 0 else "" for x in np.arange(0, 181, 10)]
    )
    axes[0].set_ylim(-5, 5)
    axes[0].set_yticks([])
    axes[0].spines[['left', 'right', 'bottom']].set_visible(False)
    axes[0].text(x=-5, y=0, s='Phase:', va='center', ha='right', fontsize=6)
    axes[0].tick_params(axis='both', which='major',  labelsize=6)
    axes[0].xaxis.set_label_position('top')
    axes[0].xaxis.tick_top()
    axes[0].yaxis.set_tick_params(width=0)

    print("Done!")

    # -------------------------------------------------------------------------
    # Middle panel: Normalized polarized flux (Q)
    # -------------------------------------------------------------------------

    print("Creating middle panel...", end=' ', flush=True)

    grid = np.linspace(0, 180, 1800)

    for i, wavelength in enumerate(wavelengths):
        label = rf'${int(wavelength)}\,\mathrm{{nm}}$'
        color = plt.get_cmap('rainbow')(i / (len(wavelengths) - 1))
        ls = '--' if wavelength == 600 else '-'

        x = np.linspace(0, 180, 180, endpoint=True)
        y = np.nan_to_num(Q[wavelength])
        f = interp1d(x, y, fill_value=0, bounds_error=False, kind='cubic')
        axes[1].plot(grid, np.clip(f(grid), 0, None), lw=1, ls=ls, color=color)

        props = dict(
            fc='white',
            ec='none',
            alpha=0.85,
            boxstyle='round,pad=0.01',
        )
        axes[1].text(
            x=68,
            y=f(68) + 0.0005,
            s=label,
            color=color,
            fontsize=4,
            ha='center',
            va='bottom',
            bbox=props,
        )

    axes[1].grid(
        ls='--',
        alpha=0.3,
        color='k',
        lw=0.5,
        dash_capstyle='round',
        dashes=(0.05, 2.5),
    )
    axes[1].set_xlim(0, 180)
    axes[1].set_xticks(np.arange(0, 181, 10))
    axes[1].set_ylabel('Normalised polarized flux', labelpad=10)
    axes[1].set_ylim(-0.005, 0.065)
    axes[1].spines[['bottom']].set_visible(False)
    axes[1].tick_params(
        bottom=False,
        top=True,
        labelbottom=False,
        labeltop=False,
        direction='inout',
    )

    print("Done!")

    # -------------------------------------------------------------------------
    # Bottom panel: Illustrations of scattering phenomena
    # -------------------------------------------------------------------------

    print("Creating bottom panel...", end=' ', flush=True)

    # Note: axes[2] displays the scattering angle, but under the hood, plotting
    # uses the phase angle (hence we need PHASE_ANGLES_OF_FEATURES here).
    for label, positions, file_name in (
        ("Glories", PHASE_ANGLES_OF_FEATURES['Glory'], 'glory.png'),
        ("Ocean Glint", PHASE_ANGLES_OF_FEATURES['Ocean Glint'], 'glint.jpg'),
        ("Rainbows", PHASE_ANGLES_OF_FEATURES['Rainbow'], 'rainbow.jpg'),
        ("Rayleigh", PHASE_ANGLES_OF_FEATURES['Rayleigh'], 'rayleigh.jpg'),
    ):
        img = plt.imread(static_dir / "bott-plot" / file_name)
        axes[2].imshow(X=img, extent=[positions[0], positions[2], 0, 10])

        axes[2].text(
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

    axes[2].set_xlabel('Scattering angle (in degrees)')
    axes[2].set_xlim(0, 180)
    axes[2].set_xticks(np.arange(0, 181, 10))
    axes[2].set_xticklabels(
        [x if x % 30 == 0 else "" for x in np.arange(0, 181, 10)[::-1]]
    )
    axes[2].set_ylim(0, 10)
    axes[2].set_yticks([])
    axes[2].spines[['left', 'right']].set_visible(False)
    axes[2].text(x=-5, y=5, s='Features:', va='center', ha='right', fontsize=6)
    axes[2].tick_params(axis='both', which='major')
    axes[2].tick_params(
        bottom=True,
        top=True,
        labelbottom=True,
        labeltop=False,
        direction='inout',
    )
    axes[2].yaxis.set_tick_params(width=0)

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
