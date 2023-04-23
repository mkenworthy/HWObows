"""
Create the yarn ball plot with random eccentric orbits.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import time

from astropy.io import ascii
from matplotlib.patches import Circle
from tqdm.auto import tqdm

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from utils.phases import kep3d
from utils.samplers import sample_e, sample_i
from utils.plotting import CBF_COLORS
from utils import paths


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

def draw_iwa(ax: plt.Axes, iwas: np.ndarray) -> None:
    """
    Auxiliary function to draw inner working angles on a given axis.
    """

    for iwa in iwas:
        ax.add_patch(
            Circle(
                xy=(0, 0),
                radius=float(iwa),
                facecolor="none",
                edgecolor="k",
                ls="--",
                lw=0.5,
                alpha=1.0,
            )
        )


def convert_lod_to_mas(
    lod: float,
    wavelength: u.Quantity = 600 * u.nm,
    diameter: u.Quantity = 6 * u.m,
) -> float:
    """
    Convert a angular separation from units of lambda / D to mas.
    """

    return (
        (lod * wavelength / diameter)
        .to(
            "mas",
            equivalencies=u.dimensionless_angles(),
        )
        .value
    )


def prepare_ax(
    ax: plt.Axes,
    star_name: str,
    distance: float,
    hz: float,
) -> None:
    """
    Auxiliary function to prepare an axis for plotting.
    """

    # Remove ticks and setup aspect ratio
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # Set up limits and draw inner working angles
    ax.set_xlim(-2 * hz, 2 * hz)
    ax.set_ylim(-2 * hz, 2 * hz)
    draw_iwa(ax, iwas)

    # Define options for text boxes
    text_kwargs = dict(
        ha="left",
        va="top",
        color="black",
        fontsize=6,
        bbox=dict(
            boxstyle="square,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
        ),
    )

    # Add text boxes with star name / distance and size of the HZ
    ax.text(
        x=0.05,
        y=0.95,
        s=star_name,
        transform=ax.transAxes,
        weight="bold",
        **text_kwargs,
    )
    ax.text(
        x=0.05,
        y=0.15,
        s=rf"$d_*$ = {distance:.1f} pc",
        transform=ax.transAxes,
        **text_kwargs,
    )
    ax.text(
        x=0.05,
        y=0.08,
        s=rf"HZ = {hz:.1f} mas",
        transform=ax.transAxes,
        **text_kwargs,
    )


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print("\nCREATE YARN PLOT\n")

    # Start timer
    script_start = time.time()

    # Set random seed (for reproducibility)
    np.random.seed(42)

    # Use colorblind-friendly colors as default color cycle
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CBF_COLORS)

    # -------------------------------------------------------------------------
    # Setup configuration for simulations and plot
    # -------------------------------------------------------------------------

    # Whether or not to create a plot
    # If False, only the output data of the simulations will be saved
    create_plot = True

    # Define grid size for plot
    # We will make a plot with (grid_size, grid_size) panels
    grid_size = 4

    # Number of orbits to plot for each star
    n_orbits_to_plot = 10

    # Number of planet simulations per star
    # This number can be larger than the number of orbits to plot
    n_planets = 10

    # Number of epochs to sample along the orbit of the planet
    n_orbit_samples = 1000

    # -------------------------------------------------------------------------
    # Read in the target list
    # -------------------------------------------------------------------------

    print("Reading in target list...", end=" ", flush=True)

    file_name = "2646_NASA_ExEP_Target_List_HWO_Table.csv"
    targets = ascii.read(paths.data / file_name, header_start=1, data_start=2)
    n_stars = len(targets)

    print(f"Done! ({n_stars} stars in total)")

    # -------------------------------------------------------------------------
    # Prepare plot (if desired)
    # -------------------------------------------------------------------------

    if create_plot:

        # Set up a new figure
        pad_inches = 0.025
        fig, axes = plt.subplots(
            nrows=grid_size,
            ncols=grid_size,
            figsize=(7 - 2 * pad_inches, 7 - 2 * pad_inches),
        )

        # Flatten the axes array for easier access
        axes = axes.flatten()

    # We only need this to make the linter happy
    else:
        pad_inches = None
        fig = None
        axes = None

    # -------------------------------------------------------------------------
    # Loop over targets and simulate orbits
    # -------------------------------------------------------------------------

    # Define inner working angles (IWA) in mas for computing beta_max
    iwas = np.array([convert_lod_to_mas(_) for _ in [1, 2, 3, 4]])

    # Set up output arrays
    beta_min = np.zeros((n_stars, n_planets, iwas.size))
    beta_max = np.zeros((n_stars, n_planets, iwas.size))

    # Loop over all the stars in the table
    print("Simulating orbits:", flush=True)
    for n, target in tqdm(enumerate(targets), total=n_stars, ncols=80):

        # Define shortcuts
        star_name = str(target["hd_name"])
        distance = target["sy_dist"]
        hz = target["st_eei_angsep"]

        # If we are creating a plot, set up the axes for this star
        if create_plot and (n < len(axes)):
            prepare_ax(axes[n], star_name, distance, hz)

        # Sample eccentricities and inclinations
        esamp = sample_e(n_planets)
        isamp = sample_i(n_planets) * u.deg
        anodesamp = np.random.random_sample((n_planets,)) * 360 * u.deg
        wsamp = np.random.random_sample((n_planets,)) * 360 * u.deg

        # Simulate the given number of planets (orbits) for this star
        for m in np.arange(n_planets):

            # Set up our orbit
            # CAUTION! you need to use Kepler's laws and know M1 and M2
            # to set P and a consistently :)
            P = 1.0 * u.year
            tperi = 2050.0 * u.year  # TODO: where does "2050" come from?
            a = hz  # in mas
            e = esamp[m]
            i = isamp[m]
            w = wsamp[m]
            anode = anodesamp[m]

            # TODO: This is where we need to compute beta_min and beta_max
            #   for the given IWA values and store them in the output arrays

            # Plot some orbits (if desired)
            if create_plot and (n < len(axes)) and (m < n_orbits_to_plot):
                epochs = np.linspace(tperi, tperi + P, 100, endpoint=True)
                _, _, Xorb, Yorb, Zorb, _, _, _ = kep3d(
                    epochs, P, tperi, a, e, i, w, anode
                )
                axes[n].plot(Xorb, Yorb)

    print()

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    # TODO: This is where we need to save the output arrays for beta_min and
    #   beta_max to disk. (Currently, we do not compute values for them.)

    if create_plot:

        print('Saving plot...', end=' ', flush=True)

        fig.tight_layout(pad=0)
        file_path = paths.figures / "ball_of_yarn.pdf"
        plt.savefig(
            file_path,
            bbox_inches="tight",
            pad_inches=pad_inches,
        )

        print('Done!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
