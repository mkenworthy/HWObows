"""
General utility functions (e.g., for plotting).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

# Define colorblind-friendly colors (CBF)
# Source: https://arxiv.org/abs/2107.02270
CBF_COLORS = [
    "#3F90DA",  # blue
    "#FFA90E",  # orange
    "#BD1F01",  # red
    "#832DB6",  # purple
    "#94A4A2",  # gray
    "#A96B59",  # brown
    "#E76300",  # orange
    "#B9AC70",  # tan
    "#717581",  # gray
    "#92DADD",  # light blue
]


def set_fontsize(ax: plt.Axes, fontsize: float) -> None:
    """
    Auxiliary function to set the fontsize of all elements in a plot.
    """

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)
