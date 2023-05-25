"""
Script to create LaTeX code for Table 1.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import time

from datetime import datetime
from socket import gethostname

from tabulate import tabulate

import astropy.units as u
import numpy as np
import pandas as pd

from utils.constants import lambda_over_d_in_mas
from utils.paths import tables as tables_dir


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    print("\n" + 80 * "-")
    print("\nCREATE TABLE 1\n")

    # Start timer
    script_start = time.time()

    # -------------------------------------------------------------------------
    # Compute values
    # -------------------------------------------------------------------------

    print("Computing values...", end=" ", flush=True)

    multiples_of_lambda_over_d = [1, 2, 3, 4, 32, 64]
    wavelengths = [600 * u.nm, 670 * u.nm, 1000 * u.nm]

    data = []
    for lambda_over_d in multiples_of_lambda_over_d:
        data.append(
            {
                "label": "IWA" if lambda_over_d <= 4 else "OWA",
                "lambda_over_d": lambda_over_d,
                **{
                    wavelength: "{:.1f}".format(
                        np.around(
                            lambda_over_d
                            * lambda_over_d_in_mas(wavelength).value,
                            1
                        )
                    )
                    for wavelength in wavelengths
                }
            }
        )

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Create table and write to file
    # -------------------------------------------------------------------------

    print(f"Creating table...", end=" ", flush=True)

    # Create dataframe
    df = pd.DataFrame(data)
    df.set_index("label", inplace=True)

    # Define the column names
    # We have to get a little creative here because the tabulate package
    # automatically escapes LaTeX commands, which we don't want here, so we
    # have to use a hacky workaround.
    headers = {"": "", "COL1---------": "$\lambda / D$"}
    for i, wavelength in enumerate(wavelengths, start=2):
        if wavelength >= 1000 * u.nm:
            number = wavelength.value / 1_000.0
            unit = r"\micro\meter"
        else:
            number = wavelength.value
            unit = r"\nano\meter"
        value = rf"mas (at \qty{{{number:.0f}}}{{{unit}}})"
        headers[f"COL{i}" + (len(value) - 4) * "-"] = value

    # Create the LaTeX table code
    table = tabulate(
        tabular_data=df,
        headers=headers,
        tablefmt="latex_booktabs",
        stralign="right",
        disable_numparse=True,
    )
    for key, value in headers.items():
        table = table.replace(key, value)

    print("Done!", flush=True)
    print(f"Saving table to *.tex file...", end=" ", flush=True)

    file_name = "table-1-iwa-owa.tex"
    with open(tables_dir / file_name, "w") as tex_file:

        # Add warning at the start
        tex_file.write(
            "\n% ATTENTION:\n"
            "% THIS FILE IS GENERATED AUTOMATICALLY BY "
            "`create-table-1-iwa-owa.py`\n"
            "% PLEASE DO NOT EDIT MANUALLY\n\n"
            f"% LAST UPDATE: {datetime.now().isoformat()} \n"
            f"% GENERATED ON: {gethostname()} \n\n"
        )

        # Add the table line by line
        for i, line in enumerate(table.split("\n")):
            tex_file.write(line + "\n")
            if i == 7:
                tex_file.write("\midrule\n")
        tex_file.write("\n")

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.1f} seconds!")
    print("\n" + 80 * "-" + "\n")
