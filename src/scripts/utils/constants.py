"""
Define constants used in the scripts.
"""

import astropy.units as u

d = 6.0  # diameter in m
wavelength = 600e-9  # central wavelength in m

# Assumed diameter of HWO
DIAMETER = u.Quantity(6, u.m)

# Assumed occurrence rate of rocky planets in the optimistic habitable zone
# See: National Academy of Sciences Astronomy & Astrophysics 2020 Decadal
# Survey, Section I.3.1. Value originally from Bryson et al. (2021).
ETA_EARTH = 0.24

# For each scattering feature, define the start, peak, and end scattering
# angle (in degrees)
SCATTERING_FEATURES = {
    "Rainbow": (127, 138, 158),
    "Rayleigh": (90, 110, 130),
    "Ocean Glint": (50, 30, 10),
    "Glory": (10, 5, 0),
}

# Assumed wavelength for the observations
WAVELENGTH = u.Quantity(600, u.nm)


# Compute lambda / D in mas
with u.set_enabled_equivalencies(u.dimensionless_angles()):
    LAMBDA_OVER_D_IN_MAS = (WAVELENGTH / DIAMETER).to(u.mas).value


if __name__ == "__main__":
    print(f'Telescope diameter is {d:3.1f} m')
    print(f'Central wavelength is {wavelength * 1e9:5.1f} nm')
    print("LAMBDA_OVER_D_IN_MAS:", LAMBDA_OVER_D_IN_MAS)
