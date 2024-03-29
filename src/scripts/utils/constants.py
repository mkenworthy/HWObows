"""
Define constants used in the scripts.
"""

import astropy.units as u


# Random seed
RANDOM_SEED = 42

# Assumed diameter of HWO
DIAMETER = u.Quantity(6, u.m)

# Assumed occurrence rate of rocky planets in the optimistic habitable zone
# See: National Academy of Sciences Astronomy & Astrophysics 2020 Decadal
# Survey, Section I.3.1. Value originally from Bryson et al. (2021).
ETA_EARTH = 0.24

# Define the start, peak, and end PHASE angle (in degrees) for the
# different scattering features. See Table 2 in the paper.
PHASE_ANGLES_OF_FEATURES = {
    "Glory": (0, 5, 10),
    "Rainbow": (22, 42, 63),
    "Rayleigh": (50, 70, 110),
    "Ocean Glint": (130, 150, 170),
}

# Define the start, peak, and end SCATTERING angle (in degrees)
SCATTERING_ANGLES_OF_FEATURES = {
    key: (180 - end, 180 - peak, 180 - start)
    for key, (start, peak, end) in PHASE_ANGLES_OF_FEATURES.items()
}

# Assumed default wavelength for most plots / calculations
WAVELENGTH = u.Quantity(600, u.nm)


# Compute lambda / D in mas
# Make `wavelength` an optional argument here so that we can use this function
# also to compute Table 2.
def lambda_over_d_in_mas(
    wavelength: u.Quantity = WAVELENGTH,
) -> u.Quantity:
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        return (wavelength / DIAMETER).to(u.mas)


if __name__ == "__main__":

    print("\nCONSTANTS USED IN THE SCRIPTS:\n")

    print("Occurrence rate of rocky planets in the optimistic habitable zone:")
    print(f"  ETA_EARTH = {ETA_EARTH:.2f}\n")

    print("Assumed wavelength:")
    print(f"  WAVELENGTH = {WAVELENGTH.to(u.nm):.2f}\n")

    print("Assumed telescope diameter:")
    print(f"  DIAMETER = {DIAMETER.to(u.m):.2f}\n")

    print("SCATTERING_ANGLES_OF_FEATURES (start, peak, end; in degrees):")
    for feature, (start, peak, end) in SCATTERING_ANGLES_OF_FEATURES.items():
        print(f"  {feature}: ({start}, {peak}, {end}) degrees")
    print()

    print("PHASE_ANGLES_OF_FEATURES (start, peak, end; in degrees):")
    for feature, (start, peak, end) in PHASE_ANGLES_OF_FEATURES.items():
        print(f"  {feature}: ({start}, {peak}, {end}) degrees")
    print()

    print("RANDOM_SEED:")
    print(f"  RANDOM_SEED = {RANDOM_SEED}\n")
