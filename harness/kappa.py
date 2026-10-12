"""Viscosity ratio kappa for the Herning rig (D12, D55).

Copied from the MSSP code (publication-ceramicspeed-dataset/src/ceramicspeed/
calculate_kappa.py), unchanged in substance:
    kappa = nu(T) / nu_1(n, d_pw)
    nu(T):  kinematic viscosity at temperature T, ASTM D341 (Walther), from the
            viscosities at 40 C and 100 C;
    nu_1:   minimum required viscosity, ISO 281.
"""

import numpy as np


def minimum_required_viscosity(rpm, d_pw):
    """nu_1 in mm^2/s (ISO 281). rpm > 0; d_pw = mean pitch diameter in mm."""
    if rpm < 1000:
        return 45000 * rpm ** (-0.83) * d_pw ** (-0.5)
    return 4500 * rpm ** (-0.5) * d_pw ** (-0.5)


def viscosity_at_temperature(temp_c, nu_40, nu_100):
    """Kinematic viscosity in mm^2/s at temp_c (ASTM D341, Walther equation)."""
    t1, t2, t = 40.0 + 273.15, 100.0 + 273.15, temp_c + 273.15
    a1 = np.log10(np.log10(nu_40 + 0.7))
    a2 = np.log10(np.log10(nu_100 + 0.7))
    b = (a2 - a1) / (np.log10(t2) - np.log10(t1))
    a = a1 - b * np.log10(t1)
    return float(10.0 ** (10.0 ** (a + b * np.log10(t))) - 0.7)


def kappa(rpm, temp_c, d_pw, nu_40, nu_100):
    """kappa = nu(T) / nu_1(n, d_pw)."""
    return viscosity_at_temperature(temp_c, nu_40, nu_100) / minimum_required_viscosity(rpm, d_pw)
