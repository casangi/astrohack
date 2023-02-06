import numpy as np


# Physical constants
clight = 299792458.0

# Unit conversion
mm2m = 1/1000.
m2mm = 1000.
m2mils = 1e6 / 2.54
mils2m = 1. / m2mils


# Trigonometric unit conversions
rad2deg = 180. / np.pi
deg2rad = np.pi / 180.
pi = np.pi
twopi = 2.0*np.pi
fourpi = 4.0*np.pi


# Global conversion functions
def convert_to_db(val: float):
    """
    Converts a float value to decibels
    Args:
        val (float): Value to be converted to decibels
    Returns:
        Value in decibels
    """
    return 10.0 * np.log10(val)
