import numpy as np


# Physical constants
clight = 299792458.0

# Unit conversion
mm2m = 1/1000.
m2mm = 1000.
m2mils = 1e6 / 25.4
mils2m = 1. / m2mils


# Trigonometric unit conversions
rad2deg = 180. / np.pi
deg2rad = np.pi / 180.
pi = np.pi
twopi = 2.0*np.pi
fourpi = 4.0*np.pi


#https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/measures/Measures/Stokes.h
pol_codes_RL = np.array([5,6,7,8]) #'RR','RL','LR','LL'
pol_codes_XY = np.array([9,10,11,12]) #['XX','XY','YX','YY']
pol_str = np.array(['0','I','Q','U','V','RR','RL','LR','LL','XX','XY','YX','YY'])


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
