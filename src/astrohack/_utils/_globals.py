import numpy as np
from astrohack._utils._system_message import error


# Physical constants
clight = 299792458.0

# Unit conversion
mm2m = 1/1000.
m2mm = 1000.
m2mils = 1e6 / 25.4
mils2m = 1. / m2mils

# Length units
length_units = ['km', 'mi', 'm', 'cm', 'mm', 'um', 'mils']
# From m to unit
length_factors = [1e3, 1609.34, 1.0, 1e-2, 1e-3, 1e-6, 25.4/1e6]

# Trigonometric units
trigo_units = ['rad', 'deg']
# from rad to unit
trigo_factors = [1.0, np.pi/180.]

unit_dict = {'length': length_units,
             'trigonometric': trigo_units}
fact_dict = {'length': length_factors,
             'trigonometric': trigo_factors}


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


def convert_unit(unitin, unitout, kind):
    try:
        unitlist = unit_dict[kind]
        factorlist = fact_dict[kind]
    except KeyError:
        error("Unrecognized unit kind: " + kind)
        raise KeyError('Unrecogized unit kind')
    inidx = test_unit(unitin, unitlist)
    ouidx = test_unit(unitout, unitlist)
    factor = factorlist[inidx]/factorlist[ouidx]
    return factor


def test_unit(unit, unitlist):
    try:
        idx = unitlist.index(unit)
    except ValueError:
        error("Unrecognized unit: " + unit)
        raise ValueError('Unit not in list')
    return idx
