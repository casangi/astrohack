import numpy as np
import scipy.constants as constants

# Physical constants
clight = constants.speed_of_light
notavail = 'N/A'

# Length units
length_units = ['km', 'mi', 'm', 'yd', 'ft', 'in', 'cm', 'mm', 'um', 'mils']
# From m to unit
length_factors = [1e3, 1609.34, 1.0, 0.9144, 0.3048, 0.0254, 1e-2, 1e-3, 1e-6, 25.4 / 1e6]

# Trigonometric units
trigo_units = ['rad', 'deg', 'hour', 'asec', 'amin']
# from rad to unit
trigo_factors = [1.0, constants.pi / 180., constants.pi / 12., constants.pi / 180. / 3600., constants.pi / 180. / 60.]

# Time units
time_units = ['nsec', 'usec', 'msec', 'sec', 'min', 'hour', 'day']
# from sec to unit
time_factors = [1e-9, 1e-6, 1e-3, 1.0, 60.0, 3600.0, 86400.0]

unit_dict = {
    'length': length_units,
    'trigonometric': trigo_units,
    'time': time_units
}

fact_dict = {
    'length': length_factors,
    'trigonometric': trigo_factors,
    'time': time_factors
}

# Trigonometric unit conversions
pi = constants.pi
twopi = 2.0 * constants.pi
fourpi = 4.0 * constants.pi

# https://github.com/casacore/casacore/blob/dbf28794ef446bbf4e6150653dbe404379a3c429/measures/Measures/Stokes.h
pol_codes_RL = np.array([5, 6, 7, 8])  # 'RR','RL','LR','LL'
pol_codes_XY = np.array([9, 10, 11, 12])  # ['XX','XY','YX','YY']
pol_str = np.array(['0', 'I', 'Q', 'U', 'V', 'RR', 'RL', 'LR', 'LL', 'XX', 'XY', 'YX', 'YY'])

# possible complex splits:
possible_splits = ['cartesian', 'polar']

# Plot convenience
plot_types = ['deviation', 'phase', 'ancillary', 'all']
figsize = [8, 6.4]
fontsize = 6.4
markersize = 3.2


def custom_unit_checker(unit_type):
    if unit_type == "units.trig":
        return trigo_units

    elif unit_type == "units.length":
        return length_units

    elif unit_type == "units.time":
        return time_units

    else:
        return "Not found"


def custom_split_checker(split_type):
    if split_type == 'split.complex':
        return possible_splits
    else:
        return "Not found"
