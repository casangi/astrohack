import numpy as np

import toolviper.utils.logger as logger

from astrohack.utils.constants import unit_dict, fact_dict
from astrohack.utils.tools import get_str_idx_in_list


# Global conversion functions
def to_antenna_tapering(val: float | np.ndarray):
    """
    Converts a float or float array value to antenna tapering
    Args:
        val (float): Value to be converted to antenna tapering
    Returns:
        Value in antenna tapering
    """
    return 2 * to_db(val)


def to_db(val: float | np.ndarray):
    """
    Converts a float or float array value to decibels
    Args:
        val (float): Value to be converted to decibels
    Returns:
        Value in decibels
    """
    zero_filtered = np.where(val == 0, np.nan, val)
    return 10.0 * np.log10(zero_filtered)


def convert_unit(unitin, unitout, kind):
    """
    Convert between unit of the same kind
    Args:
        unitin: Origin unit
        unitout: Destination unit
        kind: 'trigonometric' or 'length'

    Returns:
        Conversion factor to go from unitin to unitout
    """
    try:
        unitlist = unit_dict[kind]
        factorlist = fact_dict[kind]

    except KeyError:

        logger.error("Unrecognized unit kind: " + kind)
        raise KeyError("Unrecognized unit kind")

    inidx = _test_unit(unitin, unitlist)
    ouidx = _test_unit(unitout, unitlist)
    factor = factorlist[inidx] / factorlist[ouidx]

    return factor


def _test_unit(unit, unitlist):
    """
    Test if a unit is known
    Args:
        unit: unit name
        unitlist: List containing unit names

    Returns:
        Unit index in unitlist
    """
    try:
        idx = unitlist.index(unit)
    except ValueError:
        logger.error("Unrecognized unit: " + unit)
        raise ValueError("Unit not in list")

    return idx


def convert_5d_grid_to_stokes(grid, pol_axis):
    """
    Convert gridded 5D data or aperture data from polariza correlations to Stokes
    Args:
        grid: Gridded 5D data expected to be of shape [time, chan, pol, x, y]
        pol_axis: The polarization correlation axis

    Returns:
        the same data now converted to Stokes parameters
    """
    grid_stokes = np.zeros_like(grid)

    if "RR" in pol_axis:
        irr = get_str_idx_in_list("RR", pol_axis)
        irl = get_str_idx_in_list("RL", pol_axis)
        ilr = get_str_idx_in_list("LR", pol_axis)
        ill = get_str_idx_in_list("LL", pol_axis)

        grid_stokes[:, :, 0, :, :] = (grid[:, :, irr, :, :] + grid[:, :, ill, :, :]) / 2
        grid_stokes[:, :, 1, :, :] = (grid[:, :, irl, :, :] + grid[:, :, ilr, :, :]) / 2
        grid_stokes[:, :, 2, :, :] = (
            1j * (grid[:, :, irl, :, :] - grid[:, :, ilr, :, :]) / 2
        )
        grid_stokes[:, :, 3, :, :] = (grid[:, :, irr, :, :] - grid[:, :, ill, :, :]) / 2
    elif "XX" in pol_axis:
        ixx = get_str_idx_in_list("XX", pol_axis)
        ixy = get_str_idx_in_list("XY", pol_axis)
        iyx = get_str_idx_in_list("YX", pol_axis)
        iyy = get_str_idx_in_list("YY", pol_axis)

        grid_stokes[:, :, 0, :, :] = (grid[:, :, ixx, :, :] + grid[:, :, iyy, :, :]) / 2
        grid_stokes[:, :, 1, :, :] = (grid[:, :, ixx, :, :] - grid[:, :, iyy, :, :]) / 2
        grid_stokes[:, :, 2, :, :] = (grid[:, :, ixy, :, :] + grid[:, :, iyx, :, :]) / 2
        grid_stokes[:, :, 3, :, :] = (
            1j * (grid[:, :, ixy, :, :] - grid[:, :, iyx, :, :]) / 2
        )
    else:
        raise ValueError("Pol not supported " + str(pol_axis))

    return grid_stokes


def convert_5d_grid_from_stokes(stokes_grid, input_pol_axis, destiny_pol_axis):
    i_i = get_str_idx_in_list("I", input_pol_axis)
    i_q = get_str_idx_in_list("Q", input_pol_axis)
    i_u = get_str_idx_in_list("U", input_pol_axis)
    i_v = get_str_idx_in_list("V", input_pol_axis)

    corr_grid = np.zeros_like(stokes_grid)

    if "RR" in destiny_pol_axis:
        irr = get_str_idx_in_list("RR", destiny_pol_axis)
        irl = get_str_idx_in_list("RL", destiny_pol_axis)
        ilr = get_str_idx_in_list("LR", destiny_pol_axis)
        ill = get_str_idx_in_list("LL", destiny_pol_axis)

        corr_grid[:, :, irr, :, :] = (
            stokes_grid[:, :, i_i, :, :] + stokes_grid[:, :, i_v, :, :]
        )
        corr_grid[:, :, irl, :, :] = (
            stokes_grid[:, :, i_q, :, :] - 1j * stokes_grid[:, :, i_u, :, :]
        )
        corr_grid[:, :, ilr, :, :] = (
            stokes_grid[:, :, i_q, :, :] + 1j * stokes_grid[:, :, i_u, :, :]
        )
        corr_grid[:, :, ill, :, :] = (
            stokes_grid[:, :, i_i, :, :] - stokes_grid[:, :, i_v, :, :]
        )

    elif "XX" in destiny_pol_axis:
        ixx = get_str_idx_in_list("XX", destiny_pol_axis)
        ixy = get_str_idx_in_list("XY", destiny_pol_axis)
        iyx = get_str_idx_in_list("YX", destiny_pol_axis)
        iyy = get_str_idx_in_list("YY", destiny_pol_axis)

        corr_grid[:, :, ixx, :, :] = (
            stokes_grid[:, :, i_i, :, :] + stokes_grid[:, :, i_q, :, :]
        )
        corr_grid[:, :, ixy, :, :] = (
            stokes_grid[:, :, i_u, :, :] - 1j * stokes_grid[:, :, i_v, :, :]
        )
        corr_grid[:, :, iyx, :, :] = (
            stokes_grid[:, :, i_u, :, :] + 1j * stokes_grid[:, :, i_v, :, :]
        )
        corr_grid[:, :, iyy, :, :] = (
            stokes_grid[:, :, i_i, :, :] - stokes_grid[:, :, i_q, :, :]
        )

    else:
        raise ValueError("Pol not supported " + str(destiny_pol_axis))

    return corr_grid


def convert_dict_from_numba(func):
    def wrapper(*args, **kwargs):
        numba_dict = func(*args, **kwargs)

        converted_dict = dict(numba_dict)

        for key, _ in numba_dict.items():
            converted_dict[key] = dict(converted_dict[key])

        return converted_dict

    return wrapper


def casa_time_to_mjd(times):
    corrected = times / 3600 / 24.0
    return corrected
