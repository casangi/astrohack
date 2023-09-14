import numpy as np

from astrohack._utils._panel_classes.telescope import Telescope


def _open_telescope(telname):
    """Open correct telescope based on the telescope string"""
    if 'VLA' in telname:
        telname = 'VLA'
    elif 'ALMA' in telname:
        telname = 'ALMA_DA'  # It does not matter which ALMA layout since the array center is the same
    telescope = Telescope(telname)
    return telescope


def _compute_antenna_relative_off(antenna, tel_lon, tel_lat, tel_rad):
    antenna_off_east = tel_rad * (antenna['longitude'] - tel_lon) * np.cos(tel_lat)
    antenna_off_north = tel_rad * (antenna['latitude'] - tel_lat)
    antenna_off_ele = antenna['radius'] - tel_rad
    antenna_dist = np.sqrt(antenna_off_east ** 2 + antenna_off_north ** 2 + antenna_off_ele ** 2)
    return antenna_off_east, antenna_off_north, antenna_off_ele, antenna_dist
