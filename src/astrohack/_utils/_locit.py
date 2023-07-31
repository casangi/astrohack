from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as units
import numpy as np

from astrohack._utils._tools import _hadec_to_elevation
from astrohack._utils._conversion import _convert_unit
from astrohack._utils._algorithms import _least_squares_fit


def _locit_chunk(locit_parms):
    print(f'procesing {locit_parms["this_ant"]} {locit_parms["this_ddi"]}')
    antenna = locit_parms['ant_info'][locit_parms['this_ant']]
    src_dict = locit_parms['obs_info']['src_dict']

    geo_pos = antenna['geocentric_position']
    ant_pos = EarthLocation.from_geocentric(geo_pos[0], geo_pos[1], geo_pos[2], 'meter')

    xds_data = locit_parms['xds_data']
    elevation_limit = locit_parms['elevation_limit'] * _convert_unit('deg', 'rad', 'trigonometric')
    gains = xds_data['PHASE_GAINS'].values
    field_id = xds_data['FIELD_ID'].values
    time = xds_data.time.values
    astro_time = Time(time, format='mjd', scale='utc', location=ant_pos)
    lst = astro_time.sidereal_time("apparent").to(units.radian)/units.radian
    precessed_ra, precessed_dec = _build_coordinate_arrays(field_id, src_dict)

    hour_angle = lst.value - precessed_ra
    elevation = _hadec_to_elevation(hour_angle, precessed_dec, antenna['latitude'])

    system, vector = _build_system(hour_angle, precessed_dec, gains, elevation, elevation_limit)
    print(system)
    print(vector)
    results, variances, residuals = _least_squares_fit(system, vector)
    print(results, variances, residuals)
    return


def _build_coordinate_arrays(field_id, src_list):
    """ If this is a bottleneck, good candidate for numba"""
    n_samples = len(field_id)
    precessed_ra = np.ndarray([n_samples])
    precessed_dec = np.ndarray([n_samples])

    for i_sample in range(n_samples):
        field = str(field_id[i_sample])
        precessed_ra[i_sample] = src_list[field]['precessed'][0]
        precessed_dec[i_sample] = src_list[field]['precessed'][1]
    return precessed_ra, precessed_dec


def _build_system(ha, dec, gains, elevation, elevation_limit):
    """ If this is a bottleneck, good candidate for numba"""
    syssize = 4
    system = np.ndarray([syssize, syssize])
    vector = np.ndarray([syssize])
    system[:, :] = 0
    vector[:] = 0
    n_samples = len(ha)
    for i_sample in range(n_samples):
        if elevation[i_sample] > elevation_limit:
            afunc = np.ndarray([6])
            afunc[0] = np.cos(ha[i_sample]) * np.cos(dec[i_sample])
            afunc[1] = -np.sin(ha[i_sample]) * np.cos(dec[i_sample])
            afunc[2] = np.sin(dec[i_sample])
            afunc[3] = np.cos(elevation[i_sample])
            for irow in range(syssize):
                weight = afunc[irow]
                for icol in range(irow):
                    value = afunc[icol]
                    system[irow, icol] += weight * value
                vector[irow] += gains[i_sample, 0, 1] * weight
        else:
            pass

    for irow in range(1, syssize):
        for icol in range(irow):
            system[icol, irow] = system[irow, icol]
    return system, vector







