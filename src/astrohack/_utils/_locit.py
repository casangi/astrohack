from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy import optimize as opt
import astropy.units as units

from astrohack._utils._tools import _hadec_to_elevation
from astrohack._utils._conversion import _convert_unit
from astrohack._utils._algorithms import _least_squares_fit
from astrohack._utils._constants import *
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger


def _locit_chunk(locit_parms):
    logger = _get_astrohack_logger()
    logger.info(f'procesing {locit_parms["this_ant"]} {locit_parms["this_ddi"]}')
    antenna = locit_parms['ant_info'][locit_parms['this_ant']]
    src_dict = locit_parms['obs_info']['src_dict']

    geo_pos = antenna['geocentric_position']
    ant_pos = EarthLocation.from_geocentric(geo_pos[0], geo_pos[1], geo_pos[2], 'meter')

    xds_data = locit_parms['xds_data']
    elevation_limit = locit_parms['elevation_limit'] * _convert_unit('deg', 'rad', 'trigonometric')
    gains = xds_data['PHASE_GAINS'].values
    field_id = xds_data['FIELD_ID'].values
    time = xds_data.time.values
    pol = xds_data.pol.values
    if len(pol) > 2:
        msg = f'Polarization scheme {pol} is not what is expected for antenna based gains'
        logger.error(msg)
        raise Exception(msg)
    if locit_parms['polarization'] in pol:
        i_pol = np.where(pol == locit_parms['polarization'])[0][0]
        gains = gains[:, 0, i_pol]
    else:
        msg = f'Polarization {locit_parms["polarization"]} is not found in data'
        logger.error(msg)
        raise Exception(msg)

    astro_time = Time(time, format='mjd', scale='utc', location=ant_pos)
    lst = astro_time.sidereal_time("apparent").to(units.radian)/units.radian
    hadec = _build_coordinate_array(field_id, src_dict, 'precessed')
    # convert to actual hour angle
    hadec[0, :] = lst.value - hadec[0, :]
    elevation = _hadec_to_elevation(hadec, antenna['latitude'])

    linalg = False
    if linalg:
        fit, variance = _solve_a_la_aips_kterm_no_slope(hadec, gains, elevation, elevation_limit)
    else:
        fit, variance = _solve_via_scipy(hadec, gains, elevation, elevation_limit, verbose=True)

    fit, variance = _convert_results(xds_data.attrs['frequency'], fit, variance, 'mm', 'deg')
    print(fit)
    print(variance)
    return


def _build_coordinate_array(field_id, src_list, key):
    """ If this is a bottleneck, good candidate for numba"""
    n_samples = len(field_id)
    coordinates = np.ndarray([2, n_samples])
    for i_sample in range(n_samples):
        field = str(field_id[i_sample])
        coordinates[:, i_sample] = src_list[field][key]
    return coordinates


def _solve_a_la_aips_kterm_no_slope(hadec, gains, elevation, elevation_limit):
    """ If this is a bottleneck, good candidate for numba"""
    syssize = 4
    system = np.ndarray([syssize, syssize])
    vector = np.ndarray([syssize])
    system[:, :] = 0
    vector[:] = 0
    n_samples = hadec.shape[1]
    for i_sample in range(n_samples):
        if elevation[i_sample] > elevation_limit:
            afunc = np.ndarray([6])
            afunc[0] = np.cos(hadec[0, i_sample]) * np.cos(hadec[1, i_sample])
            afunc[1] = -np.sin(hadec[0, i_sample]) * np.cos(hadec[1, i_sample])
            afunc[2] = np.sin(hadec[1, i_sample])
            afunc[3] = np.cos(elevation[i_sample])
            for irow in range(syssize):
                weight = afunc[irow]
                for icol in range(irow):
                    value = afunc[icol]
                    system[irow, icol] += weight * value
                vector[irow] += gains[i_sample] * weight
        else:
            pass

    for irow in range(1, syssize):
        for icol in range(irow):
            system[icol, irow] = system[irow, icol]

    fit, variance, _ = _least_squares_fit(system, vector)
    return fit, variance


def _solve_via_scipy(hadec, gains, elevation, elevation_limit, verbose=False):
    logger  = _get_astrohack_logger()
    selelev = elevation > elevation_limit
    hadec = hadec[:, selelev]
    gains = gains[selelev]
    # First guess is no error in positions and no instrumental delay
    p0 = [0, 0, 0, 0]
    # Errors are not limited, but the instrumental delay is pegged to the -pi to pi range
    liminf = [-np.inf, -np.inf, -np.inf, -pi]
    limsup = [+np.inf, +np.inf, +np.inf, +pi]

    maxfevs = [100000, 1000000, 10000000]
    for maxfev in maxfevs:
        try:
            fit, covar = opt.curve_fit(_geometric_delay_phase, hadec, gains, p0=p0, bounds=[liminf, limsup], maxfev=maxfev)
        except RuntimeError:
            if verbose:
                logger.info("Increasing number of iterations")
                continue
            else:
                if verbose:
                    logger.info("Converged with less than {0:d} iterations".format(maxfev))
                break

    variance = np.diag(covar)
    return fit, variance


def _geometric_delay_phase(coordinates, xoff, yoff, zoff, inst_delay):
    ha, dec = coordinates
    cosdec = np.cos(dec)
    zterm = twopi*np.sin(dec)*zoff
    xterm = twopi*cosdec*np.cos(ha)*xoff
    yterm = -twopi*cosdec*np.sin(ha)*yoff
    return xterm + yterm + zterm + inst_delay


def _convert_results(frequency, fit, variance, lengthunit, trigounit):
    wavelength = clight/frequency
    lengthfact = wavelength * _convert_unit('m', lengthunit, 'length')
    newfit = np.copy(fit)
    newvar = np.copy(variance)
    newfit[0:3] *= lengthfact
    newvar[0:3] *= lengthfact
    trigofact = _convert_unit('rad', trigounit, 'trigonometric')
    newfit[3] *= trigofact
    newvar[3] *= trigofact
    return newfit, newvar






