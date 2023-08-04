import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy import optimize as opt
import astropy.units as units
import xarray as xr

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
    fit_kterm = locit_parms['fit_kterm']
    fit_slope = locit_parms['fit_slope']

    geo_pos = antenna['geocentric_position']
    ant_pos = EarthLocation.from_geocentric(geo_pos[0], geo_pos[1], geo_pos[2], 'meter')

    xds_data = locit_parms['xds_data']
    elevation_limit = locit_parms['elevation_limit'] * _convert_unit('deg', 'rad', 'trigonometric')
    pol = xds_data.attrs['polarization_scheme']

    if len(pol) > 2:
        msg = f'Polarization scheme {pol} is not what is expected for antenna based gains'
        logger.error(msg)
        raise Exception(msg)
    if locit_parms['polarization'] in pol:
        i_pol = np.where(np.array(pol) == locit_parms['polarization'])[0][0]
        gains = xds_data[f'P{i_pol}_PHASE_GAINS'].values
        time = getattr(xds_data, f'p{i_pol}_time').values
        field_id = xds_data[f'P{i_pol}_FIELD_ID'].values
    elif locit_parms['polarization'] == 'both':
        gains = np.concatenate([xds_data[f'P0_PHASE_GAINS'].values, xds_data[f'P1_PHASE_GAINS'].values])
        field_id = np.concatenate([xds_data[f'P0_FIELD_ID'].values, xds_data[f'P1_FIELD_ID'].values])
        time = np.concatenate([xds_data.p0_time.values, xds_data.p1_time.values])
    else:
        msg = f'Polarization {locit_parms["polarization"]} is not found in data'
        logger.error(msg)
        raise Exception(msg)

    astro_time = Time(time, format='mjd', scale='utc', location=ant_pos)
    lst = astro_time.sidereal_time("apparent").to(units.radian) / units.radian
    coordinates = _build_coordinate_array(field_id, src_dict, 'precessed', antenna['latitude'], time)
    coordinates[0, :] = lst.value - coordinates[0, :]

    # convert to actual hour angle

    linalg = locit_parms['fit_engine'] == 'linear algebra'
    if linalg:
        fit, variance = _solve_linear_algebra(coordinates, gains, elevation_limit, fit_kterm, fit_slope)
    else:
        if locit_parms['fit_engine'] == 'scipy':
            fit, variance = _solve_scipy_optimize_curve_fit(coordinates, gains, elevation_limit, fit_kterm, fit_slope,
                                                            verbose=True)
        else:
            msg = f'Unrecognized fitting engine: {locit_parms["fit_engine"]}'
            logger.erro(msg)
            raise Exception(msg)

    _print_eval_res(fit, variance, locit_parms['polarization'], clight/xds_data.attrs['frequency'])

    output_xds = xr.Dataset()
    output_xds.attrs['polarization'] = locit_parms['polarization']
    output_xds.attrs['wavelength'] = clight/xds_data.attrs['frequency']
    output_xds.attrs['position_fit'] = fit[1:4]
    output_xds.attrs['position_error'] = variance[1:4]
    output_xds.attrs['instrumental_delay_fit'] = fit[0]
    output_xds.attrs['instrumental_delay_error'] = variance[0]
    if fit_kterm and fit_slope:
        output_xds.attrs['kterm_fit'] = fit[4]
        output_xds.attrs['kterm_error'] = variance[4]
        output_xds.attrs['slope_fit'] = fit[5]
        output_xds.attrs['slope_error'] = variance[5]
    elif fit_kterm and not fit_slope:
        output_xds.attrs['koff_fit'] = fit[4]
        output_xds.attrs['koff_error'] = variance[4]
    elif not fit_kterm and fit_slope:
        output_xds.attrs['slope_fit'] = fit[4]
        output_xds.attrs['slope_error'] = variance[4]
    else:
        pass  # Nothing to be added to the attributes

    output_xds.attrs['antenna_info'] = antenna

    coords = {'time': coordinates[3, :]}
    output_xds['GAINS'] = xr.DataArray(gains, dims=['time'])
    output_xds['HOUR_ANGLE'] = xr.DataArray(coordinates[0, :], dims=['time'])
    output_xds['DECLINATION'] = xr.DataArray(coordinates[1, :], dims=['time'])
    output_xds['ELEVATION'] = xr.DataArray(coordinates[2, :], dims=['time'])
    output_xds['LST'] = xr.DataArray(lst, dims=['time'])

    basename = locit_parms['position_name']
    outname = "/".join([basename, 'ant_'+antenna['name'], f'{locit_parms["this_ddi"]}'])
    output_xds = output_xds.assign_coords(coords)
    output_xds.to_zarr(outname, mode="w", compute=True, consolidated=True)
    return


def _build_coordinate_array(field_id, src_list, key, latitude, time):
    """ If this is a bottleneck, good candidate for numba"""
    n_samples = len(field_id)
    coordinates = np.ndarray([4, n_samples])
    for i_sample in range(n_samples):
        field = str(field_id[i_sample])
        coordinates[0:2, i_sample] = src_list[field][key]
        coordinates[2, i_sample] = _hadec_to_elevation(src_list[field][key], latitude)
        coordinates[3, i_sample] = time[i_sample]-time[0]  # time is set to zero at the beginning of obs
    return coordinates


def _geometrical_coeffs(coordinates):
    ha, dec = coordinates[0:2]
    cosdec = np.cos(dec)
    xterm = twopi*np.cos(ha) * cosdec
    yterm = -twopi*np.sin(ha) * cosdec
    zterm = twopi*np.sin(dec)
    return [1.0, xterm, yterm, zterm]


def _kterm_coeff(coordinates):
    elevation = coordinates[2]
    return twopi*np.cos(elevation)


def _slope_coeff(coordinates):
    return coordinates[3]


def _solve_linear_algebra(coordinates, gains, elevation_limit, fit_kterm, fit_slope):
    """ If this is a bottleneck, good candidate for numba"""
    npar = 4 + fit_slope + fit_kterm
    if fit_kterm and fit_slope:
        coeff_function = _coeff_system_kterm_slope
    elif fit_kterm and not fit_slope:
        coeff_function = _coeff_system_kterm_noslope
    elif not fit_kterm and fit_slope:
        coeff_function = _coeff_system_nokterm_slope
    else:
        coeff_function = _coeff_system_nokterm_noslope

    system = np.zeros([npar, npar])
    vector = np.zeros([npar])
    n_samples = coordinates.shape[1]
    for i_sample in range(n_samples):
        if coordinates[2, i_sample] > elevation_limit:
            coeffs = coeff_function(coordinates[:, i_sample])
            for irow in range(npar):
                for icol in range(irow + 1):
                    system[irow, icol] += coeffs[irow] * coeffs[icol]
                vector[irow] += gains[i_sample] * coeffs[irow]
        else:
            pass

    for irow in range(1, npar):
        for icol in range(irow):
            system[icol, irow] = system[irow, icol]

    fit, variance, _ = _least_squares_fit(system, vector)
    return fit, variance


def _coeff_system_nokterm_noslope(coordinates):
    coeffs = _geometrical_coeffs(coordinates)
    return coeffs


def _coeff_system_kterm_noslope(coordinates):
    coeffs = _geometrical_coeffs(coordinates)
    coeffs.append(_kterm_coeff(coordinates))
    return coeffs


def _coeff_system_nokterm_slope(coordinates):
    coeffs = _geometrical_coeffs(coordinates)
    coeffs.append(_slope_coeff(coordinates))
    return coeffs


def _coeff_system_kterm_slope(coordinates):
    coeffs = _geometrical_coeffs(coordinates)
    coeffs.append(_kterm_coeff(coordinates))
    coeffs.append(_slope_coeff(coordinates))
    return coeffs


def _solve_scipy_optimize_curve_fit(coordinates, gains, elevation_limit, fit_kterm, fit_slope, verbose=False):
    logger  = _get_astrohack_logger()
    selelev = coordinates[2, :] > elevation_limit
    coordinates = coordinates[:, selelev]
    gains = gains[selelev]

    npar = 4 + fit_slope + fit_kterm
    if fit_kterm and fit_slope:
        func_function = _phase_model_kterm_slope
    elif fit_kterm and not fit_slope:
        func_function = _phase_model_kterm_noslope
    elif not fit_kterm and fit_slope:
        func_function = _phase_model_nokterm_slope
    else:
        func_function = _phase_model_nokterm_noslope

    # First guess is no error in positions and no instrumental delay
    p0 = np.zeros(npar)
    p0[0] = 1
    # Position Errors, k term and phase slope are not constrained, but the instrumental delay is pegged to the
    # -pi to pi range
    liminf = np.full(npar, -np.inf)
    limsup = np.full(npar, +np.inf)
    liminf[0] = -pi
    limsup[0] = +pi

    maxfevs = [100000, 1000000, 10000000]
    for maxfev in maxfevs:
        try:
            fit, covar = opt.curve_fit(func_function, coordinates, gains, p0=p0, bounds=[liminf, limsup], maxfev=maxfev)
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


def _phase_model_nokterm_noslope(coordinates,  inst_delay, xoff, yoff, zoff):
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    return xterm + yterm + zterm + inst_delay


def _phase_model_kterm_noslope(coordinates, inst_delay, xoff, yoff, zoff, koff):
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    kterm = _kterm_coeff(coordinates) * koff
    return xterm + yterm + zterm + inst_delay + kterm


def _phase_model_nokterm_slope(coordinates, inst_delay, xoff, yoff, zoff, slope):
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    sterm = _slope_coeff(coordinates) * slope
    return xterm + yterm + zterm + inst_delay + sterm


def _phase_model_kterm_slope(coordinates, inst_delay, xoff, yoff, zoff, koff, slope):
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    sterm = _slope_coeff(coordinates) * slope
    kterm = _kterm_coeff(coordinates) * koff
    return xterm + yterm + zterm + inst_delay + kterm + sterm


def _print_eval_res(fit, variance, fittype, wavelength):
    print(80 * '*')
    print(fittype)
    fitstr = 'fit: '
    errstr = 'err: '
    for i in range(len(fit)):
        if 0 < i < 4:
            fitstr += f'{fit[i]*wavelength:16.8f}'
            errstr += f'{variance[i]*wavelength:16.8f}'
        else:
            fitstr += f'{fit[i]*180/pi:16.8f}'
            errstr += f'{variance[i]*180/pi:16.8f}'
    print(fitstr)
    print(errstr)


