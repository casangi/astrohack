from prettytable import PrettyTable
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy import optimize as opt

import graphviper.utils.logger as logger
import astropy.units as units
import xarray as xr
import numpy as np

from astrohack._utils._panel_classes.telescope import Telescope
from astrohack._utils._locit_commons import _get_telescope_lat_lon_rad, _compute_antenna_relative_off
from astrohack._utils._locit_commons import _time_label, _elevation_label, _declination_label
from astrohack._utils._locit_commons import _plot_antenna_position
from astrohack._utils._plot_commons import _create_figure_and_axes, _close_figure, _scatter_plot
from astrohack._utils._locit_commons import _plot_boxes_limits_and_labels, _plot_corrections, _hour_angle_label
from astrohack._utils._tools import _hadec_to_elevation, _format_value_error, _param_to_list, _add_prefix
from astrohack._utils._conversion import _convert_unit
from astrohack._utils._algorithms import _least_squares_fit
from astrohack._utils._constants import *


def _locit_separated_chunk(locit_parms):
    """
    This is the chunk function for locit when treating each DDI separately
    Args:
        locit_parms: the locit parameter dictionary

    Returns:
    xds save to disk in the .zarr format
    """
    xds_data = locit_parms['xds_data']
    field_id, time, delays, freq = _get_data_from_locit_xds(xds_data, locit_parms['polarization'])

    coordinates, delays, lst, elevation_limit = _build_filtered_arrays(field_id, time, delays, locit_parms)

    
    if len(delays) == 0:
        msg = f'{locit_parms["this_ant"]} {locit_parms["this_ddi"]} has no valid data, skipping'
        logger.warning(msg)
        return

    fit, variance = _fit_data(coordinates, delays, locit_parms)
    model, chi_squared = _compute_chi_squared(delays, fit, coordinates, locit_parms['fit_kterm'],
                                              locit_parms['fit_delay_rate'])
    _create_output_xds(coordinates, lst, delays, fit, variance, chi_squared, model, locit_parms, freq, elevation_limit)
    return


def _locit_combined_chunk(locit_parms):
    """
    This is the chunk function for locit when we are combining the DDIs for an antenna for a single solution
    Args:
        locit_parms: the locit parameter dictionary

    Returns:
    xds save to disk in the .zarr format
    """
    data = locit_parms['data_dict']

    delay_list = []
    time_list = []
    field_list = []
    freq_list = []
    for ddi, xds_data in data.items():
        this_field_id, this_time, this_delays, freq = _get_data_from_locit_xds(xds_data, locit_parms['polarization'])
        freq_list.append(freq)
        field_list.append(this_field_id)
        time_list.append(this_time)
        delay_list.append(this_delays)

    delays = np.concatenate(delay_list)
    time = np.concatenate(time_list)
    field_id = np.concatenate(field_list)

    coordinates, delays, lst, elevation_limit = _build_filtered_arrays(field_id, time, delays, locit_parms)

    
    if len(delays) == 0:
        msg = f'{locit_parms["this_ant"]} {locit_parms["this_ddi"]} has no valid data, skipping'
        logger.warning(msg)
        return

    fit, variance = _fit_data(coordinates, delays, locit_parms)
    model, chi_squared = _compute_chi_squared(delays, fit, coordinates, locit_parms['fit_kterm'],
                                              locit_parms['fit_delay_rate'])
    _create_output_xds(coordinates, lst, delays, fit, variance, chi_squared, model, locit_parms, freq_list,
                       elevation_limit)
    return


def _locit_difference_chunk(locit_parms):
    """
    This is the chunk function for locit when we are combining two DDIs for an antenna for a single solution by using
    the difference in phase between the two DDIs of different frequencies
    Args:
        locit_parms: the locit parameter dictionary

    Returns:
    xds save to disk in the .zarr format
    """
    data = locit_parms['data_dict']
    ddi_list = list(data.keys())
    nddis = len(ddi_list)
    
    if nddis != 2:
        msg = f'The difference method support only 2 DDIs, {nddis} DDIs provided.'
        logger.error(msg)
        return

    ddi_0 = _get_data_from_locit_xds(data[ddi_list[0]], locit_parms['polarization'], get_phases=True, split_pols=True)
    ddi_1 = _get_data_from_locit_xds(data[ddi_list[1]], locit_parms['polarization'], get_phases=True, split_pols=True)

    time, field_id, delays, freq = _delays_from_phase_differences(ddi_0, ddi_1,
                                                                  multi_pol=locit_parms['polarization'] == 'both')

    coordinates, delays, lst, elevation_limit = _build_filtered_arrays(field_id, time, delays, locit_parms)
    
    if len(delays) == 0:
        msg = f'{locit_parms["this_ant"]} {locit_parms["this_ddi"]} has no valid data, skipping'
        logger.warning(msg)
        return
    fit, variance = _fit_data(coordinates, delays, locit_parms)
    model, chi_squared = _compute_chi_squared(delays, fit, coordinates, locit_parms['fit_kterm'],
                                              locit_parms['fit_delay_rate'])
    _create_output_xds(coordinates, lst, delays, fit, variance, chi_squared, model, locit_parms, freq,
                       elevation_limit)
    return


def _delays_from_phase_differences(ddi_0, ddi_1, multi_pol=False):
    """
    Compute delays from the difference in phase between two DDIs of different frequencies
    Args:
        ddi_0: First DDI
        ddi_1: Second DDI
        multi_pol: is the DDI data split by polarization?

    Returns:
    Matched times, matched field ids, matched phase difference delays, difference in frequency
    """
    
    freq = ddi_0[3] - ddi_1[3]
    fields = ddi_0[0]
    if freq > 0:
        pos_time, pos_phase = ddi_0[1:3]
        neg_time, neg_phase = ddi_1[1:3]
    elif freq < 0:
        pos_time, pos_phase = ddi_1[1:3]
        neg_time, neg_phase = ddi_0[1:3]
        freq *= -1
    else:
        msg = f'The two DDIs must have different frequencies'
        logger.error(msg)
        raise Exception(msg)

    if multi_pol:
        time = []
        field_id = []
        phase = []
        for i_pol in range(2):
            this_time, this_field_id, this_phase = _match_times_and_phase_difference(pos_time[i_pol], neg_time[i_pol],
                                                                                     pos_phase[i_pol], neg_phase[i_pol],
                                                                                     fields[i_pol])
            time.append(this_time)
            field_id.append(this_field_id)
            phase.append(this_phase)

        time = np.concatenate(time)
        field_id = np.concatenate(field_id)
        phase = np.concatenate(phase)

    else:
        time, field_id, phase = _match_times_and_phase_difference(pos_time, neg_time, pos_phase, neg_phase, fields)

    delays = phase/twopi/freq
    return time, field_id, delays, freq


def _match_times_and_phase_difference(pos_time, neg_time, pos_phase, neg_phase, fields, tolerance=1e-8):
    """
    match times and compute the phase differences for the simple case, calls _different_times for the complicated case
    Args:
        pos_time: Time for the positive phase
        neg_time: Time for the negative phase
        pos_phase: Positive phase
        neg_phase: Negative phase
        fields: Field ids
        tolerance: Tolerance in time to match time arrays

    Returns:
    Matched times, matched field ids, -pi, pi wrapped matched phase difference
    """
    n_pos_time, n_neg_time = len(pos_time), len(neg_time)
    if n_pos_time == n_neg_time:
        if np.all(np.isclose(pos_time, neg_time, tolerance)):  # this the simplest case times are already matched!
            return pos_time, fields, _phase_wrapping(pos_phase - neg_phase)
        else:
            return _different_times(pos_time, neg_time, pos_phase, neg_phase, fields, tolerance)
    else:
        return _different_times(pos_time, neg_time, pos_phase, neg_phase, fields, tolerance)


def _different_times(pos_time, neg_time, pos_phase, neg_phase, fields, tolerance=1e-8):
    """
    match times and compute the phase differences for the complicated case
    Args:
        pos_time: Time for the positive phase
        neg_time: Time for the negative phase
        pos_phase: Positive phase
        neg_phase: Negative phase
        fields: Field ids
        tolerance: Tolerance in time to match time arrays

    Returns:
    Matched times, matched field ids, -pi, pi wrapped matched phase difference
    """
    # This solution is not optimal but numpy does not have a task for it, if it ever becomes a bottleneck we can JIT it
    out_times = np.sort([time for time in pos_time if np.isclose(neg_time, time, tolerance).any()])
    ntimes = out_times.shape[0]
    out_phase = np.ndarray(ntimes)
    out_field = np.ndarray(ntimes, dtype=np.integer)
    for i_time in range(ntimes):
        i_t0 = abs(pos_time - out_times[i_time]) < tolerance
        i_t1 = abs(neg_time - out_times[i_time]) < tolerance
        out_phase[i_time] = pos_phase[i_t0][0] - neg_phase[i_t1][0]
        out_field[i_time] = fields[i_t0][0]
    return out_times, out_field, _phase_wrapping(out_phase)


def _phase_wrapping(phase):
    """
    Wraps phase to the -pi to pi interval
    Args:
        phase: phase to be wrapped

    Returns:
    Phase wrapped to the -pi to pi interval
    """
    return (phase + pi) % (2 * pi) - pi


def _get_data_from_locit_xds(xds_data, pol_selection, get_phases=False, split_pols=False):
    """
    Extract data from a .locit.zarr xds, converts the phase gains to delays using the xds frequency
    Args:
        xds_data: The .locit.zarr xds
        pol_selection: Which polarization is requested from the xds
        get_phases: return phases rather than delays
        split_pols: Different polarizations are not concatenated in a single array if True


    Returns:
        the field ids
        the time in mjd
        The delays in seconds or phases in radians
        Xds frequency

    """
    
    pol = xds_data.attrs['polarization_scheme']
    freq = xds_data.attrs['frequency']
    if len(pol) != 2:
        msg = f'Polarization scheme {pol} is not what is expected for antenna based gains'
        logger.error(msg)
        raise Exception(msg)
    if pol_selection in pol:
        i_pol = np.where(np.array(pol) == pol_selection)[0][0]
        phases = xds_data[f'P{i_pol}_PHASE_GAINS'].values
        time = getattr(xds_data, f'p{i_pol}_time').values
        field_id = xds_data[f'P{i_pol}_FIELD_ID'].values
    elif pol_selection == 'both':
        phases = [xds_data[f'P0_PHASE_GAINS'].values, xds_data[f'P1_PHASE_GAINS'].values]
        field_id = [xds_data[f'P0_FIELD_ID'].values, xds_data[f'P1_FIELD_ID'].values]
        time = [xds_data.p0_time.values, xds_data.p1_time.values]
        if not split_pols:
            phases = np.concatenate(phases)
            field_id = np.concatenate(field_id)
            time = np.concatenate(time)
    else:
        msg = f'Polarization {pol_selection} is not found in data'
        logger.error(msg)
        raise Exception(msg)
    if get_phases:
        return field_id, time, phases, freq  # field_id, time, phases, frequency
    else:
        return field_id, time, phases/twopi/freq, freq  # field_id, time, delays, frequency


def _create_output_xds(coordinates, lst, delays, fit, variance, chi_squared, model, locit_parms, frequency,
                       elevation_limit):
    """
    Create the output xds from the computed quantities and the fit results
    Args:
        coordinates: The coordinate array used in the fitting
        lst: The local sidereal time
        delays: The fitted delays
        fit: The fit results
        variance: the fit error bars
        locit_parms: the input parameters
        frequency: The frequency or frequencies of the input xds or xdses
        elevation_limit: the elevation cutoff

    Returns:
    The xds on zarr format on disk
    """
    fit_kterm = locit_parms['fit_kterm']
    fit_rate = locit_parms['fit_delay_rate']
    antenna = locit_parms['ant_info'][locit_parms['this_ant']]
    error = np.sqrt(variance)

    output_xds = xr.Dataset()
    output_xds.attrs['polarization'] = locit_parms['polarization']
    output_xds.attrs['frequency'] = frequency
    output_xds.attrs['position_fit'] = fit[1:4]
    output_xds.attrs['position_error'] = error[1:4]
    output_xds.attrs['fixed_delay_fit'] = fit[0]
    output_xds.attrs['fixed_delay_error'] = error[0]
    output_xds.attrs['antenna_info'] = antenna
    output_xds.attrs['elevation_limit'] = elevation_limit
    output_xds.attrs['chi_squared'] = chi_squared

    if fit_kterm and fit_rate:
        output_xds.attrs['koff_fit'] = fit[4]
        output_xds.attrs['koff_error'] = error[4]
        output_xds.attrs['rate_fit'] = fit[5]
        output_xds.attrs['rate_error'] = error[5]
    elif fit_kterm and not fit_rate:
        output_xds.attrs['koff_fit'] = fit[4]
        output_xds.attrs['koff_error'] = error[4]
    elif not fit_kterm and fit_rate:
        output_xds.attrs['rate_fit'] = fit[4]
        output_xds.attrs['rate_error'] = error[4]
    else:
        pass  # Nothing to be added to the attributes

    coords = {'time': coordinates[3, :]}
    output_xds['DELAYS'] = xr.DataArray(delays, dims=['time'])
    output_xds['MODEL'] = xr.DataArray(model, dims=['time'])
    output_xds['HOUR_ANGLE'] = xr.DataArray(coordinates[0, :], dims=['time'])
    output_xds['DECLINATION'] = xr.DataArray(coordinates[1, :], dims=['time'])
    output_xds['ELEVATION'] = xr.DataArray(coordinates[2, :], dims=['time'])
    output_xds['LST'] = xr.DataArray(lst, dims=['time'])

    basename = locit_parms['position_name']
    outname = "/".join([basename, 'ant_'+antenna['name']])
    if locit_parms['combine_ddis'] == 'no':
        outname += "/"+f'{locit_parms["this_ddi"]}'
    output_xds = output_xds.assign_coords(coords)
    output_xds.to_zarr(outname, mode="w", compute=True, consolidated=True)


def _fit_data(coordinates, delays, locit_parms):
    """
    Execute the fitting using the desired engine, scipy or linear algebra
    Args:
        coordinates: the shape [4, : ] array with the ha, dec, elevation and time arrays
        delays: The delays to be fitted
        locit_parms: the locit input paramters

    Returns:
    fit: the fit results
    variance: the diagonal of the covariance matrix
    """
    
    fit_kterm = locit_parms['fit_kterm']
    fit_rate = locit_parms['fit_delay_rate']

    linalg = locit_parms['fit_engine'] == 'linear algebra'
    if linalg:
        fit, variance = _solve_linear_algebra(coordinates, delays, fit_kterm, fit_rate)
    else:
        if locit_parms['fit_engine'] == 'scipy':
            fit, variance = _solve_scipy_optimize_curve_fit(coordinates, delays, fit_kterm, fit_rate, verbose=True)
        else:
            msg = f'Unrecognized fitting engine: {locit_parms["fit_engine"]}'
            logger.error(msg)
            raise Exception(msg)
    return fit, variance


def _compute_chi_squared(delays, fit, coordinates, fit_kterm, fit_rate):
    """
    Compute a model from fit results and computes the chi squared value of that model with respect to the data
    Args:
        delays: The observed delays
        fit: The fit results
        coordinates: ha, dec, elevation, time
        fit_kterm: K term fitted?
        fit_rate: delay rate fitted?

    Returns:
    The delay model and the chi squared value
    """
    model_function, _ = _define_fit_function(fit_kterm, fit_rate)
    model = model_function(coordinates, *fit)
    n_delays = len(delays)
    chi_squared = np.sum((model-delays)**2/n_delays)
    return model, chi_squared


def _build_filtered_arrays(field_id, time, delays, locit_parms):
    """Build the coordinate arrays (ha, dec, elevation, time) for use in the fitting and filters data below the elevation limit

    Args:
        field_id: Array with the observed field per delay
        time: Time array with the time of each delay
        delays: The delay array
        locit_parms: Locit main function parameters

    Returns:
    coordinates (ha, dec, ele, time), delays, local sidereal time all filtered by elevation limit and the elevation_limit
    """
    """ Build the coordinate arrays (ha, dec, elevation, angle) for use in the fitting"""
    elevation_limit = locit_parms['elevation_limit'] * _convert_unit('deg', 'rad', 'trigonometric')
    antenna = locit_parms['ant_info'][locit_parms['this_ant']]
    src_list = locit_parms['obs_info']['src_dict']
    geo_pos = antenna['geocentric_position']
    ant_pos = EarthLocation.from_geocentric(geo_pos[0], geo_pos[1], geo_pos[2], 'meter')
    astro_time = Time(time, format='mjd', scale='utc', location=ant_pos)
    lst = astro_time.sidereal_time("apparent").to(units.radian) / units.radian
    key = 'precessed'

    n_samples = len(field_id)
    coordinates = np.ndarray([4, n_samples])
    for i_sample in range(n_samples):
        field = str(field_id[i_sample])
        coordinates[0:2, i_sample] = src_list[field][key]
        coordinates[2, i_sample] = _hadec_to_elevation(src_list[field][key], antenna['latitude'])
        coordinates[3, i_sample] = time[i_sample]-time[0]  # time is set to zero at the beginning of obs

    # convert to actual hour angle and wrap it to the [-pi, pi) interval
    coordinates[0, :] = lst.value - coordinates[0, :]
    coordinates[0, :] = np.where(coordinates[0, :] < 0, coordinates[0, :] + twopi, coordinates[0, :])

    # Filter data below elevation limit
    selection = coordinates[2, :] > elevation_limit
    delays = delays[selection]
    coordinates = coordinates[:, selection]
    lst = lst[selection]

    return coordinates, delays, lst, elevation_limit


def _geometrical_coeffs(coordinates):
    """
    Compute the position related coefficients for the fitting, also the 1 corresponding to the fixed delay
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)

    Returns:
    the fixed delay coefficient (1), the x, y and z position delay coeffcients
    """
    ha, dec = coordinates[0:2]
    cosdec = np.cos(dec)
    xterm = np.cos(ha) * cosdec
    yterm = -np.sin(ha) * cosdec
    zterm = np.sin(dec)
    return [1.0, xterm, yterm, zterm]


def _kterm_coeff(coordinates):
    """ Compute the k term (offset from antenna elevation axis) coefficient from elevation

    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)

    Returns:
    The offset from antenna elevation axis delay coefficient
    """
    elevation = coordinates[2]
    return np.cos(elevation)


def _rate_coeff(coordinates):
    """Compute the delay rate coefficient (basically the time)

    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)

    Returns:
    The delay rate coeeficient (time)
    """
    return coordinates[3]


def _solve_linear_algebra(coordinates, delays, fit_kterm, fit_rate):
    """

    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        delays: The delays
        fit_kterm: fit elevation axis offset term
        fit_rate: fit delay rate term

    Returns:
    The fit results and the diagonal of the covariance matrix.
    """
    npar = 4 + fit_rate + fit_kterm

    system = np.zeros([npar, npar])
    vector = np.zeros([npar])
    n_samples = coordinates.shape[1]
    for i_sample in range(n_samples):
        coeffs = _system_coefficients(coordinates[:, i_sample], fit_kterm, fit_rate)
        for irow in range(npar):
            for icol in range(irow + 1):
                system[irow, icol] += coeffs[irow] * coeffs[icol]
            vector[irow] += delays[i_sample] * coeffs[irow]

    for irow in range(1, npar):
        for icol in range(irow):
            system[icol, irow] = system[irow, icol]

    fit, variance, _ = _least_squares_fit(system, vector)

    return fit, variance


def _system_coefficients(coordinates, fit_kterm, fit_rate):
    """ Build coefficient list for linear algebra fit

    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        fit_kterm: fit elevation axis offset term
        fit_rate: Fit delay rate term

    Returns:

    """
    coeffs = _geometrical_coeffs(coordinates)
    if fit_kterm:
        coeffs.append(_kterm_coeff(coordinates))
    if fit_rate:
        coeffs.append(_rate_coeff(coordinates))
    return coeffs


def _define_fit_function(fit_kterm, fit_rate):
    """
    Define the fitting function based on the presence of the delay rate and elevation axis offset terms
    Args:
        fit_kterm: fit elevation axis offset?
        fit_rate: fit delay rate?

    Returns:
    The appropriate fitting function and the total number of parameters
    """
    npar = 4 + fit_rate + fit_kterm
    if fit_kterm and fit_rate:
        fit_function = _delay_model_kterm_rate
    elif fit_kterm and not fit_rate:
        fit_function = _delay_model_kterm_norate
    elif not fit_kterm and fit_rate:
        fit_function = _delay_model_nokterm_rate
    else:
        fit_function = _delay_model_nokterm_norate
    return fit_function, npar


def _solve_scipy_optimize_curve_fit(coordinates, delays, fit_kterm, fit_rate, verbose=False):
    """
    Fit a delay model to the observed delays using scipy optimize curve_fit algorithm
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        delays: The observed delays
        fit_kterm: fit elevation axis offset term
        fit_rate: Fit delay rate term
        verbose: Display fitting messages

    Returns:
    The fit results and the diagonal of the covariance matrix
    """
    

    fit_function, npar = _define_fit_function(fit_kterm, fit_rate)

    # First guess is no errors in positions, no fixed delay and no delay rate
    p0 = np.zeros(npar)
    liminf = np.full(npar, -np.inf)
    limsup = np.full(npar, +np.inf)

    maxfevs = [100000, 1000000, 10000000]
    for maxfev in maxfevs:
        try:
            fit, covar = opt.curve_fit(fit_function, coordinates, delays, p0=p0, bounds=[liminf, limsup], maxfev=maxfev)
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


def _delay_model_nokterm_norate(coordinates, fixed_delay, xoff, yoff, zoff):
    """
    Delay model with no elevation axis offset or delay rate
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        fixed_delay: Fixed delay value
        xoff: X direction delay in antenna frame
        yoff: Y direction delay in antenna frame
        zoff: Z direction delay in antenna frame

    Returns:
    Delays model at coordinates
    """
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    return xterm + yterm + zterm + fixed_delay


def _delay_model_kterm_norate(coordinates, fixed_delay, xoff, yoff, zoff, koff):
    """
    Delay model with elevation axis offset and no delay rate
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        fixed_delay: Fixed delay value
        xoff: X direction delay in antenna frame
        yoff: Y direction delay in antenna frame
        zoff: Z direction delay in antenna frame
        koff: Elevation axis offset delay

    Returns:
    Delays model at coordinates
    """
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    kterm = _kterm_coeff(coordinates) * koff
    return xterm + yterm + zterm + fixed_delay + kterm


def _delay_model_nokterm_rate(coordinates, fixed_delay, xoff, yoff, zoff, rate):
    """
    Delay model with delay rate and no elevation axis offset
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        fixed_delay: Fixed delay value
        xoff: X direction delay in antenna frame
        yoff: Y direction delay in antenna frame
        zoff: Z direction delay in antenna frame
        rate: delay rate

    Returns:
    Delays model at coordinates
    """
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    sterm = _rate_coeff(coordinates) * rate
    return xterm + yterm + zterm + fixed_delay + sterm


def _delay_model_kterm_rate(coordinates, fixed_delay, xoff, yoff, zoff, koff, rate):
    """
    Delay model with delay rate and elevation axis offset
    Args:
        coordinates: coordinate arrays (ha, dec, ele, time)
        fixed_delay: Fixed delay value
        xoff: X direction delay in antenna frame
        yoff: Y direction delay in antenna frame
        zoff: Z direction delay in antenna frame
        koff: Elevation axis offset delay
        rate: delay rate

    Returns:
    Delays model at coordinates
    """
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    sterm = _rate_coeff(coordinates) * rate
    kterm = _kterm_coeff(coordinates) * koff
    return xterm + yterm + zterm + fixed_delay + kterm + sterm


def _export_fit_results(data_dict, parm_dict):
    """
    Export fit results to a txt file listing the different DDIs as different solutions if data is not combined
    Args:
        data_dict: the mds content
        parm_dict: Dictionary of the parameters given to the calling function

    Returns:
    text file with fit results in convenient units
    """
    pos_unit = parm_dict['position_unit']
    del_unit = parm_dict['delay_unit']
    len_fact = _convert_unit('m', pos_unit, 'length')
    del_fact = _convert_unit('sec', del_unit, kind='time')
    pos_fact = len_fact * clight
    combined = parm_dict['combined']

    if combined:
        field_names = ['Antenna', f'RMS [{del_unit}]', f'F. delay [{del_unit}]', f'X offset [{pos_unit}]',
                       f'Y offset [{pos_unit}]', f'Z offset [{pos_unit}]']
        specifier = 'combined_'+data_dict._meta_data['combine_ddis']

    else:
        field_names = ['Antenna', 'DDI', f'RMS [{del_unit}]', f'F. delay [{del_unit}]', f'X offset [{pos_unit}]',
                       f'Y offset [{pos_unit}]', f'Z offset [{pos_unit}]']
        specifier = 'separated_ddis'
    kterm_present = data_dict._meta_data["fit_kterm"]
    rate_present = data_dict._meta_data['fit_delay_rate']
    if kterm_present:
        field_names.extend([f'K offset [{pos_unit}]'])
    if rate_present:
        tim_unit = parm_dict['time_unit']
        slo_unit = f'{del_unit}/{tim_unit}'
        slo_fact = del_fact / _convert_unit('day', tim_unit, 'time')
        field_names.extend([f'Rate [{slo_unit}]'])
    else:
        slo_unit = notavail
        slo_fact = 1.0

    table = PrettyTable()
    table.field_names = field_names
    table.align = 'c'
    full_antenna_list = Telescope(data_dict._meta_data['telescope_name']).ant_list
    selected_antenna_list = _param_to_list(parm_dict['ant'], data_dict, 'ant')

    for ant_name in full_antenna_list:
        ant_key = _add_prefix(ant_name, 'ant')
        row = [ant_name]
        if ant_key in selected_antenna_list:
            if ant_key in data_dict.keys():
                if ant_name == data_dict._meta_data['reference_antenna']:
                    ant_name += ' (ref)'

                antenna = data_dict[ant_key]
                if combined:
                    table.add_row(_export_xds(row, antenna.attrs, del_fact, pos_fact, slo_fact, kterm_present,
                                              rate_present))
                else:
                    ddi_list = _param_to_list(parm_dict['ddi'], data_dict[ant_key], 'ddi')
                    for ddi_key in ddi_list:
                        row = [ant_name, ddi_key.split('_')[1]]
                        table.add_row(_export_xds(row, data_dict[ant_key][ddi_key].attrs, del_fact, pos_fact, slo_fact,
                                                  kterm_present, rate_present))

    outname = parm_dict['destination']+f'/position_{specifier}_fit_results.txt'
    outfile = open(outname, 'w')
    outfile.write(table.get_string()+'\n')
    outfile.close()


def _export_xds(row, attributes, del_fact, pos_fact, slo_fact, kterm_present, rate_present):
    """
    Export the data from a single X array DataSet attributes to a table row (a list)
    Args:
        row: row onto which the data results are to be added
        attributes: The XDS attributes dictionary
        del_fact: Delay unit scaling factor
        pos_fact: Position unit scaling factor
        slo_fact: Delay rate unit scaling factor
        kterm_present: Is the elevation axis offset term present?
        rate_present: Is the delay rate term present?

    Returns:
    The filled table row
    """
    tolerance = 1e-4

    rms = np.sqrt(attributes["chi_squared"])*del_fact
    row.append(f'{rms:.2e}')
    row.append(_format_value_error(attributes['fixed_delay_fit'], attributes['fixed_delay_error'], del_fact,
               tolerance))
    position, poserr = _rotate_to_gmt(np.copy(attributes['position_fit']), attributes['position_error'],
                                      attributes['antenna_info']['longitude'])
    for i_pos in range(3):
        row.append(_format_value_error(position[i_pos], poserr[i_pos],  pos_fact, tolerance))
    if kterm_present:
        row.append(_format_value_error(attributes['koff_fit'], attributes['koff_error'], pos_fact, tolerance))
    if rate_present:
        row.append(_format_value_error(attributes['rate_fit'], attributes['rate_error'], slo_fact, tolerance))
    return row


def _plot_sky_coverage_chunk(parm_dict):
    """
    Plot the sky coverage for a XDS
    Args:
        parm_dict: Parameter dictionary from the caller function enriched with the XDS data

    Returns:
    PNG file with the sky coverage
    """
    
    combined = parm_dict['combined']
    antenna = parm_dict['this_ant']
    destination = parm_dict['destination']

    if combined:
        export_name = f'{destination}/position_sky_coverage_{antenna}.png'
        suptitle = f'Sky coverage for antenna {antenna.split("_")[1]}'
    else:
        ddi = parm_dict['this_ddi']
        export_name = f'{destination}/position_sky_coverage_{antenna}_{ddi}.png'
        suptitle = f'Sky coverage for antenna {antenna.split("_")[1]}, DDI {ddi.split("_")[1]}'

    xds = parm_dict['xds_data']
    figuresize = parm_dict['figure_size']
    angle_unit = parm_dict['angle_unit']
    time_unit = parm_dict['time_unit']
    display = parm_dict['display']
    dpi = parm_dict['dpi']
    antenna_info = xds.attrs['antenna_info']

    time = xds.time.values * _convert_unit('day', time_unit, 'time')
    angle_fact = _convert_unit('rad', angle_unit, 'trigonometric')
    ha = xds['HOUR_ANGLE'] * angle_fact
    dec = xds['DECLINATION'] * angle_fact
    ele = xds['ELEVATION'] * angle_fact

    fig, axes = _create_figure_and_axes(figuresize, [2, 2])

    elelim, elelines, declim, declines, halim = _plot_borders(angle_fact, antenna_info['latitude'],
                                                              xds.attrs['elevation_limit'])
    timelabel = _time_label(time_unit)
    halabel = _hour_angle_label(angle_unit)
    declabel = _declination_label(angle_unit)
    _scatter_plot(axes[0, 0], time, timelabel, ele, _elevation_label(angle_unit), 'Time vs Elevation', ylim=elelim,
                  hlines=elelines)
    _scatter_plot(axes[0, 1], time, timelabel, ha, halabel, 'Time vs Hour angle', ylim=halim)
    _scatter_plot(axes[1, 0], time, timelabel, dec, declabel, 'Time vs Declination', ylim=declim, hlines=declines)
    _scatter_plot(axes[1, 1], ha, halabel, dec, declabel, 'Hour angle vs Declination', ylim=declim, xlim=halim,
                  hlines=declines)

    _close_figure(fig, suptitle, export_name, dpi, display)
    return


def _plot_delays_chunk(parm_dict):
    """
    Plot the delays and optionally the delay model for a XDS
    Args:
        parm_dict: Parameter dictionary from the caller function enriched with the XDS data

    Returns:
    PNG file with the delay plots
    """
    combined = parm_dict['combined']
    plot_model = parm_dict['plot_model']
    antenna = parm_dict['this_ant']
    destination = parm_dict['destination']
    if combined:
        export_name = f'{destination}/position_delays_{antenna}_combined_{parm_dict["comb_type"]}.png'
        suptitle = f'Delays for antenna {antenna.split("_")[1]}'
    else:
        ddi = parm_dict['this_ddi']
        export_name = f'{destination}/position_delays_{antenna}_separated_{ddi}.png'
        suptitle = f'Delays for antenna {antenna.split("_")[1]}, DDI {ddi.split("_")[1]}'

    xds = parm_dict['xds_data']
    figuresize = parm_dict['figure_size']
    angle_unit = parm_dict['angle_unit']
    time_unit = parm_dict['time_unit']
    delay_unit = parm_dict['delay_unit']
    display = parm_dict['display']
    dpi = parm_dict['dpi']
    antenna_info = xds.attrs['antenna_info']

    time = xds.time.values * _convert_unit('day', time_unit, 'time')
    angle_fact = _convert_unit('rad', angle_unit, 'trigonometric')
    delay_fact = _convert_unit('sec', delay_unit, kind='time')
    ha = xds['HOUR_ANGLE'] * angle_fact
    dec = xds['DECLINATION'] * angle_fact
    ele = xds['ELEVATION'] * angle_fact
    delays = xds['DELAYS'].values * delay_fact

    elelim, elelines, declim, declines, halim = _plot_borders(angle_fact, antenna_info['latitude'],
                                                              xds.attrs['elevation_limit'])
    delay_minmax = [np.min(delays), np.max(delays)]
    delay_border = 0.05*(delay_minmax[1]-delay_minmax[0])
    delaylim = [delay_minmax[0]-delay_border, delay_minmax[1]+delay_border]

    fig, axes = _create_figure_and_axes(figuresize, [2, 2])

    ylabel = f'Delays [{delay_unit}]'
    if plot_model:
        model = xds['MODEL'].values * delay_fact
    else:
        model = None
    _scatter_plot(axes[0, 0], time, _time_label(time_unit), delays, ylabel, 'Time vs Delays', ylim=delaylim,
                  model=model)
    _scatter_plot(axes[0, 1], ele, _elevation_label(angle_unit), delays, ylabel, 'Elevation vs Delays',
                  xlim=elelim, vlines=elelines, ylim=delaylim, model=model)
    _scatter_plot(axes[1, 0], ha, _hour_angle_label(angle_unit), delays, ylabel, 'Hour Angle vs Delays', xlim=halim,
                  ylim=delaylim, model=model)
    _scatter_plot(axes[1, 1], dec, _declination_label(angle_unit), delays, ylabel, 'Declination vs Delays',
                  xlim=declim, vlines=declines, ylim=delaylim, model=model)

    _close_figure(fig, suptitle, export_name, dpi, display)
    return


def _plot_borders(angle_fact, latitude, elevation_limit):
    """
    Compute plot limits and position of lines to be added to the plots
    Args:
        angle_fact: Angle scaling unit factor
        latitude: Antenna latitude
        elevation_limit: The elevation limit in the data set

    Returns:
    Elevation limits, elevation lines, declination limits, declination lines and hour angle limits
    """
    latitude *= angle_fact
    elevation_limit *= angle_fact
    right_angle = pi/2*angle_fact
    border = 0.05 * right_angle
    elelim = [-border-right_angle/2, right_angle+border]
    border *= 2
    declim = [-border-right_angle, right_angle+border]
    border *= 2
    halim = [-border, 4*right_angle+border]
    elelines = [0, elevation_limit]  # lines at zero and elevation limit
    declines = [latitude-right_angle, latitude+right_angle]
    return elelim, elelines, declim, declines, halim


def _rotate_to_gmt(positions, errors, longitude):
    """
    Rotate geometrical delays from antenna reference frame to GMT reference frame
    Args:
        positions: geometrical delays
        errors: geometrical delay errors
        longitude: Antenna longitude

    Returns:
    Rotated geometrical delays and associated errors
    """
    xpos, ypos = positions[0:2]
    delta_lon = longitude
    cosdelta = np.cos(delta_lon)
    sindelta = np.sin(delta_lon)
    newpositions = positions
    newpositions[0] = xpos*cosdelta - ypos*sindelta
    newpositions[1] = xpos*sindelta + ypos*cosdelta
    newerrors = errors
    xerr, yerr = errors[0:2]
    newerrors[0] = np.sqrt((xerr*cosdelta)**2 + (yerr*sindelta)**2)
    newerrors[1] = np.sqrt((yerr*cosdelta)**2 + (xerr*sindelta)**2)
    return newpositions, newerrors


def _plot_position_corrections(parm_dict, data_dict):
    """
    Plot the position corrections on top of an array configuration plot
    Args:
        parm_dict: Calling function parameter dictionary
        data_dict: The MDS contents

    Returns:
    PNG file(s) with the correction plots
    """
    telescope = Telescope(data_dict._meta_data['telescope_name'])
    destination = parm_dict['destination']
    ref_ant = data_dict._meta_data['reference_antenna']
    combined = parm_dict['combined']

    ant_list = _param_to_list(parm_dict['ant'], data_dict, 'ant')
    if combined:
        filename = f'{destination}/position_corrections_combined_{data_dict._meta_data["combine_ddis"]}.png'
        attribute_list = []
        for ant in ant_list:
            attribute_list.append(data_dict[ant].attrs)
        _plot_corrections_sub(attribute_list, filename, telescope, ref_ant, parm_dict)

    else:
        ddi_list = []
        if parm_dict['ddi'] == 'all':
            for ant in ant_list:
                ddi_list.extend(data_dict[ant].keys())
            ddi_list = np.unique(ddi_list)
        else:
            ddi_list = parm_dict['ddi']
            for i_ddi in range(len(ddi_list)):
                ddi_list[i_ddi] = 'ddi_'+ddi_list[i_ddi]
        for ddi in ddi_list:
            filename = f'{destination}/position_corrections_separated_{ddi}.png'
            attribute_list = []
            for ant in ant_list:
                if ddi in data_dict[ant].keys():
                    attribute_list.append(data_dict[ant][ddi].attrs)
            _plot_corrections_sub(attribute_list, filename, telescope, ref_ant, parm_dict)


def _plot_corrections_sub(attributes_list, filename, telescope, ref_ant, parm_dict):
    """
    Does the actual individual position correction plots
    Args:
        attributes_list: List of XDS attributes
        filename: Name of the PNG file to be created
        telescope: Telescope object used in observations
        ref_ant: Reference antenna in the data set
        parm_dict: Parameter dictionary of the caller's caller

    Returns:
    PNG file with the position corrections plot
    """
    tel_lon, tel_lat, tel_rad = _get_telescope_lat_lon_rad(telescope)
    length_unit = parm_dict['unit']
    scaling = parm_dict['scaling']
    len_fac = _convert_unit('m', length_unit, 'length')
    corr_fac = clight * scaling
    figure_size = parm_dict['figure_size']
    box_size = parm_dict['box_size']
    dpi = parm_dict['dpi']
    display = parm_dict['display']

    xlabel = f'East [{length_unit}]'
    ylabel = f'North [{length_unit}]'

    fig, axes = _create_figure_and_axes(figure_size, [2, 2], default_figsize=[8, 8])
    xy_whole = axes[0, 0]
    xy_inner = axes[0, 1]
    z_whole = axes[1, 0]
    z_inner = axes[1, 1]

    for attributes in attributes_list:
        antenna = attributes['antenna_info']
        ew_off, ns_off, _, _ = _compute_antenna_relative_off(antenna, tel_lon, tel_lat, tel_rad, len_fac)
        corrections, _ = _rotate_to_gmt(np.copy(attributes['position_fit']), attributes['position_error'],
                                        antenna['longitude'])
        corrections = np.array(corrections)*corr_fac
        text = '  '+antenna['name']
        if antenna['name'] == ref_ant:
            text += '*'
        _plot_antenna_position(xy_whole, xy_inner, ew_off, ns_off, text, box_size, marker='+')
        _plot_corrections(xy_whole, xy_inner, ew_off, ns_off, corrections[0], corrections[1], box_size)
        _plot_antenna_position(z_whole, z_inner, ew_off, ns_off, text, box_size, marker='+')
        _plot_corrections(z_whole, z_inner, ew_off, ns_off, 0, corrections[2], box_size)

    _plot_boxes_limits_and_labels(xy_whole, xy_inner, xlabel, ylabel, box_size, 'X & Y, outer array',
                                  'X & Y, inner array')
    _plot_boxes_limits_and_labels(z_whole, z_inner, xlabel, ylabel, box_size, 'Z, outer array',
                                  'Z, inner array')
    _close_figure(fig, 'Position corrections', filename, dpi, display)

