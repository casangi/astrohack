from prettytable import PrettyTable
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy import optimize as opt
from matplotlib import pyplot as plt

import astropy.units as units
import xarray as xr

from astrohack._utils._tools import _hadec_to_elevation, _format_value_error
from astrohack._utils._conversion import _convert_unit
from astrohack._utils._algorithms import _least_squares_fit
from astrohack._utils._constants import *
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._panel_classes.telescope import Telescope


def _locit_separated_chunk(locit_parms):
    """
    This is the chunk function for locit when treating each DDI separately
    Args:
        locit_parms: the locit parameter dictionary

    Returns:
    xds save to disk in the .zarr format
    """
    xds_data = locit_parms['xds_data']
    field_id, time, delays = _get_data_from_locit_xds(xds_data, locit_parms['polarization'])

    coordinates, delays, lst, elevation_limit = _build_filtered_arrays(field_id, time, delays, locit_parms)

    logger = _get_astrohack_logger()
    if len(delays) == 0:
        msg = f'{locit_parms["this_ant"]} {locit_parms["this_ddi"]} has no valid data, skipping'
        logger.warning(msg)
        return

    fit, variance = _fit_data(coordinates, delays, locit_parms)
    _create_output_xds(coordinates, lst, delays, fit, variance, locit_parms, xds_data.attrs['frequency'],
                       elevation_limit)
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
        this_field_id, this_time, this_delays = _get_data_from_locit_xds(xds_data, locit_parms['polarization'])
        freq_list.append(xds_data.attrs['frequency'])
        field_list.append(this_field_id)
        time_list.append(this_time)
        delay_list.append(this_delays)

    delays = np.concatenate(delay_list)
    time = np.concatenate(time_list)
    field_id = np.concatenate(field_list)

    coordinates, delays, lst, elevation_limit = _build_filtered_arrays(field_id, time, delays, locit_parms)

    logger = _get_astrohack_logger()
    if len(delays) == 0:
        msg = f'{locit_parms["this_ant"]} {locit_parms["this_ddi"]} has no valid data, skipping'
        logger.warning(msg)
        return

    fit, variance = _fit_data(coordinates, delays, locit_parms)
    _create_output_xds(coordinates, lst, delays, fit, variance, locit_parms, freq_list,
                       elevation_limit)
    return


def _get_data_from_locit_xds(xds_data, pol_selection):
    """
    Extract data from a .locit.zarr xds, converts the phase gains to delays using the xds frequency
    Args:
        xds_data: The .locit.zarr xds
        pol_selection: Which polarization is requested from the xds

    Returns:
        the field ids
        the time in mjd
        The delays in seconds

    """
    logger = _get_astrohack_logger()
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
        phases = np.concatenate([xds_data[f'P0_PHASE_GAINS'].values, xds_data[f'P1_PHASE_GAINS'].values])
        field_id = np.concatenate([xds_data[f'P0_FIELD_ID'].values, xds_data[f'P1_FIELD_ID'].values])
        time = np.concatenate([xds_data.p0_time.values, xds_data.p1_time.values])
    else:
        msg = f'Polarization {pol_selection} is not found in data'
        logger.error(msg)
        raise Exception(msg)
    return field_id, time, phases/twopi/freq  # field_id, time, delays


def _create_output_xds(coordinates, lst, delays, fit, variance, locit_parms, frequency, elevation_limit):
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
    fit_slope = locit_parms['fit_slope']
    antenna = locit_parms['ant_info'][locit_parms['this_ant']]

    output_xds = xr.Dataset()
    output_xds.attrs['polarization'] = locit_parms['polarization']
    output_xds.attrs['frequency'] = frequency
    output_xds.attrs['position_fit'] = fit[1:4]
    output_xds.attrs['position_error'] = variance[1:4]
    output_xds.attrs['fixed_delay_fit'] = fit[0]
    output_xds.attrs['fixed_delay_error'] = variance[0]
    output_xds.attrs['antenna_info'] = antenna
    output_xds.attrs['elevation_limit'] = elevation_limit

    if fit_kterm and fit_slope:
        output_xds.attrs['koff_fit'] = fit[4]
        output_xds.attrs['koff_error'] = variance[4]
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

    coords = {'time': coordinates[3, :]}
    output_xds['DELAYS'] = xr.DataArray(delays, dims=['time'])
    output_xds['HOUR_ANGLE'] = xr.DataArray(coordinates[0, :], dims=['time'])
    output_xds['DECLINATION'] = xr.DataArray(coordinates[1, :], dims=['time'])
    output_xds['ELEVATION'] = xr.DataArray(coordinates[2, :], dims=['time'])
    output_xds['LST'] = xr.DataArray(lst, dims=['time'])

    basename = locit_parms['position_name']
    outname = "/".join([basename, 'ant_'+antenna['name']])
    if not locit_parms['combine_ddis']:
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
    logger = _get_astrohack_logger()
    fit_kterm = locit_parms['fit_kterm']
    fit_slope = locit_parms['fit_slope']

    linalg = locit_parms['fit_engine'] == 'linear algebra'
    if linalg:
        fit, variance = _solve_linear_algebra(coordinates, delays, fit_kterm, fit_slope)
    else:
        if locit_parms['fit_engine'] == 'scipy':
            fit, variance = _solve_scipy_optimize_curve_fit(coordinates, delays, fit_kterm, fit_slope, verbose=True)
        else:
            msg = f'Unrecognized fitting engine: {locit_parms["fit_engine"]}'
            logger.error(msg)
            raise Exception(msg)
    return fit, variance


def _build_filtered_arrays(field_id, time, delays, locit_parms):
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

    # convert to actual hour angle
    coordinates[0, :] = lst.value - coordinates[0, :]
    coordinates[0, :] = np.where(coordinates[0, :] < 0, coordinates[0, :] + twopi, coordinates[0, :])

    # Filter data below elevation limit
    selection = coordinates[2, :] > elevation_limit
    delays = delays[selection]
    coordinates = coordinates[:, selection]
    lst = lst[selection]

    return coordinates, delays, lst, elevation_limit


def _geometrical_coeffs(coordinates):
    """Compute the position related coefficients for the fitting, also the 1 corresponding to the fixed phase"""
    ha, dec = coordinates[0:2]
    cosdec = np.cos(dec)
    xterm = np.cos(ha) * cosdec
    yterm = -np.sin(ha) * cosdec
    zterm = np.sin(dec)
    return [1.0, xterm, yterm, zterm]


def _kterm_coeff(coordinates):
    """Compute the k term coefficient from elevation"""
    elevation = coordinates[2]
    return np.cos(elevation)


def _slope_coeff(coordinates):
    """Compute the phase slope coefficient (basically the time)"""
    return coordinates[3]


def _solve_linear_algebra(coordinates, gains, fit_kterm, fit_slope):
    """Fit a phase model to the gain solutions using linear algebra, AIPS style"""
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
        coeffs = coeff_function(coordinates[:, i_sample])
        for irow in range(npar):
            for icol in range(irow + 1):
                system[irow, icol] += coeffs[irow] * coeffs[icol]
            vector[irow] += gains[i_sample] * coeffs[irow]

    for irow in range(1, npar):
        for icol in range(irow):
            system[icol, irow] = system[irow, icol]

    fit, variance, _ = _least_squares_fit(system, vector)

    return fit, variance


def _coeff_system_nokterm_noslope(coordinates):
    """build coefficient list for linear algebra fit with no k or slope terms"""
    coeffs = _geometrical_coeffs(coordinates)
    return coeffs


def _coeff_system_kterm_noslope(coordinates):
    """build coefficient list for linear algebra fit with k term and no slope term"""
    coeffs = _geometrical_coeffs(coordinates)
    coeffs.append(_kterm_coeff(coordinates))
    return coeffs


def _coeff_system_nokterm_slope(coordinates):
    """build coefficient list for linear algebra fit with slope term and no k term"""
    coeffs = _geometrical_coeffs(coordinates)
    coeffs.append(_slope_coeff(coordinates))
    return coeffs


def _coeff_system_kterm_slope(coordinates):
    """build coefficient list for linear algebra fit with slope and k terms"""
    coeffs = _geometrical_coeffs(coordinates)
    coeffs.append(_kterm_coeff(coordinates))
    coeffs.append(_slope_coeff(coordinates))
    return coeffs


def _solve_scipy_optimize_curve_fit(coordinates, gains, fit_kterm, fit_slope, verbose=False):
    """Fit a phase model to the gain solutions using scipy optimize curve_fit algorithm"""
    logger = _get_astrohack_logger()

    npar = 4 + fit_slope + fit_kterm
    if fit_kterm and fit_slope:
        func_function = _phase_model_kterm_slope
    elif fit_kterm and not fit_slope:
        func_function = _phase_model_kterm_noslope
    elif not fit_kterm and fit_slope:
        func_function = _phase_model_nokterm_slope
    else:
        func_function = _phase_model_nokterm_noslope

    # First guess is no errors in positions, no fixed delay and no delay rate
    p0 = np.zeros(npar)
    liminf = np.full(npar, -np.inf)
    limsup = np.full(npar, +np.inf)

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


def _phase_wrap(values, wrap=pi):
    """Simple angle wrapping routine to limit values to the -wrap to wrap range"""
    values = values % 2*wrap
    return np.where(values > wrap, values-2*wrap, values)


def _phase_model_nokterm_noslope(coordinates, fixed_delay, xoff, yoff, zoff):
    """Phase model for scipy fitting with no k or slope terms"""
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    return xterm + yterm + zterm + fixed_delay


def _phase_model_kterm_noslope(coordinates, fixed_delay, xoff, yoff, zoff, koff):
    """Phase model for scipy fitting with k term and no slope term"""
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    kterm = _kterm_coeff(coordinates) * koff
    return xterm + yterm + zterm + fixed_delay + kterm


def _phase_model_nokterm_slope(coordinates, fixed_delay, xoff, yoff, zoff, slope):
    """Phase model for scipy fitting with slope term and no k term"""
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    sterm = _slope_coeff(coordinates) * slope
    return xterm + yterm + zterm + fixed_delay + sterm


def _phase_model_kterm_slope(coordinates, fixed_delay, xoff, yoff, zoff, koff, slope):
    """"Phase model for scipy fitting with k and slope terms"""
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    sterm = _slope_coeff(coordinates) * slope
    kterm = _kterm_coeff(coordinates) * koff
    return xterm + yterm + zterm + fixed_delay + kterm + sterm


def _export_fit_results(data_dict, parm_dict):
    """Export fit results to a txt file listing the different DDIs as different solutions"""
    pos_unit = parm_dict['position_unit']
    del_unit = parm_dict['delay_unit']
    len_fact = _convert_unit('m', pos_unit, 'length')
    del_fact = _convert_unit('sec', del_unit, kind='time')
    pos_fact = len_fact * clight
    combined = data_dict._meta_data['combine_ddis']

    if combined:
        field_names = ['Antenna', f'Fixed delay  [{del_unit}]', f'X offset [{pos_unit}]',
                       f'Y offset [{pos_unit}]', f'Z offset [{pos_unit}]']
    else:
        field_names = ['Antenna', 'DDI', f'Fixed delay  [{del_unit}]', f'X offset [{pos_unit}]',
                       f'Y offset [{pos_unit}]', f'Z offset [{pos_unit}]']
    kterm_present = data_dict._meta_data["fit_kterm"]
    slope_present = data_dict._meta_data["fit_slope"]
    if kterm_present:
        field_names.extend([f'K offset [{pos_unit}]'])
    if slope_present:
        tim_unit = parm_dict['time_unit']
        slo_unit = f'{del_unit}/{tim_unit}'
        slo_fact = del_fact / _convert_unit('day', tim_unit, 'time')
        field_names.extend([f'Delay rate [{slo_unit}]'])
    else:
        slo_unit = 'N/A'
        slo_fact = 1.0

    table = PrettyTable()
    table.field_names = field_names
    table.align = 'l'
    if combined:
        for ant_key, antenna in data_dict.items():
            row = [ant_key]
            table.add_row(_export_xds(row, antenna.attrs, del_fact, pos_fact, slo_fact, kterm_present, slope_present))
    else:
        for ant_key, antenna in data_dict.items():
            for ddi_key, ddi in antenna.items():
                row = [ant_key, ddi_key]
                table.add_row(_export_xds(row, ddi.attrs, del_fact, pos_fact, slo_fact, kterm_present, slope_present))

    outname = parm_dict['destination']+'/locit_fit_results.txt'
    outfile = open(outname, 'w')
    outfile.write(table.get_string()+'\n')
    outfile.close()


def _export_xds(row, attributes, del_fact, pos_fact, slo_fact, kterm_present, slope_present):
    tolerance = 1e-4
    """Export data from the xds to the proper units as a row to be added to a pretty table"""
    row.append(_format_value_error(attributes['fixed_delay_fit'], attributes['fixed_delay_error'], del_fact,
               tolerance))
    position, poserr = _rotate_to_gmt(attributes['position_fit'], attributes['position_error'],
                                      attributes['antenna_info']['longitude'])
    for i_pos in range(3):
        row.append(_format_value_error(position[i_pos], poserr[i_pos],  pos_fact, tolerance))
    if kterm_present:
        row.append(_format_value_error(attributes['koff_fit'], attributes['koff_error'], pos_fact, tolerance))
    if slope_present:
        row.append(_format_value_error(attributes['slope_fit'], attributes['slope_error'], slo_fact, tolerance))
    return row


def _plot_sky_coverage_chunk(parm_dict):
    """Plot the sky coverage for an antenna and DDI"""
    logger = _get_astrohack_logger()
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

    if figuresize is None or figuresize == 'None':
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 2, figsize=figuresize)

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

    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.savefig(export_name, dpi=dpi)
    if not display:
        plt.close()
    return


def _plot_delays_chunk(parm_dict):
    """Plot the delays for an antenna and DDI, optionally with the fit included"""
    combined = parm_dict['combined']
    plot_fit = parm_dict['plot_fit']
    antenna = parm_dict['this_ant']
    destination = parm_dict['destination']
    if combined:
        export_name = f'{destination}/position_delays_{antenna}.png'
        suptitle = f'Delays for antenna {antenna.split("_")[1]}'
    else:
        ddi = parm_dict['this_ddi']
        export_name = f'{destination}/position_delays_{antenna}_{ddi}.png'
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

    if figuresize is None or figuresize == 'None':
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 2, figsize=figuresize)

    ylabel = f'Delays [{delay_unit}]'
    if plot_fit:
        n_samp = len(time)
        coordinates = np.ndarray([4, n_samp])
        coordinates[0, :] = ha
        coordinates[1, :] = dec
        coordinates[2, :] = ele
        coordinates[3, :] = time

        phase = xds.attrs['fixed_delay_fit']
        xoff, yoff, zoff = xds.attrs['position_fit']
        try:
            kterm = xds.attrs['koff_fit']
        except KeyError:
            kterm = None
        try:
            slope = xds.attrs['slope_fit']
        except KeyError:
            slope = None

        if slope is None and kterm is None:
            fit = _phase_model_nokterm_noslope(coordinates, phase, xoff, yoff, zoff)
        elif slope is None and kterm is not None:
            fit = _phase_model_kterm_noslope(coordinates, phase, xoff, yoff, zoff, kterm)
        elif slope is not None and kterm is None:
            fit = _phase_model_nokterm_slope(coordinates,  phase, xoff, yoff, zoff, slope)
        else:
            fit = _phase_model_kterm_slope(coordinates, phase, xoff, yoff, zoff, kterm, slope)
        fit *= delay_fact

        _scatter_plot(axes[0, 0], time, _time_label(time_unit), delays, ylabel, 'Time vs Delays', fit=fit, ylim=delaylim)
        _scatter_plot(axes[0, 1], ele, _elevation_label(angle_unit), delays, ylabel, 'Elevation vs Delays',
                      xlim=elelim, vlines=elelines, fit=fit, ylim=delaylim)
        _scatter_plot(axes[1, 0], ha, _hour_angle_label(angle_unit), delays, ylabel, 'Hour Angle vs Delays', xlim=halim,
                      fit=fit, ylim=delaylim)
        _scatter_plot(axes[1, 1], dec, _declination_label(angle_unit), delays, ylabel, 'Declination vs Delays',
                      xlim=declim, vlines=declines, fit=fit, ylim=delaylim)
    else:
        _scatter_plot(axes[0, 0], time, _time_label(time_unit), delays, ylabel, 'Time vs Delays', ylim=delaylim)
        _scatter_plot(axes[0, 1], ele, _elevation_label(angle_unit), delays, ylabel, 'Elevation vs Delays',
                      xlim=elelim, vlines=elelines, ylim=delaylim)
        _scatter_plot(axes[1, 0], ha, _hour_angle_label(angle_unit), delays, ylabel, 'Hour Angle vs Delays', xlim=halim,
                      ylim=delaylim)
        _scatter_plot(axes[1, 1], dec, _declination_label(angle_unit), delays, ylabel, 'Declination vs Delays',
                      xlim=declim, vlines=declines, ylim=delaylim)
    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.savefig(export_name, dpi=dpi)
    if not display:
        plt.close()
    return


def _time_label(unit):
    return f'Time from observation start [{unit}]'


def _elevation_label(unit):
    return f'Elevation [{unit}]'


def _declination_label(unit):
    return f'Declination [{unit}]'


def _hour_angle_label(unit):
    return f'Hour Angle [{unit}]'


def _plot_borders(angle_fact, latitude, elevation_limit):
    """Compute plot borders and and lines to be added to plots"""
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


def _scatter_plot(ax, xdata, xlabel, ydata, ylabel, title, xlim=None, ylim=None, hlines=None, vlines=None, fit=None):
    """Plot the data"""
    ax.plot(xdata, ydata, ls='', marker='+', color='red', label='data')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if hlines is not None:
        for hline in hlines:
            ax.axhline(hline, color='black', ls='--')
    if vlines is not None:
        for vline in vlines:
            ax.axvline(vline, color='black', ls='--')
    if fit is not None:
        ax.plot(xdata, fit, ls='', marker='x', color='blue', label='fit')
        ax.legend()
    return


def _rotate_to_gmt(positions, errors, longitude):
    xpos, ypos = positions[0:2]
    delta_lon = longitude
    cosdelta = np.cos(delta_lon)
    sindelta = np.sin(delta_lon)
    newpositions = positions
    #
    newpositions[0] = xpos*cosdelta - ypos*sindelta
    newpositions[1] = xpos*sindelta + ypos*cosdelta

    newerrors = errors
    xerr, yerr = errors[0:2]
    newerrors[0] = np.sqrt((xerr*cosdelta)**2 + (yerr*sindelta)**2)
    newerrors[1] = np.sqrt((yerr*cosdelta)**2 + (xerr*sindelta)**2)
    return newpositions, newerrors


def _open_telescope(telname):
    """Open correct telescope based on the telescope string"""
    if 'VLA' in telname:
        telname = 'VLA'
    elif 'ALMA' in telname:
        telname = 'ALMA_DA'  # It does not matter which ALMA layout since the array center is the same
    telescope = Telescope(telname)
    return telescope
