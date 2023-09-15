from prettytable import PrettyTable
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy import optimize as opt

import astropy.units as units
import xarray as xr

from astrohack._utils._locit_commons import _open_telescope, _get_telescope_lat_lon_rad, _compute_antenna_relative_off
from astrohack._utils._locit_commons import  _scatter_plot, _time_label, _elevation_label, _declination_label
from astrohack._utils._locit_commons import _create_figure_and_axes, _plot_antenna_position, _close_figure
from astrohack._utils._locit_commons import _plot_boxes_limits_and_labels, _plot_corrections, _hour_angle_label
from astrohack._utils._tools import _hadec_to_elevation, _format_value_error
from astrohack._utils._conversion import _convert_unit
from astrohack._utils._algorithms import _least_squares_fit
from astrohack._utils._constants import *
from astrohack.client import _get_astrohack_logger


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
    model, chi_squared = _compute_chi_squared(delays, fit, coordinates, locit_parms['fit_kterm'],
                                              locit_parms['fit_slope'])
    _create_output_xds(coordinates, lst, delays, fit, variance, chi_squared, model, locit_parms,
                       xds_data.attrs['frequency'], elevation_limit)
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
    model, chi_squared = _compute_chi_squared(delays, fit, coordinates, locit_parms['fit_kterm'],
                                              locit_parms['fit_slope'])
    _create_output_xds(coordinates, lst, delays, fit, variance, chi_squared, model, locit_parms, freq_list,
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
    output_xds.attrs['chi_squared'] = chi_squared

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
    output_xds['MODEL'] = xr.DataArray(model, dims=['time'])
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


def _compute_chi_squared(delays, fit, coordinates, fit_kterm, fit_slope):
    model_function, _ = _define_fit_function(fit_kterm, fit_slope)
    model = model_function(coordinates, *fit)
    n_delays = len(delays)
    chi_squared = np.sum((model-delays)**2/n_delays)
    return model, chi_squared


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


def _define_fit_function(fit_kterm, fit_slope):
    npar = 4 + fit_slope + fit_kterm
    if fit_kterm and fit_slope:
        fit_function = _phase_model_kterm_slope
    elif fit_kterm and not fit_slope:
        fit_function = _phase_model_kterm_noslope
    elif not fit_kterm and fit_slope:
        fit_function = _phase_model_nokterm_slope
    else:
        fit_function = _phase_model_nokterm_noslope
    return fit_function, npar


def _solve_scipy_optimize_curve_fit(coordinates, gains, fit_kterm, fit_slope, verbose=False):
    """Fit a phase model to the gain solutions using scipy optimize curve_fit algorithm"""
    logger = _get_astrohack_logger()

    fit_function, npar = _define_fit_function(fit_kterm, fit_slope)

    # First guess is no errors in positions, no fixed delay and no delay rate
    p0 = np.zeros(npar)
    liminf = np.full(npar, -np.inf)
    limsup = np.full(npar, +np.inf)

    maxfevs = [100000, 1000000, 10000000]
    for maxfev in maxfevs:
        try:
            fit, covar = opt.curve_fit(fit_function, coordinates, gains, p0=p0, bounds=[liminf, limsup], maxfev=maxfev)
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
    include_missing = parm_dict['include_missing']

    if combined:
        field_names = ['Antenna', f'RMS [{del_unit}]', f'F. delay [{del_unit}]', f'X offset [{pos_unit}]',
                       f'Y offset [{pos_unit}]', f'Z offset [{pos_unit}]']
        nfields = 5
    else:
        field_names = ['Antenna', 'DDI', f'RMS [{del_unit}]', f'F. delay [{del_unit}]', f'X offset [{pos_unit}]',
                       f'Y offset [{pos_unit}]', f'Z offset [{pos_unit}]']
        nfields = 6
    kterm_present = data_dict._meta_data["fit_kterm"]
    slope_present = data_dict._meta_data["fit_slope"]
    if kterm_present:
        field_names.extend([f'K offset [{pos_unit}]'])
        nfields += 1
    if slope_present:
        tim_unit = parm_dict['time_unit']
        slo_unit = f'{del_unit}/{tim_unit}'
        slo_fact = del_fact / _convert_unit('day', tim_unit, 'time')
        field_names.extend([f'Rate [{slo_unit}]'])
        nfields += 1
    else:
        slo_unit = notavail
        slo_fact = 1.0

    table = PrettyTable()
    table.field_names = field_names
    table.align = 'c'
    antenna_list = _open_telescope(data_dict._meta_data['telescope_name']).ant_list

    for ant_name in antenna_list:
        ant_key = 'ant_'+ant_name
        if ant_name == data_dict._meta_data['reference_antenna']:
            ant_name += ' (ref)'
        row = [ant_name]
        if ant_key in data_dict.keys():
            antenna = data_dict[ant_key]
            if combined:
                table.add_row(_export_xds(row, antenna.attrs, del_fact, pos_fact, slo_fact, kterm_present,
                                          slope_present))
            else:
                for ddi_key, ddi in antenna.items():
                    row = [ant_name, ddi_key.split('_')[1]]
                    table.add_row(_export_xds(row, ddi.attrs, del_fact, pos_fact, slo_fact, kterm_present,
                                              slope_present))
        else:
            if include_missing:
                for ifield in range(nfields):
                    row.append(notavail)
                table.add_row(row)

    outname = parm_dict['destination']+'/locit_fit_results.txt'
    outfile = open(outname, 'w')
    outfile.write(table.get_string()+'\n')
    outfile.close()


def _export_xds(row, attributes, del_fact, pos_fact, slo_fact, kterm_present, slope_present):
    tolerance = 1e-4
    """Export data from the xds to the proper units as a row to be added to a pretty table"""

    rms = np.sqrt(attributes["chi_squared"])*del_fact
    row.append(f'{rms:.2e}')
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
    """Plot the delays for an antenna and DDI, optionally with the fit included"""
    combined = parm_dict['combined']
    plot_model = parm_dict['plot_model']
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


def _rotate_to_gmt(positions, errors, longitude):
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
    telescope = _open_telescope(data_dict._meta_data['telescope_name'])
    tel_lon, tel_lat, tel_rad = _get_telescope_lat_lon_rad(telescope)
    length_unit = parm_dict['unit']
    scaling = parm_dict['scaling']
    len_fac = _convert_unit('m', length_unit, 'length')
    corr_fac = clight * scaling / len_fac
    figure_size = parm_dict['figure_size']
    box_size = parm_dict['box_size']
    dpi = parm_dict['dpi']
    display = parm_dict['display']
    destination = parm_dict['destination']
    filename = f'{destination}/position_corrections.png'

    fig, axes = _create_figure_and_axes(figure_size, [2, 2])
    xy_whole = axes[0, 0]
    xy_inner = axes[0, 1]
    z_whole = axes[1, 0]
    z_inner = axes[1, 1]
    ref_ant = data_dict._meta_data['reference_antenna']

    combined = data_dict._meta_data['combine_ddis']
    if combined:
        for xds in data_dict.values():
            attributes = xds.attrs
            antenna = attributes['antenna_info']
            ew_off, ns_off, _, _ = _compute_antenna_relative_off(antenna, tel_lon, tel_lat, tel_rad, len_fac)
            corrections, _ = _rotate_to_gmt(attributes['position_fit'], attributes['position_error'],
                                            antenna['longitude'])
            corrections = np.array(corrections)*corr_fac
            text = ' '+antenna['name']
            if antenna['name'] == ref_ant:
                text += '*'
            _plot_antenna_position(xy_whole, xy_inner, ew_off, ns_off, text, box_size, marker='.')
            _plot_corrections(xy_whole, xy_inner, ew_off, ns_off, corrections[0], corrections[1], box_size)
            _plot_antenna_position(z_whole, z_inner, ew_off, ns_off, text, box_size, marker='.')
            _plot_corrections(z_whole, z_inner, ew_off, ns_off, 0, corrections[2], box_size)
    else:
        raise Exception('multiple DDIs not yet supported')
    xlabel = f'East [{length_unit}]'
    ylabel = f'North [{length_unit}]'
    _plot_boxes_limits_and_labels(xy_whole, xy_inner, xlabel, ylabel, box_size, 'X & Y, outer array',
                                  'X & Y, inner array')
    _plot_boxes_limits_and_labels(z_whole, z_inner, xlabel, ylabel, box_size, 'Z, outer array',
                                  'Z, inner array')

    _close_figure(fig, 'Position corrections', filename, dpi, display)



