import numpy as np
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


def _locit_chunk(locit_parms):
    """
    This is the chunk function for locit
    Args:
        locit_parms: the locit parameter dictionary

    Returns:
    xds save to disk in the .zarr format
    """
    logger = _get_astrohack_logger()
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

    if len(gains) == 0:
        msg = f'{locit_parms["this_ant"]} {locit_parms["this_ddi"]} has no valid data, skipping'
        logger.warning(msg)
        return
    astro_time = Time(time, format='mjd', scale='utc', location=ant_pos)
    lst = astro_time.sidereal_time("apparent").to(units.radian) / units.radian
    coordinates = _build_coordinate_array(field_id, src_dict, 'precessed', antenna['latitude'], time, lst)

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

    output_xds = xr.Dataset()
    output_xds.attrs['polarization'] = locit_parms['polarization']
    output_xds.attrs['wavelength'] = clight/xds_data.attrs['frequency']
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


def _build_coordinate_array(field_id, src_list, key, latitude, time, lst):
    """ Build the coordinate arrays (ha, dec, elevation, angle) for use in the fitting"""
    n_samples = len(field_id)
    coordinates = np.ndarray([4, n_samples])
    for i_sample in range(n_samples):
        field = str(field_id[i_sample])
        coordinates[0:2, i_sample] = src_list[field][key]
        coordinates[2, i_sample] = _hadec_to_elevation(src_list[field][key], latitude)
        coordinates[3, i_sample] = time[i_sample]-time[0]  # time is set to zero at the beginning of obs

    # convert to actual hour angle
    coordinates[0, :] = lst.value - coordinates[0, :]
    coordinates[0, :] = np.where(coordinates[0, :] < 0, coordinates[0, :] + twopi, coordinates[0, :])
    return coordinates


def _geometrical_coeffs(coordinates):
    """Compute the position related coefficients for the fitting, also the 1 corresponding to the delay"""
    ha, dec = coordinates[0:2]
    cosdec = np.cos(dec)
    xterm = twopi*np.cos(ha) * cosdec
    yterm = -twopi*np.sin(ha) * cosdec
    zterm = twopi*np.sin(dec)
    return [1.0, xterm, yterm, zterm]


def _kterm_coeff(coordinates):
    """Compute the k term coefficient from elevation"""
    elevation = coordinates[2]
    return twopi*np.cos(elevation)


def _slope_coeff(coordinates):
    """Compute the phase slope coefficient (basically the time)"""
    return coordinates[3]


def _solve_linear_algebra(coordinates, gains, elevation_limit, fit_kterm, fit_slope):
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


def _solve_scipy_optimize_curve_fit(coordinates, gains, elevation_limit, fit_kterm, fit_slope, verbose=False):
    """Fit a phase model to the gain solutions using scipy optimize curve_fit algorithm"""
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


def _phase_wrap(values, wrap=pi):
    """Simple angle wrapping routine to limit values to the -wrap to wrap range"""
    values = values % 2*wrap
    return np.where(values > wrap, values-2*wrap, values)


def _phase_model_nokterm_noslope(coordinates,  inst_delay, xoff, yoff, zoff):
    """Phase model for scipy fitting with no k or slope terms"""
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    return xterm + yterm + zterm + inst_delay


def _phase_model_kterm_noslope(coordinates, inst_delay, xoff, yoff, zoff, koff):
    """Phase model for scipy fitting with k term and no slope term"""
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    kterm = _kterm_coeff(coordinates) * koff
    return xterm + yterm + zterm + inst_delay + kterm


def _phase_model_nokterm_slope(coordinates, inst_delay, xoff, yoff, zoff, slope):
    """Phase model for scipy fitting with slope term and no k term"""
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    sterm = _slope_coeff(coordinates) * slope
    return xterm + yterm + zterm + inst_delay + _phase_wrap(sterm)


def _phase_model_kterm_slope(coordinates, inst_delay, xoff, yoff, zoff, koff, slope):
    """"Phase model for scipy fitting with k and slope terms"""
    coeffs = _geometrical_coeffs(coordinates)
    xterm = coeffs[1] * xoff
    yterm = coeffs[2] * yoff
    zterm = coeffs[3] * zoff
    sterm = _slope_coeff(coordinates) * slope
    kterm = _kterm_coeff(coordinates) * koff
    return xterm + yterm + zterm + inst_delay + kterm + _phase_wrap(sterm)


def _export_fit_separate_ddis(data_dict, parm_dict):
    """Export fit results to a txt file listing the different DDIs as different solutions"""
    pos_unit = parm_dict['position_unit']
    ang_unit = parm_dict['angle_unit']
    len_fact = _convert_unit('m', pos_unit, 'length')
    ang_fact = _convert_unit('rad', ang_unit, kind='trigonometric')

    field_names = ['Antenna', 'DDI', f'Fixed delay  [{ang_unit}]', f'X offset [{pos_unit}]',
                   f'Y offset [{pos_unit}]', f'Z offset [{pos_unit}]']
    kterm_present = data_dict._meta_data["fit_kterm"]
    slope_present = data_dict._meta_data["fit_slope"]
    if kterm_present:
        field_names.extend([f'K offset [{pos_unit}]'])
    if slope_present:
        tim_unit = parm_dict['time_unit']
        slo_unit = f'{ang_unit}/{tim_unit}'
        slope_fact = ang_fact / _convert_unit('day', tim_unit, 'time')
        field_names.extend([f'Phase slope [{slo_unit}]'])

    table = PrettyTable()
    table.field_names = field_names

    table.align = 'l'
    for ant_key, antenna in data_dict.items():
        for ddi_key, ddi in antenna.items():
            pos_fact = len_fact * ddi.attrs['wavelength']
            row = [ant_key, ddi_key, _format_value_error(ddi.attrs['fixed_delay_fit'], ddi.attrs['fixed_delay_error'],
                                                         scaling=ang_fact)]
            for i_pos in range(3):
                row.append(_format_value_error(ddi.attrs['position_fit'][i_pos], ddi.attrs['position_error'][i_pos],
                                               scaling=pos_fact))
            if kterm_present:
                row.append(_format_value_error(ddi.attrs['koff_fit'], ddi.attrs['koff_error'], scaling=pos_fact))
            if slope_present:
                row.append(_format_value_error(ddi.attrs['slope_fit'], ddi.attrs['slope_error'], scaling=slope_fact))
            table.add_row(row)

    outname = parm_dict['destination']+'/locit_fit_results_separated_ddis.txt'
    outfile = open(outname, 'w')
    outfile.write(table.get_string()+'\n')
    outfile.close()


def _export_fit_combine_ddis(data_dict, parm_dict):
    """Export fit results to a txt file combining the different DDIs in a single lower uncertainty solution"""
    pos_unit = parm_dict['position_unit']
    ang_unit = parm_dict['angle_unit']
    len_fact = _convert_unit('m', pos_unit, 'length')
    ang_fact = _convert_unit('rad', ang_unit, kind='trigonometric')

    field_names = ['Antenna', f'Fixed delay  [{ang_unit}]', f'X offset [{pos_unit}]',
                   f'Y offset [{pos_unit}]', f'Z offset [{pos_unit}]']
    npar = 4
    kterm_present = data_dict._meta_data["fit_kterm"]
    slope_present = data_dict._meta_data["fit_slope"]
    if kterm_present:
        field_names.extend([f'K offset [{pos_unit}]'])
        npar += 1
        i_kterm = npar-1
    if slope_present:
        tim_unit = parm_dict['time_unit']
        slo_unit = f'{ang_unit}/{tim_unit}'
        slope_fact = ang_fact / _convert_unit('day', tim_unit, 'time')
        field_names.extend([f'Phase slope [{slo_unit}]'])
        npar += 1
        i_slope = npar-1

    table = PrettyTable()
    table.field_names = field_names
    table.align = 'l'

    for ant_key, antenna in data_dict.items():
        n_ddi = len(antenna.keys())
        params = np.zeros([npar, n_ddi])
        weight = np.zeros([npar, n_ddi])
        row = [ant_key]
        i_ddi = 0
        for ddi_key, ddi in antenna.items():
            pos_fact = len_fact * ddi.attrs['wavelength']
            params[0, i_ddi] = ddi.attrs['fixed_delay_fit'] * ang_fact
            weight[0, i_ddi] = 1/(ddi.attrs['fixed_delay_error'] * ang_fact)**2
            params[1:4, i_ddi] = np.array(ddi.attrs['position_fit'])*pos_fact
            weight[1:4, i_ddi] = 1/(np.array(ddi.attrs['position_error'])*pos_fact)**2
            if kterm_present:
                params[i_kterm, i_ddi] = ddi.attrs['koff_fit'] * pos_fact
                weight[i_kterm, i_ddi] = 1 / (ddi.attrs['koff_error'] * pos_fact) ** 2
            if slope_present:
                params[i_slope, i_ddi] = ddi.attrs['slope_fit'] * slope_fact
                weight[i_slope, i_ddi] = 1 / (ddi.attrs['slope_error'] * slope_fact) ** 2
            i_ddi += 1

        avgparams = np.average(params, weights=weight, axis=1)
        avgerrors = 1/np.sqrt(np.sum(weight, axis=1))
        for i_par in range(npar):
            row.append(_format_value_error(avgparams[i_par], avgerrors[i_par], 1.0))
        table.add_row(row)

    outname = parm_dict['destination']+'/locit_fit_results_combined_ddis.txt'
    outfile = open(outname, 'w')
    outfile.write(table.get_string()+'\n')
    outfile.close()


def _plot_sky_coverage_chunk(parm_dict):
    """Plot the sky coverage for an antenna and DDI"""
    logger = _get_astrohack_logger()
    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    xds = parm_dict['xds_data']
    figuresize = parm_dict['figure_size']
    angle_unit = parm_dict['angle_unit']
    time_unit = parm_dict['time_unit']
    display = parm_dict['display']
    dpi = parm_dict['dpi']
    antenna_info = xds.attrs['antenna_info']
    export_name = f'{destination}/position_sky_coverage_{antenna}_{ddi}.png'

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
    timelabel = _time_label(time_units)
    halabel = _hour_angle_label(angle_unit)
    declabel = _declination_label(angle_unit)
    _scatter_plot(axes[0, 0], time, timelabel, ele, _elevation_label(angle_unit), 'Time vs Elevation', ylim=elelim,
                  hlines=elelines)
    _scatter_plot(axes[0, 1], time, timelabel, ha, halabel, 'Time vs Hour angle', ylim=halim)
    _scatter_plot(axes[1, 0], time, timelabel, dec, declabel, 'Time vs Declination', ylim=declim, hlines=declines)
    _scatter_plot(axes[1, 1], ha, halabel, dec, declabel, 'Hour angle vs Declination', ylim=declim, xlim=halim,
                  hlines=declines)

    fig.suptitle(f'Sky coverage for antenna {antenna.split("_")[1]}, DDI {ddi.split("_")[1]}')
    fig.tight_layout()
    plt.savefig(export_name, dpi=dpi)
    if not display:
        plt.close()
    return


def _plot_gains_chunk(parm_dict):
    """Plot the gain solutions for an antenna and DDI, optionally with the fit included"""
    plot_fit = parm_dict['plot_fit']
    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    xds = parm_dict['xds_data']
    figuresize = parm_dict['figure_size']
    angle_unit = parm_dict['angle_unit']
    time_unit = parm_dict['time_unit']
    display = parm_dict['display']
    dpi = parm_dict['dpi']
    antenna_info = xds.attrs['antenna_info']
    export_name = f'{destination}/position_gains_{antenna}_{ddi}.png'

    time = xds.time.values * _convert_unit('day', time_unit, 'time')
    angle_fact = _convert_unit('rad', angle_unit, 'trigonometric')
    ha = xds['HOUR_ANGLE'] * angle_fact
    dec = xds['DECLINATION'] * angle_fact
    ele = xds['ELEVATION'] * angle_fact
    gains = xds['GAINS'].values * angle_fact

    elelim, elelines, declim, declines, halim = _plot_borders(angle_fact, antenna_info['latitude'],
                                                              xds.attrs['elevation_limit'])
    gainslim = [-1.05*pi*angle_fact, 1.05*pi*angle_fact]

    if figuresize is None or figuresize == 'None':
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 2, figsize=figuresize)

    ylabel = f'Phase gains [{angle_unit}]'
    if plot_fit:
        n_samp = len(time)
        coordinates = np.ndarray([4, n_samp])
        coordinates[0, :] = ha
        coordinates[1, :] = dec
        coordinates[2, :] = ele
        coordinates[3, :] = time

        delay = xds.attrs['fixed_delay_fit']
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
            fit = _phase_model_nokterm_noslope(coordinates, delay, xoff, yoff, zoff)
        elif slope is None and kterm is not None:
            fit = _phase_model_kterm_noslope(coordinates, delay, xoff, yoff, zoff, kterm)
        elif slope is not None and kterm is None:
            fit = _phase_model_nokterm_slope(coordinates,  delay, xoff, yoff, zoff, slope)
        else:
            fit = _phase_model_kterm_slope(coordinates, delay, xoff, yoff, zoff, kterm, slope)
        fit *= angle_fact

        _scatter_plot(axes[0, 0], time, _time_label(time_unit), gains, ylabel, 'Time vs Gains', fit=fit, ylim=gainslim)
        _scatter_plot(axes[0, 1], ele, _elevation_label(angle_unit), gains, ylabel, 'Elevation vs Gains',
                      xlim=elelim, vlines=elelines, fit=fit, ylim=gainslim)
        _scatter_plot(axes[1, 0], ha, _hour_angle_label(angle_unit), gains, ylabel, 'Hour Angle vs Gains', xlim=halim,
                      fit=fit, ylim=gainslim)
        _scatter_plot(axes[1, 1], dec, _declination_label(angle_unit), gains, ylabel, 'Declination vs Gains',
                      xlim=declim, vlines=declines, fit=fit, ylim=gainslim)
    else:
        _scatter_plot(axes[0, 0], time, _time_label(time_unit), gains, ylabel, 'Time vs Gains', ylim=gainslim)
        _scatter_plot(axes[0, 1], ele, _elevation_label(angle_unit), gains, ylabel, 'Elevation vs Gains',
                      xlim=elelim, vlines=elelines, ylim=gainslim)
        _scatter_plot(axes[1, 0], ha, _hour_angle_label(angle_unit), gains, ylabel, 'Hour Angle vs Gains', xlim=halim,
                      ylim=gainslim)
        _scatter_plot(axes[1, 1], dec, _declination_label(angle_unit), gains, ylabel, 'Declination vs Gains',
                      xlim=declim, vlines=declines, ylim=gainslim)
    fig.suptitle(f'Gains for antenna {antenna.split("_")[1]}, DDI {ddi.split("_")[1]}')
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
