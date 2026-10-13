import pathlib
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, CIRS, AltAz
from astropy.time import Time
import astropy.units as u
from scipy import optimize as opt

import toolviper.utils.logger as logger
import xarray as xr

from astrohack.io.position_mds import AstrohackPositionFile
from astrohack.utils.text import (
    get_data_name,
    create_dataset_label,
    param_to_list,
)

from astrohack.utils.conversion import convert_unit
from astrohack.utils.algorithms import least_squares, phase_wrapping
from astrohack.utils.constants import twopi


def locit_separated_chunk(locit_parms: dict, output_mds: AstrohackPositionFile):
    """
    This is the chunk function for locit when treating each DDI separately
    Args:
        locit_parms: the locit parameter dictionary
        output_mds: Output mds file onto which to add results

    Returns:
    xds save to disk in the .zarr format
    """
    input_xdt = locit_parms["xdt_data"]
    field_id, time, delays, scans, freq = _get_data_from_locit_xds(
        input_xdt, locit_parms["polarization"]
    )
    _locit_common_flow(
        field_id,
        time,
        delays,
        scans,
        freq,
        locit_parms,
        input_xdt.parent.attrs["antenna_info"],
        input_xdt.parent.parent.attrs["source_dict"],
        locit_parms["this_ant"],
        locit_parms["this_ddi"],
        output_mds,
    )


def locit_combined_chunk(locit_parms: dict, output_mds: AstrohackPositionFile):
    """
    This is the chunk function for locit when we are combining the DDIs for an antenna for a single solution
    Args:
        locit_parms: the locit parameter dictionary
        output_mds: Output mds file onto which to add results

    Returns:
    xds save to disk in the .zarr format
    """
    ant_xdt = locit_parms["xdt_data"]

    delay_list = []
    time_list = []
    field_list = []
    freq_list = []
    scan_list = []

    for ddi, xdt_data in ant_xdt.items():
        this_field_id, this_time, this_delays, this_scans, freq = (
            _get_data_from_locit_xds(xdt_data, locit_parms["polarization"])
        )
        freq_list.append(freq)
        field_list.append(this_field_id)
        time_list.append(this_time)
        delay_list.append(this_delays)
        scan_list.append(this_scans)

    delays = np.concatenate(delay_list)
    time = np.concatenate(time_list)
    field_id = np.concatenate(field_list)
    scans = np.concatenate(scan_list)

    _locit_common_flow(
        field_id,
        time,
        delays,
        scans,
        freq_list,
        locit_parms,
        ant_xdt.attrs["antenna_info"],
        ant_xdt.parent.attrs["source_dict"],
        locit_parms["this_ant"],
        None,
        output_mds,
    )


def locit_difference_chunk(locit_parms: dict, output_mds: AstrohackPositionFile):
    """
    This is the chunk function for locit when we are combining two DDIs for an antenna for a single solution by using
    the difference in phase between the two DDIs of different frequencies
    Args:
        locit_parms: the locit parameter dictionary
        output_mds: Output mds file onto which to add results

    """
    ant_xdt = locit_parms["xdt_data"]
    ddi_list = param_to_list(locit_parms["ddi"], ant_xdt, "ddi")
    nddis = len(ddi_list)

    if nddis != 2:
        msg = f"The difference method support only 2 DDIs, {nddis} DDIs provided for Antenna {locit_parms['this_ant'].split('_')[1]}."
        logger.error(msg)
    else:
        ddi_0 = _get_data_from_locit_xds(
            ant_xdt[ddi_list[0]],
            locit_parms["polarization"],
            get_phases=True,
            split_pols=True,
        )
        ddi_1 = _get_data_from_locit_xds(
            ant_xdt[ddi_list[1]],
            locit_parms["polarization"],
            get_phases=True,
            split_pols=True,
        )

        time, field_id, delays, freq, scans = _delays_from_phase_differences(
            ddi_0, ddi_1
        )
        _locit_common_flow(
            field_id,
            time,
            delays,
            scans,
            freq,
            locit_parms,
            ant_xdt.attrs["antenna_info"],
            ant_xdt.parent.attrs["source_dict"],
            locit_parms["this_ant"],
            None,
            output_mds,
        )


def _locit_common_flow(
    field_id,
    time,
    delays,
    scans,
    freq_info,
    locit_parms,
    antenna_info,
    source_dict,
    ant_key,
    ddi_key,
    output_mds,
):
    """
    Function to oversee the common flow between the different modes of locit
    Args:
        field_id: Field IDs
        time: Time axis
        delays: delays
        scans: Scan IDs
        freq_info: Frquency information
        locit_parms: locit parameters
        antenna_info: Antenna information
        source_dict: Source dictionary
        ant_key: antenna key
        ddi_key: ddi key, None should be used for cases without ddi
        output_mds: Output position mds object

    Returns:
        Saves locit execution to disk.
    """
    if _has_valid_data(field_id, time, delays, locit_parms["this_ant"]):
        coordinates, delays, scans, lst, elevation_limit, nin = _build_filtered_arrays(
            field_id, time, delays, scans, locit_parms, antenna_info, source_dict
        )
        if _selection_leaves_data(nin, locit_parms["this_ant"]):
            fit, variance, converged = _fit_data(coordinates, delays, locit_parms)
            if converged:
                model, chi_squared = _compute_chi_squared(
                    delays,
                    fit,
                    coordinates,
                    locit_parms["fit_kterm"],
                    locit_parms["fit_delay_rate"],
                )
                out_xds = _create_output_xds(
                    coordinates,
                    lst,
                    delays,
                    scans,
                    fit,
                    variance,
                    chi_squared,
                    model,
                    locit_parms,
                    freq_info,
                    elevation_limit,
                    antenna_info,
                )
                if ddi_key is not None:
                    if not pathlib.Path(
                        f"{output_mds.filename}/{ant_key}/.zattrs"
                    ).exists():
                        ant_xdt = xr.DataTree(name=f"{ant_key}")
                        ant_xdt.attrs["antenna_info"] = antenna_info
                        ant_xdt = ant_xdt.assign(
                            {
                                f"{ddi_key}": xr.DataTree(
                                    dataset=out_xds, name=f"{ddi_key}"
                                )
                            }
                        )
                        output_mds.add_node(ant_xdt, [ant_key])
                    else:
                        output_mds.add_node(out_xds, [ant_key, ddi_key])
                else:
                    output_mds.add_node(out_xds, [ant_key])


def _delays_from_phase_differences(ddi_0, ddi_1):
    """
    Compute delays from the difference in phase between two DDIs of different frequencies
    Args:
        ddi_0: First DDI
        ddi_1: Second DDI

    Returns:
    Matched times, matched field ids, matched phase difference delays, matched scans, difference in frequency
    """

    freq = ddi_0[4] - ddi_1[4]
    if freq > 0:
        pos_time, pos_phase = ddi_0[1:3]
        neg_time, neg_phase = ddi_1[1:3]
        fields = ddi_0[0]
        scans = ddi_0[3]
    elif freq < 0:
        pos_time, pos_phase = ddi_1[1:3]
        neg_time, neg_phase = ddi_0[1:3]
        freq *= -1
        fields = ddi_1[0]
        scans = ddi_1[3]
    else:
        msg = "The two DDIs must have different frequencies"
        logger.error(msg)
        raise ValueError(msg)

    if isinstance(fields, list):
        time = []
        field_id = []
        phase = []
        out_scans = []
        for i_pol in range(len(fields)):
            this_time, this_field_id, this_phase, this_scan = (
                _match_times_and_phase_difference(
                    pos_time[i_pol],
                    neg_time[i_pol],
                    pos_phase[i_pol],
                    neg_phase[i_pol],
                    fields[i_pol],
                    scans[i_pol],
                )
            )
            time.append(this_time)
            field_id.append(this_field_id)
            phase.append(this_phase)
            out_scans.append(this_scan)

        time = np.concatenate(time)
        field_id = np.concatenate(field_id)
        phase = np.concatenate(phase)
        out_scans = np.concatenate(out_scans)

    else:
        time, field_id, phase, out_scans = _match_times_and_phase_difference(
            pos_time, neg_time, pos_phase, neg_phase, fields, scans
        )

    delays = phase / twopi / freq
    return time, field_id, delays, freq, out_scans


def _match_times_and_phase_difference(
    pos_time, neg_time, pos_phase, neg_phase, fields, scans, tolerance=1e-8
):
    """
    match times and compute the phase differences for the simple case, calls _different_times for the complicated case
    Args:
        pos_time: Time for the positive phase
        neg_time: Time for the negative phase
        pos_phase: Positive phase
        neg_phase: Negative phase
        fields: Field ids
        scans: Scan ids
        tolerance: Tolerance in time to match time arrays

    Returns:
    Matched times, matched field ids, -pi, pi wrapped matched phase difference, matched scans
    """
    n_pos_time, n_neg_time = len(pos_time), len(neg_time)
    if n_pos_time == n_neg_time:
        if np.all(
            np.isclose(pos_time, neg_time, tolerance)
        ):  # this the simplest case times are already matched!
            return pos_time, fields, phase_wrapping(pos_phase - neg_phase), scans
        else:
            return _different_times(
                pos_time, neg_time, pos_phase, neg_phase, fields, scans, tolerance
            )
    else:
        return _different_times(
            pos_time, neg_time, pos_phase, neg_phase, fields, scans, tolerance
        )


def _different_times(
    pos_time, neg_time, pos_phase, neg_phase, fields, scans, tolerance=1e-8
):
    """
    match times and compute the phase differences for the complicated case
    Args:
        pos_time: Time for the positive phase
        neg_time: Time for the negative phase
        pos_phase: Positive phase
        neg_phase: Negative phase
        fields: Field ids
        scans: Scan ids
        tolerance: Tolerance in time to match time arrays

    Returns:
    Matched times, matched field ids, -pi, pi wrapped matched phase difference
    """
    # This solution is not optimal but numpy does not have a task for it, if it ever becomes a bottleneck we can JIT it
    out_times = np.sort(
        [time for time in pos_time if np.isclose(neg_time, time, tolerance).any()]
    )
    ntimes = out_times.shape[0]
    out_phase = np.ndarray(ntimes)
    out_field = np.ndarray(ntimes, dtype=np.int64)
    out_scans = np.ndarray(ntimes, dtype=np.int64)

    for i_time in range(ntimes):
        i_pos = np.absolute(pos_time - out_times[i_time]).argmin()
        i_neg = np.absolute(neg_time - out_times[i_time]).argmin()
        out_phase[i_time] = pos_phase[i_pos] - neg_phase[i_neg]
        out_field[i_time] = fields[i_pos]
        out_scans[i_time] = scans[i_pos]

    return out_times, out_field, phase_wrapping(out_phase), out_scans


def _has_valid_data(field_id, time, delays, antenna, ddi=None):
    """
    Determine if locit xds has valid data for locit purposes
    :param field_id: Array of field ids in time.
    :param time: Time axis.
    :param delays: Array of delays  in time
    :param antenna: Antenna key
    :param ddi: DDI key
    :return:
    """
    msg = f"Antenna {get_data_name(antenna)} "
    if ddi is not None:
        msg += f"DDI {get_data_name(ddi)} "
    msg += "has no valid data"
    if len(field_id) == 0 or len(time) == 0 or len(delays) == 0:
        logger.warning(msg)
        return False
    else:
        return True


def _selection_leaves_data(nin, antenna, ddi=None):
    """
    Determine if elevation limit and bad scans take out all the data.
    :param nin: Number of filtered points
    :param antenna: antenna key
    :param ddi: ddi key
    :return: True or False
    """
    msg = f"Antenna {get_data_name(antenna)} "
    if ddi is not None:
        msg += f"DDI {get_data_name(ddi)} "
    msg += "selection excludes all data, try decreasing the elevation limit or changing excluded scans."
    if nin > 0:
        return True
    else:
        logger.warning(msg)
        return False


def _get_data_from_locit_xds(
    xds_data, pol_selection, get_phases=False, split_pols=False
):
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
        scans for delays
        Xds frequency

    """

    pol = xds_data.attrs["polarization_scheme"]
    freq = xds_data.attrs["frequency"]

    if len(pol) != 2:
        msg = (
            f"Polarization scheme {pol} is not what is expected for antenna based gains"
        )
        logger.error(msg)
        raise ValueError(msg)
    elif pol_selection == "both":
        phases = [
            xds_data["P0_PHASE_GAINS"].values,
            xds_data["P1_PHASE_GAINS"].values,
        ]
        field_id = [xds_data["P0_FIELD_ID"].values, xds_data["P1_FIELD_ID"].values]
        time = [xds_data.p0_time.values, xds_data.p1_time.values]
        scans = [xds_data["P0_SCANS"].values, xds_data["P1_SCANS"].values]
        if not split_pols:
            phases = np.concatenate(phases)
            field_id = np.concatenate(field_id)
            time = np.concatenate(time)
            scans = np.concatenate(scans)
    else:
        sel_pol_list = [*pol_selection]
        phases = []
        time = []
        field_id = []
        scans = []
        for pol_item in sel_pol_list:
            if pol_item in pol:
                i_pol = np.where(np.array(pol) == pol_item)[0][0]
                phases.append(xds_data[f"P{i_pol}_PHASE_GAINS"].values)
                time.append(xds_data[f"p{i_pol}_time"].values)
                field_id.append(xds_data[f"P{i_pol}_FIELD_ID"].values)
                scans.append(xds_data[f"P{i_pol}_SCANS"].values)
            else:
                msg = f"Polarization {pol_selection} is not found in data"
                logger.warning(msg)

        if len(phases) == 0:
            msg = f"No valid data found for polarization selection {pol_selection}"
            logger.error(msg)
            raise RuntimeError(msg)

        if not split_pols:
            phases = np.concatenate(phases)
            field_id = np.concatenate(field_id)
            time = np.concatenate(time)
            scans = np.concatenate(scans)

    if get_phases:
        return field_id, time, phases, scans, freq  # field_id, time, phases, frequency
    else:
        return (
            field_id,
            time,
            phases / twopi / freq,
            scans,
            freq,
        )  # field_id, time, delays, frequency


def _create_output_xds(
    coordinates,
    lst,
    delays,
    scans,
    fit,
    variance,
    chi_squared,
    model,
    locit_parms,
    frequency,
    elevation_limit,
    antenna_info,
):
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
    The xdt to be plugged to root.
    """
    fit_kterm = locit_parms["fit_kterm"]
    fit_rate = locit_parms["fit_delay_rate"]
    error = np.sqrt(variance)

    output_xds = xr.Dataset()
    output_xds.attrs["polarization"] = locit_parms["polarization"]
    output_xds.attrs["frequency"] = frequency
    output_xds.attrs["position_fit"] = fit[1:4]
    output_xds.attrs["position_error"] = error[1:4]
    output_xds.attrs["fixed_delay_fit"] = fit[0]
    output_xds.attrs["fixed_delay_error"] = error[0]
    output_xds.attrs["antenna_info"] = antenna_info
    output_xds.attrs["elevation_limit"] = elevation_limit
    output_xds.attrs["chi_squared"] = chi_squared

    if fit_kterm and fit_rate:
        output_xds.attrs["koff_fit"] = fit[4]
        output_xds.attrs["koff_error"] = error[4]
        output_xds.attrs["rate_fit"] = fit[5]
        output_xds.attrs["rate_error"] = error[5]
    elif fit_kterm and not fit_rate:
        output_xds.attrs["koff_fit"] = fit[4]
        output_xds.attrs["koff_error"] = error[4]
    elif not fit_kterm and fit_rate:
        output_xds.attrs["rate_fit"] = fit[4]
        output_xds.attrs["rate_error"] = error[4]
    else:
        pass  # Nothing to be added to the attributes

    coords = {"time": coordinates[3, :]}
    output_xds["DELAYS"] = xr.DataArray(delays, dims=["time"])
    output_xds["SCANS"] = xr.DataArray(scans, dims=["time"])
    output_xds["MODEL"] = xr.DataArray(model, dims=["time"])
    output_xds["HOUR_ANGLE"] = xr.DataArray(coordinates[0, :], dims=["time"])
    output_xds["DECLINATION"] = xr.DataArray(coordinates[1, :], dims=["time"])
    output_xds["ELEVATION"] = xr.DataArray(coordinates[2, :], dims=["time"])
    output_xds["LST"] = xr.DataArray(lst, dims=["time"])

    return output_xds.assign_coords(coords)


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
    try:
        ddi_id = locit_parms["this_ddi"]
    except KeyError:
        ddi_id = None
    label = create_dataset_label(locit_parms["this_ant"], ddi_id)

    fit_kterm = locit_parms["fit_kterm"]
    fit_rate = locit_parms["fit_delay_rate"]
    fit_engine = locit_parms["fit_engine"]

    if fit_engine == "linear algebra":
        try:
            fit, variance = _solve_linear_algebra(
                coordinates, delays, fit_kterm, fit_rate
            )
            return fit, variance, True
        except np.linalg.LinAlgError:
            logger.warning(
                f"Fitting failed for {label}, please try another fitting engine or DDI combination"
            )
            return np.nan, np.nan, False

    elif fit_engine == "scipy":
        try:
            fit, variance = _solve_scipy_optimize_curve_fit(
                coordinates, delays, fit_kterm, fit_rate, verbose=True
            )
            return fit, variance, True
        except TypeError:
            logger.warning(
                f"Fitting failed for {label}, please try another fitting engine or DDI combination"
            )
            return np.nan, np.nan, False

    else:
        msg = f'Unrecognized fitting engine: {locit_parms["fit_engine"]}'
        logger.error(msg)
        return np.nan, np.nan, False


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
    chi_squared = np.sum((model - delays) ** 2 / n_delays)
    return model, chi_squared


def _build_coordinate_array(ant_info, delay_time, field_ids, source_dict):
    """
    Build coordinate array using only astropy functions.
    Args:
        ant_info: Dictionary with antenna information
        delay_time: Time of delay samples in mjd
        field_ids: Field IDs for delay measurements
        source_dict: Dictionary with source information

    Returns:
        coordinate array of shape [4, n_samples]  and lst, first dimension of coordinate array is:\
         hour angle, declination elevation and time since first sample.
    """
    geo_pos = ant_info["geocentric_position"]
    ant_location = EarthLocation.from_geocentric(
        geo_pos[0],
        geo_pos[1],
        geo_pos[2],
        "meter",
    )
    n_times = len(field_ids)
    j2000_radec = np.zeros((n_times, 2))
    keys_are_str = isinstance(list(source_dict.keys())[0], str)
    for row, field_id in enumerate(field_ids):
        if keys_are_str:
            src_key = str(field_id)
        else:
            src_key = field_id
        j2000_radec[row, :] = source_dict[src_key]["fk5"]
    astropy_times = Time(delay_time, format="mjd", scale="utc", location=ant_location)
    skycoords = SkyCoord(
        ra=j2000_radec[:, 0] * u.rad, dec=j2000_radec[:, 1] * u.rad, frame="icrs"
    ).transform_to(CIRS(obstime=astropy_times))
    lst = astropy_times.sidereal_time("apparent").to(u.rad) / u.rad
    ra = skycoords.ra.rad
    hour_angle = lst - ra
    altaz_frame = AltAz(location=ant_location, obstime=astropy_times)
    altaz_coords = skycoords.transform_to(altaz_frame)

    coordinates = np.zeros((4, n_times))
    coordinates[0, :] = np.where(
        hour_angle.value > 0, hour_angle.value, hour_angle.value + twopi
    )
    coordinates[1, :] = skycoords.dec.rad
    coordinates[2, :] = altaz_coords.alt.rad
    coordinates[3, :] = delay_time - delay_time[0]
    return coordinates, lst


def _filter_data_on_elevation_and_scans(
    elevation_limit_deg, scans_to_exclude, coordinates, lst, scans, delays
):
    """
    Filter delays based on elevation and scans to exclude
    Args:
        elevation_limit_deg: Elevation limit in degrees
        scans_to_exclude: List of scans to exclude
        coordinates: Coordinates (ha, dec, el, time)
        lst: local sidereal time array
        scans: scan array
        delays: delay array

    Returns:
        elevation limit in radians, filtered scans, filtered delays, filtered coordinates, filtered lst
    """
    elevation_limit_rad = elevation_limit_deg * convert_unit(
        "deg", "rad", "trigonometric"
    )
    el_selection = coordinates[2, :] > elevation_limit_rad
    scan_selection = np.full_like(el_selection, True)
    for bad_scan in scans_to_exclude:
        scan_selection = np.logical_and(scan_selection, scans != bad_scan)
    final_selection = np.logical_and(el_selection, scan_selection)

    delays = delays[final_selection]
    coordinates = coordinates[:, final_selection]
    lst = lst[final_selection]
    scans = scans[final_selection]
    return elevation_limit_rad, scans, delays, coordinates, lst


def _build_filtered_arrays(
    field_id, time, delays, scans, locit_parms, antenna_info, source_dict
):
    """Build the coordinate arrays (ha, dec, elevation, time) for use in the fitting and filters data below the \
    elevation limit

    Args:
        field_id: Array with the observed field per delay
        time: Time array with the time of each delay
        delays: The delay array
        scans: The scan array
        locit_parms: Locit main function parameters
        antenna_info: Antenna info
        source_dict: Dictionary with the source name and coordinates

    Returns:
    coordinates (ha, dec, ele, time), delays, scans, local sidereal time all filtered by elevation limit and the \
    elevation_limit
    """

    coordinates, lst = _build_coordinate_array(
        antenna_info, time, field_id, source_dict
    )
    elevation_limit_rad, scans, delays, coordinates, lst = (
        _filter_data_on_elevation_and_scans(
            locit_parms["elevation_limit"],
            locit_parms["exclude_scans"],
            coordinates,
            lst,
            scans,
            delays,
        )
    )
    return coordinates, delays, scans, lst, elevation_limit_rad, lst.size


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
    """Compute the k term (offset from antenna elevation axis) coefficient from elevation

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

    fit, variance, _ = least_squares(system, vector)

    return fit, variance


def _system_coefficients(coordinates, fit_kterm, fit_rate):
    """Build coefficient list for linear algebra fit

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


def _solve_scipy_optimize_curve_fit(
    coordinates, delays, fit_kterm, fit_rate, verbose=False
):
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
    covar = None
    fit = None
    for maxfev in maxfevs:
        try:
            results = opt.curve_fit(
                fit_function,
                coordinates,
                delays,
                p0=p0,
                bounds=[liminf, limsup],
                maxfev=maxfev,
            )
            fit, covar = results[0:2]
        except RuntimeError:
            if verbose:
                logger.info("Increasing number of iterations")
                continue
            else:
                if verbose:
                    logger.info(
                        "Converged with less than {0:d} iterations".format(maxfev)
                    )
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
