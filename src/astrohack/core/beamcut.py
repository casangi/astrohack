from copy import deepcopy

import astropy
import numpy as np
import toolviper.utils.logger as logger
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress

from astrohack.antenna.telescope import get_proper_telescope
from astrohack.io.beamcut_mds import AstrohackBeamcutFile
from astrohack.utils.constants import sig_2_fwhm
from astrohack.utils.text import (
    convert_unit,
    create_dataset_label,
    format_value_unit,
)


###########################################################
### Working Chunks
###########################################################
def process_beamcut_chunk(beamcut_chunk_params: dict, output_mds: AstrohackBeamcutFile):
    """
    Ingests a holog_xds containing beamcuts and produces a beamcut_xdtree containing the cuts separated in xdses.

    :param beamcut_chunk_params: Parameter dictionary with inputs
    :type beamcut_chunk_params: dict

    :param output_mds: Output mds file
    :type output_mds: AstrohackBeamcutFile

    :return: Beamcut_xdtree containing the different cuts for this antenna and DDI.
    :rtype: xr.DataTree
    """
    ddi_key = beamcut_chunk_params["this_ddi"]
    ant_key = beamcut_chunk_params["this_ant"]
    xdt_data = beamcut_chunk_params["xdt_data"]
    datalabel = create_dataset_label(ant_key, ddi_key)

    # This assumes that there will be no more than one mapping
    n_children = len(xdt_data.keys())
    if n_children == 0:
        logger.warning(f"No beamcut data for {datalabel}")
    else:
        logger.info(f"processing {datalabel}")

        cut_xdtree = xr.DataTree(name=f"{ant_key}-{ddi_key}")
        n_cut = 0
        for map_xds in xdt_data.values():
            cut_xdtree, n_cut = _extract_cuts_from_visibilities(
                map_xds, cut_xdtree, n_cut, ant_key, ddi_key
            )

        _beamcut_multi_lobes_gaussian_fit(cut_xdtree, datalabel)

        output_mds.add_node(cut_xdtree, [ant_key, ddi_key])


###########################################################
### Data extraction
###########################################################
def _extract_cuts_from_visibilities(map_xds, cut_xdtree, n_cut, antenna, ddi):
    """
    Creates data tree containing the different cuts from a holog xds.

    :param map_xds: holog xds containing visibilities with beam cuts.
    :type map_xds: xarray.Dataset

    :param antenna: Antenna key
    :type antenna: str

    :param ddi: DDI key
    :type ddi: str

    :return: Data tree containing the beamcut xdses.
    :rtype: xarray.DataTree
    """
    scan_time_ranges = map_xds.attrs["scan_time_ranges"]
    scan_list = map_xds.attrs["scan_list"]
    obs_summ = deepcopy(map_xds.attrs["summary"])
    obs_summ["spectral"]["channel width"] *= obs_summ["spectral"]["number of channels"]
    obs_summ["spectral"]["number of channels"] = 1
    cut_xdtree.attrs["summary"] = obs_summ

    lm_offsets = map_xds.DIRECTIONAL_COSINES.values
    time_axis = map_xds.time.values
    corr_axis = map_xds.pol.values
    visibilities = map_xds.VIS.values
    weights = map_xds.WEIGHT.values

    nchan = visibilities.shape[1]
    fchan = 4
    lchan = int(nchan - fchan)
    for iscan, scan_number in enumerate(scan_list):
        scan_time_range = scan_time_ranges[iscan]
        time_selection = np.logical_and(
            time_axis >= scan_time_range[0], time_axis < scan_time_range[1]
        )
        if np.sum(time_selection) == 0:
            logger.warning(f"Scan {scan_number} has no data for {antenna} DDI {ddi}")
            continue
        time = time_axis[time_selection]
        this_lm_offsets = lm_offsets[time_selection, :]

        lm_angle, lm_dist, direction, xlabel = (
            _cut_direction_determination_and_label_creation(this_lm_offsets)
        )
        hands_dict = _get_parallel_hand_indexes(corr_axis)

        try:
            avg_vis = np.average(
                visibilities[time_selection, fchan:lchan, :],
                axis=1,
                weights=weights[time_selection, fchan:lchan, :],
            )

        except ZeroDivisionError:
            # This should only happen when all the data for a correlation is flagged.
            # Since the flagged data is the one with zero weights we set the weights to 1 on those cases just to
            # warrant that averages will work, further down the line there are mechanisms to catch wholly invalid data.
            for i_corr in range(weights.shape[2]):
                if np.sum(weights[time_selection, fchan:lchan, i_corr]) == 0:
                    weights[time_selection, fchan:lchan, i_corr] = 1.0

            avg_vis = np.average(
                visibilities[time_selection, fchan:lchan, :],
                axis=1,
                weights=weights[time_selection, fchan:lchan, :],
            )

        avg_wei = np.average(weights[time_selection, fchan:lchan, :], axis=1)

        avg_time = np.average(time) * convert_unit("sec", "day", "time")
        timestr = astropy.time.Time(avg_time, format="mjd").to_value(
            "iso", subfmt="date_hm"
        )

        xds = xr.Dataset()
        coords = {"lm_dist": lm_dist, "time": time}

        xds.attrs.update(
            {
                "scan_number": scan_number,
                "lm_angle": lm_angle,
                "available_corrs": list(hands_dict.keys()),
                "direction": direction,
                "xlabel": xlabel,
                "time_string": timestr,
            }
        )

        xds["lm_offsets"] = xr.DataArray(this_lm_offsets, dims=["time", "lm"])
        all_corr_ymax = 1e-34
        for parallel_hand, icorr in hands_dict.items():
            amp = np.abs(avg_vis[:, icorr])
            maxamp = np.max(amp)
            if maxamp > all_corr_ymax:
                all_corr_ymax = maxamp
            xds[f"{parallel_hand}_amplitude"] = xr.DataArray(amp, dims="lm_dist")
            xds[f"{parallel_hand}_phase"] = xr.DataArray(
                np.angle(avg_vis[:, icorr]), dims="lm_dist"
            )
            xds[f"{parallel_hand}_weight"] = xr.DataArray(
                avg_wei[:, icorr], dims="lm_dist"
            )
        xds.attrs.update({"all_corr_ymax": all_corr_ymax})
        cut_xdtree = cut_xdtree.assign(
            {
                f"cut_{n_cut}": xr.DataTree(
                    dataset=xds.assign_coords(coords), name=f"cut_{n_cut}"
                )
            }
        )
        n_cut += 1

    return cut_xdtree, n_cut


def _cut_direction_determination_and_label_creation(lm_offsets, angle_unit="deg"):
    """
    Determines cut's direction using a linear regression between L and M offsets.

    :param lm_offsets: Array containing the cut's L and M offsets over type expected to be of shape [n_time, lm]
    :type lm_offsets: numpy.ndarray

    :param angle_unit: Unit to represent cut's direction in a mixed cut.
    :type angle_unit: str

    :return: Tuple containing the cuts direction angle in the sky, distance from center for each point, direction \
    label, and x-axis label for plots.
    :rtype: tuple([float, numpy.array, str, str])

    """
    dx = lm_offsets[-1, 0] - lm_offsets[0, 0]
    dy = lm_offsets[-1, 1] - lm_offsets[0, 1]

    # determine where to flip signal of distances
    lm_dist = np.sqrt(lm_offsets[:, 0] ** 2 + lm_offsets[:, 1] ** 2)
    imin_lm = np.argmin(lm_dist)
    x_min, y_min = lm_offsets[imin_lm, :]
    lm_dist[:imin_lm] = -lm_dist[:imin_lm]

    if np.isclose(dx, dy, rtol=3e-1):  # X case
        result = linregress(lm_offsets[:, 0], lm_offsets[:, 1])
        lm_angle = np.arctan(result.slope) + np.pi / 2
        direction = "("
        if dy < 0 and dx < 0:
            direction += "NW -> SE"
            # Fix the sign of the minimum
            if x_min > 0:
                lm_dist[imin_lm] *= -1

        elif dy < 0 < dx:
            # Fix the sign of the minimum
            if x_min < 0:
                lm_dist[imin_lm] *= -1
            direction += "NE -> SW"

        elif dy > 0 > dx:
            # Fix the sign of the minimum
            if x_min > 0:
                lm_dist[imin_lm] *= -1
            direction += "SW -> NE"

        else:
            # Fix the sign of the minimum
            if x_min < 0:
                lm_dist[imin_lm] *= -1
            direction += "SE -> NW"

        direction += (
            r", $\theta$ = "
            + f"{format_value_unit(convert_unit('rad', angle_unit, 'trigonometric') * lm_angle, angle_unit)}"
        )
        xlabel = "Mixed offset"
    elif np.abs(dy) > np.abs(dx):  # Elevation case
        result = linregress(lm_offsets[:, 1], lm_offsets[:, 0])
        lm_angle = np.arctan(result.slope)
        # Fix the sign of the minimum
        if y_min < 0:
            lm_dist[imin_lm] *= -1
        direction = "("
        if dy < 0:
            direction += "N -> S"
            lm_dist *= -1  # Flip as sense is negative
        else:
            direction += "S -> N"
        xlabel = "Elevation offset"
    else:  # Azimuth case
        result = linregress(lm_offsets[:, 0], lm_offsets[:, 1])
        lm_angle = np.arctan(result.slope) + np.pi / 2
        # Fix the sign of the minimum
        if x_min < 0:
            lm_dist[imin_lm] *= -1
        direction = "("
        if dx > 0:
            direction += "E -> W"
        else:
            direction += "W -> E"
            lm_dist *= -1  # Flip as sense is negative
        xlabel = "Azimuth offset"

    direction += ")"

    return lm_angle, lm_dist, direction, xlabel


def _get_parallel_hand_indexes(corr_axis):
    """
    Get the indices of parallel hands along the correlation axis.

    :param corr_axis: Visibilities correlation axis
    :rtype: numpy.array(str)

    :return: Dictionary containing the parallel hands and their indices
    :rtype: dict
    """
    if "L" in corr_axis[0] or "R" in corr_axis[0]:
        parallel_hands = ["RR", "LL"]
    else:
        parallel_hands = ["XX", "YY"]

    hands_dict = {}
    for icorr, corr in enumerate(corr_axis):
        if corr in parallel_hands:
            hands_dict[corr] = icorr
    return hands_dict


###########################################################
### Multiple side lobe Gaussian fitting
###########################################################
def _fwhm_gaussian(x_axis, x_off, amp, fwhm):
    """
    Returns a gaussian of the same shape as x_axis with fwhm instead of sigma as the input parameter

    :param x_axis: X-axis of the beam cut data
    :type x_axis: numpy.array

    :param x_off: X offset of the center of the gaussian
    :type x_off: float

    :param amp: Amplitude of the gaussian
    :type amp: float

    :param fwhm: Full width at half maximum of the gaussian
    :type fwhm: float

    :return: Gaussian evaluated at x_axis points
    :rtype: numpy.array
    """
    sigma = fwhm / sig_2_fwhm
    return amp * np.exp(-((x_axis - x_off) ** 2) / (2 * sigma**2))


def _build_multi_gaussian_initial_guesses(
    x_data, y_data, pb_fwhm, datalabel, min_dist_fraction=1.0, max_n_pbs=15
):
    """
    Build initial guesses array for a multi gaussian fitting from X and Y axes heuristics

    :param x_data: Fit's X-axis data.
    :type x_data: numpy.array

    :param y_data: Fit's Y-axis data.
    :type  y_data: numpy.array

    :param pb_fwhm: Estimated FWHM of the primary beam
    :type pb_fwhm: float

    :param min_dist_fraction: Fraction of pb_fwhm to use as estimate for the minimal peak distance
    :type min_dist_fraction: float

    :return: Tuple containing the initial_guesses, bounds and number of peaks to fit
    :rtype: tuple([list, list([list]), int])
    """

    initial_guesses = []
    lower_bounds = []
    upper_bounds = []
    step = float(np.median(np.diff(x_data)))
    min_dist = np.abs(min_dist_fraction * pb_fwhm / step)
    x_range = x_data[-1] - x_data[0]
    n_pbs_in_range = x_range / pb_fwhm
    if min_dist < 1:
        min_dist = 1

    peaks, _ = find_peaks(y_data, distance=min_dist)

    if len(peaks) == 0:
        peaks = np.array([x_data.size // 2])
    if n_pbs_in_range > max_n_pbs:
        logger.warning(f"{datalabel} sampling is erratic, fit will probably fail")
        peak_interval = int(np.ceil(len(peaks) / max_n_pbs))
        peaks = peaks[::peak_interval]

    dx = x_data[-1] - x_data[0]
    if dx < 0:
        peaks = peaks[::-1]
    for ipeak in peaks:
        initial_guesses.extend([x_data[ipeak], y_data[ipeak], pb_fwhm])
        lower_bounds.extend([-np.inf, 0, 0])
        upper_bounds.extend([np.inf, np.inf, np.inf])
    bounds = (lower_bounds, upper_bounds)
    return initial_guesses, bounds, len(peaks)


def _multi_gaussian(xdata, *args):
    """
    Produces a multiple gaussian Y array from parameters derived fromm list of arguments

    :param xdata: X-axis data
    :type xdata: numpy.array

    :param args: List of gaussian parameters [x_off_0, amp_0, fwhm_0, ..., x_off_n, amp_n, fwhm_n]
    :type args: list([float])

    :return: Multiple gaussian evaluated at x-axis points
    :rtype: numpy.array
    """
    nargs = len(args)
    if nargs % 3 != 0:
        raise ValueError(f"Number of arguments should be multiple of 3, (got {nargs})")
    y_values = np.zeros_like(xdata)
    for iarg in range(0, nargs, 3):
        y_values += _fwhm_gaussian(xdata, args[iarg], args[iarg + 1], args[iarg + 2])
    return y_values


def _perform_curvefit_with_given_functions(
    x_data, y_data, initial_guesses, bounds, fit_func, datalabel, maxit=5000
):
    """
    Invoke scipy optimize curve_fit with customized parameters

    :param x_data: x-axis data for the curve fit
    :type x_data: numpy.array

    :param y_data: y-axis data to be fitted
    :type y_data: numpy.array

    :param initial_guesses: list of initial guesses
    :type initial_guesses: list

    :param bounds: List containing the lists of lower and upper bounds for each parameter
    :type bounds: list([list])

    :param fit_func: Function with which to fit the parameters.
    :type fit_func: function

    :param datalabel: Data label for messaging
    :type datalabel: str

    :param maxit: Maximum number of iterations
    :type maxit: int

    :return: Tuple containing sucess flag and fit results (NaNs if fit failed)
    :rtype: tuple([bool, numpy.array])
    """
    try:
        results = curve_fit(
            fit_func,
            x_data,
            y_data,
            p0=initial_guesses,
            bounds=bounds,
            maxfev=int(maxit),
        )
        fit_pars = results[0]
        return True, fit_pars
    except RuntimeError:
        logger.warning(f"{fit_func.__name__} fit to lobes failed for {datalabel}.")
        return False, np.full_like(initial_guesses, np.nan)


def _identify_pb_and_sidelobes_in_fit(
    datalabel, x_data, fit_pars, primary_fwhm_estimate
):
    """
    Identify primary beam and first sidelobes in fit using expected beam shape heuristics.

    :param datalabel: Data label for messaging
    :type datalabel: str

    :param x_data: X-axis data
    :type x_data: numpy.array

    :param fit_pars: Fit parameters
    :type fit_pars: numpy.array

    :return: Tuple containing: Number of peaks in beam, filtered fit parameters, primary beam center offset, primary \
    beam measured fwhm, ratio between left and right first sidelobes
    :rtype: tuple([int, numpy.array, float, float, float])
    """
    centers = fit_pars[0::3]
    amps = fit_pars[1::3]
    fwhms = fit_pars[2::3]

    # select fits that are within x_data
    x_min = np.min(x_data)
    x_max = np.max(x_data)
    selection = ~np.logical_or(centers < x_min, centers > x_max)

    # apply selection
    centers = centers[selection]
    amps = amps[selection]
    fwhms = fwhms[selection]

    # Reconstruct fit metadata
    n_peaks = centers.shape[0]
    fit_pars = np.zeros(3 * n_peaks)
    fit_pars[0::3] = centers
    fit_pars[1::3] = amps
    fit_pars[2::3] = fwhms

    # This assumes the primary beam is the closest to the center, which is expected
    i_pb_cen = np.argmin(np.abs(centers))
    # This assumes the primary beam is the strongest
    i_pb_amp = np.argmax(amps)

    pb_problem = False
    if i_pb_cen != i_pb_amp:
        pb_problem = True
    else:
        if (
            fwhms[i_pb_cen] < 0.5 * primary_fwhm_estimate
            or fwhms[i_pb_cen] > 3.0 * primary_fwhm_estimate
        ):
            pb_problem = True

    if pb_problem:
        logger.warning(
            f"Cannot reliably identify primary beam for {datalabel}, assuming largest amplitude peak."
        )
        pb_center = centers[i_pb_amp]
        pb_fwhm = fwhms[i_pb_amp]
        first_side_lobe_ratio = np.nan

    else:
        pb_fwhm = fwhms[i_pb_cen]
        pb_center = centers[i_pb_cen]

        pb_cen = centers[i_pb_cen]
        i_closest_to_center = np.argsort(np.abs(centers - pb_cen))
        n_peaks = i_closest_to_center.shape[0]
        if n_peaks >= 3:
            if centers[i_closest_to_center[1]] < 0:
                i_lsl = i_closest_to_center[1]
                i_rsl = i_closest_to_center[2]
            else:
                i_lsl = i_closest_to_center[2]
                i_rsl = i_closest_to_center[1]
            left_first_sl_amp = amps[i_lsl]
            right_first_sl_amp = amps[i_rsl]
            first_side_lobe_ratio = left_first_sl_amp / right_first_sl_amp
        else:
            logger.warning(
                f"Only {n_peaks} peaks identified for {datalabel}, cannot provide a side lobe ratio"
            )
            first_side_lobe_ratio = np.nan

    return n_peaks, fit_pars, pb_center, pb_fwhm, first_side_lobe_ratio


def estimate_pb_center_and_fwhm_from_data(x_data, y_data, datalabel):
    """
    Estimate primary beam parameters forcefully from data
    :param x_data: Cut X data
    :param y_data: Cut Y data
    :param datalabel: Data label
    :return: estimated primary beam center, fwhm and amplitude.
    """
    logger.info(
        f"Estimating primary beam center and fwhm from maximum amplitude for {datalabel}"
    )
    i_max = np.argmax(y_data.values)
    amp_max = y_data.values[i_max]
    pb_center = x_data[i_max]
    n_pnt = x_data.shape[0]

    hw_plus = np.nan
    for i_pnt in range(i_max, n_pnt):
        if y_data[i_pnt] <= amp_max / 2:
            hw_plus = x_data[i_pnt]
            break
    hw_minus = np.nan
    for i_pnt in range(i_max, 0, -1):
        if y_data[i_pnt] <= amp_max / 2:
            hw_minus = x_data[i_pnt]
            break
    pb_fwhm = hw_plus - hw_minus
    fslr = np.nan

    return pb_center, pb_fwhm, fslr, amp_max


def _beamcut_multi_lobes_gaussian_fit(cut_xdtree, datalabel):
    """
    Execute multi gaussian fit to beam cut data.

    :param cut_xdtree: Datatree containing a single beam cut.
    :type cut_xdtree: xarray.DataTree

    :param datalabel: Data label for messaging
    :type datalabel: str

    :return: None
    :rtype: NoneType
    """
    # Get the summary from the first cut, but it should be equal anyway
    summary = cut_xdtree.attrs["summary"]
    wavelength = summary["spectral"]["rep. wavelength"]
    telescope = get_proper_telescope(
        summary["general"]["telescope name"], summary["general"]["antenna name"]
    )
    primary_fwhm = 1.2 * wavelength / telescope.diameter
    for cut_xds in cut_xdtree.children.values():
        cut_xds.attrs["expected_pb_fwhm"] = primary_fwhm
        x_data = cut_xds["lm_dist"].values
        for parallel_hand in cut_xds.attrs["available_corrs"]:
            y_data = cut_xds[f"{parallel_hand}_amplitude"]
            this_corr_data_label = (
                f"{datalabel}, {cut_xds.attrs['direction']}, corr = {parallel_hand}"
            )
            initial_guesses, bounds, n_peaks = _build_multi_gaussian_initial_guesses(
                x_data, y_data, primary_fwhm, this_corr_data_label
            )

            # This is a test for empty data
            if np.nansum(y_data) == 0:
                logger.warning(
                    f"{this_corr_data_label} appears to contain no valid data"
                )
                fit_succeeded = False
                fit_pars = np.full_like(initial_guesses, np.nan)
            else:
                fit_succeeded, fit_pars = _perform_curvefit_with_given_functions(
                    x_data,
                    y_data,
                    initial_guesses,
                    bounds,
                    _multi_gaussian,
                    this_corr_data_label,
                )

            if fit_succeeded:
                fit = _multi_gaussian(x_data, *fit_pars)
                n_peaks, fit_pars, pb_center, pb_fwhm, first_side_lobe_ratio = (
                    _identify_pb_and_sidelobes_in_fit(
                        this_corr_data_label, x_data, fit_pars, primary_fwhm
                    )
                )
            else:
                pb_center, pb_fwhm, first_side_lobe_ratio, amp_max = (
                    estimate_pb_center_and_fwhm_from_data(
                        x_data, y_data, this_corr_data_label
                    )
                )
                fit = _multi_gaussian(x_data, pb_center, amp_max, pb_fwhm)
                n_peaks = 1
                fit_pars = [pb_center, amp_max, pb_fwhm]
                if np.all(np.isfinite(fit_pars)):
                    fit_succeeded = True

            cut_xds.attrs[f"{parallel_hand}_amp_fit_pars"] = fit_pars
            cut_xds.attrs[f"{parallel_hand}_n_peaks"] = n_peaks
            cut_xds.attrs[f"{parallel_hand}_pb_fwhm"] = pb_fwhm
            cut_xds.attrs[f"{parallel_hand}_pb_center"] = pb_center
            cut_xds.attrs[f"{parallel_hand}_first_side_lobe_ratio"] = (
                first_side_lobe_ratio
            )
            cut_xds.attrs[f"{parallel_hand}_fit_succeeded"] = fit_succeeded

            cut_xds[f"{parallel_hand}_amp_fit"] = xr.DataArray(fit, dims="lm_dist")
    return
