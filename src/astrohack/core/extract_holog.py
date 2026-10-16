import os

import astropy
import numpy as np
import toolviper.utils.logger as logger
import xarray as xr
from casacoretables import tables as ctables
from numba import njit
from numba.core import types

from astrohack.antenna.telescope import get_proper_telescope
from astrohack.core.holog_obs_dict import HologObsDict
from astrohack.io.holog_mds import AstrohackHologFile
from astrohack.utils.algorithms import calculate_optimal_grid_parameters
from astrohack.utils.constants import clight, njit_caching, pol_str, twopi
from astrohack.utils.conversion import casa_time_to_mjd
from astrohack.utils.gridding import grid_1d_data
from astrohack.utils.imaging import calculate_parallactic_angle_chunk
from astrohack.utils.text import create_dataset_label, undscr
from astrohack.utils.tools import get_valid_state_ids


def extract_holog_preprocessing(extract_holog_params, pnt_mds):
    holog_obs_dict = extract_holog_params["holog_obs_dict"]

    # Get spectral windows
    ctb = ctables.table(
        os.path.join(extract_holog_params["ms_name"], "DATA_DESCRIPTION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ddi_spw = ctb.getcol("SPECTRAL_WINDOW_ID")
    ddpol_indexol = ctb.getcol("POLARIZATION_ID")
    ctb.close()

    ant_names = pnt_mds.root.attrs["antenna_names"]
    ant_ids = pnt_mds.root.attrs["antenna_ids"]
    ant_stations = pnt_mds.root.attrs["antenna_stations"]

    # Create holog_obs_dict or modify user supplied holog_obs_dict.
    if holog_obs_dict is None:
        holog_obs_dict = HologObsDict.create_from_pointing_file(
            pnt_mds=pnt_mds,
            exclude_antennas=extract_holog_params["exclude_antennas"],
            baseline_average_distance=extract_holog_params["baseline_average_distance"],
            baseline_average_nearest=extract_holog_params["baseline_average_nearest"],
        )

    user_ddi_sel = extract_holog_params["ddi"]
    if isinstance(user_ddi_sel, int):
        user_ddi_sel = [user_ddi_sel]

    if user_ddi_sel != "all":
        holog_obs_dict.select("ddi", user_ddi_sel)

    user_ant_sel = extract_holog_params["ant"]
    if isinstance(user_ant_sel, int):
        user_ant_sel = [user_ant_sel]

    if user_ant_sel != "all":
        holog_obs_dict.select("antenna", user_ant_sel)

    ctb = ctables.table(
        os.path.join(extract_holog_params["ms_name"], "STATE"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    # Scan intent (with subscan intent) is stored in the OBS_MODE column of the STATE sub-table.
    obs_modes = ctb.getcol("OBS_MODE")
    ctb.close()

    state_ids = get_valid_state_ids(obs_modes)

    spw_ctb = ctables.table(
        os.path.join(extract_holog_params["ms_name"], "SPECTRAL_WINDOW"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    pol_ctb = ctables.table(
        os.path.join(extract_holog_params["ms_name"], "POLARIZATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    # If we have an EVLA run from before 2023 the pointing table needs to be fixed.
    n_mapping = 0
    n_ref_ants = 0
    for ddi_key, ddi_dict in holog_obs_dict.items():
        n_map_ddi = 0
        for map_dict in ddi_dict.values():
            n_map_ddi += len(map_dict["ant"])
            for ant in map_dict["ant"].values():
                n_ref_ants += len(ant)
        n_mapping += n_map_ddi

    if n_mapping == 0:
        msg = "Data selection excludes all data, consider creating a customized holog_obs_dict with generate_holog_obs_dict"
        logger.error(msg)
        raise RuntimeError(msg)
    if n_ref_ants == 0:
        msg = "No reference antennas found for any of the DDIs"
        logger.error(msg)
        raise RuntimeError(msg)

    looping_dict = {}
    for ddi_key in holog_obs_dict.keys():
        ddi = int(ddi_key.replace("ddi_", ""))
        spw_setup_id = ddi_spw[ddi]
        pol_setup_id = ddpol_indexol[ddi]

        chan_freq = spw_ctb.getcol("CHAN_FREQ", startrow=spw_setup_id, nrow=1)[0, :]
        chan_width = spw_ctb.getcol("CHAN_WIDTH", startrow=spw_setup_id, nrow=1)[0, :]
        eff_bw = spw_ctb.getcol("EFFECTIVE_BW", startrow=spw_setup_id, nrow=1)[0, :]
        ref_freq = spw_ctb.getcol("REF_FREQUENCY", startrow=spw_setup_id, nrow=1)[0]
        total_bw = spw_ctb.getcol("TOTAL_BANDWIDTH", startrow=spw_setup_id, nrow=1)[0]
        chan_setup = {
            "chan_freq": chan_freq,
            "chan_width": chan_width,
            "eff_bw": eff_bw,
            "ref_freq": ref_freq,
            "total_bw": total_bw,
        }
        pol_setup = {
            "pol": pol_str[
                pol_ctb.getcol("CORR_TYPE", startrow=pol_setup_id, nrow=1)[0, :]
            ]
        }

        ddi_dict = {}
        for map_key, map_data in holog_obs_dict[ddi_key].items():
            scans = map_data["scans"]
            if len(scans) > 1:
                logger.info(
                    f"Pre-processing ddi: {ddi}, scans: [{scans[0]} ... {scans[-1]}]"
                )
            else:
                logger.info(f"Pre-processing ddi: {ddi}, scan: {scans}")

            if len(list(map_data["ant"].keys())) != 0:
                map_ant_list = []
                ref_ant_per_map_ant_list = []

                map_ant_name_list = []
                ref_ant_per_map_ant_name_list = []
                ref_ant_stations_list = []
                for map_ant_name, ref_ant_name_list in map_data["ant"].items():
                    ref_ant_ids = _convert_ant_name_to_id(
                        ref_ant_name_list,
                        ant_names,
                        ant_ids,
                    )

                    map_ant_id = _convert_ant_name_to_id(
                        map_ant_name, ant_names, ant_ids
                    )[0]

                    ref_ant_per_map_ant_list.append(ref_ant_ids)
                    map_ant_list.append(map_ant_id)

                    ref_ant_per_map_ant_name_list.append(
                        list(map_data["ant"][map_ant_name])
                    )
                    ref_ant_stations = _convert_ant_name_to_id(
                        ref_ant_name_list, ant_names, ant_stations
                    )
                    ref_ant_stations_list.append(ref_ant_stations)
                    map_ant_name_list.append(map_ant_name)

                map_dict = {
                    "ref_ant_per_map_ant_tuple": tuple(ref_ant_per_map_ant_list),
                    "map_ant_tuple": tuple(map_ant_list),
                    "ref_ant_per_map_ant_name_tuple": tuple(
                        ref_ant_per_map_ant_name_list
                    ),
                    "map_ant_name_tuple": tuple(map_ant_name_list),
                    "ant_stations": ant_stations,
                    "scans": scans,
                    "sel_state_ids": state_ids,
                    "ant_names": ant_names,
                    "chan_setup": chan_setup,
                    "pol_setup": pol_setup,
                }
                ddi_dict[map_key] = map_dict

            else:
                logger.warning(
                    f"{create_dataset_label(ddi_key, map_key)} has no holography data to extract."
                )
        looping_dict[ddi_key] = ddi_dict

    spw_ctb.close()
    pol_ctb.close()

    return looping_dict, holog_obs_dict


def process_extract_holog_chunk(
    extract_holog_params: dict, holog_mds: AstrohackHologFile
):
    """Perform data query on holography data chunk and get unique time and state_ids/

    Args:
        extract_holog_params (dict): parameters controlling work to be done on this chunk
        holog_mds (AstrohackHologFile): Output holog mds file
    """

    ms_name = extract_holog_params["ms_name"]
    data_column = extract_holog_params["data_column"]
    time_interval = extract_holog_params["time_smoothing_interval"]
    pointing_interpolation_method = extract_holog_params[
        "pointing_interpolation_method"
    ]
    holog_name = extract_holog_params["holog_name"]

    pnt_mds = extract_holog_params["pnt_mds"]
    inp_data_dict = extract_holog_params["dic_data"]

    ddi_key = extract_holog_params["this_ddi"]
    map_key = extract_holog_params["this_map"]

    ddi_id = int(ddi_key.replace("ddi_", ""))

    scans = inp_data_dict["scans"]
    ant_names = inp_data_dict["ant_names"]
    ant_stations = inp_data_dict["ant_stations"]
    ref_ant_per_map_ant_tuple = inp_data_dict["ref_ant_per_map_ant_tuple"]
    map_ant_tuple = inp_data_dict["map_ant_tuple"]
    map_ant_name_tuple = inp_data_dict["map_ant_name_tuple"]
    sel_state_ids = inp_data_dict["sel_state_ids"]
    chan_setup = inp_data_dict["chan_setup"]
    pol = inp_data_dict["pol_setup"]["pol"]
    chan_freq = chan_setup["chan_freq"]

    # This piece of information is no longer used leaving them here commented out for completeness
    # ref_ant_per_map_ant_name_tuple = extract_holog_params["ref_ant_per_map_ant_name_tuple"]

    if len(ref_ant_per_map_ant_tuple) != len(map_ant_tuple):
        msg = "Reference antenna per mapping antenna list and mapping antenna list should have same length."
        logger.error(msg)
        raise RuntimeError(msg)

    table_obj = ctables.table(
        ms_name, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    )
    scans = [int(scan) for scan in scans]
    if sel_state_ids:
        ctb = ctables.taql(
            "select %s, SCAN_NUMBER, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG, FIELD_ID from "
            "$table_obj WHERE DATA_DESC_ID == %s AND SCAN_NUMBER in %s AND STATE_ID in %s"
            % (data_column, ddi_id, scans, list(sel_state_ids))
        )
    else:
        ctb = ctables.taql(
            "select %s, SCAN_NUMBER, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG, FIELD_ID from "
            "$table_obj WHERE DATA_DESC_ID == %s AND SCAN_NUMBER in %s"
            % (data_column, ddi_id, scans)
        )
    vis_data = ctb.getcol(data_column)
    weight = ctb.getcol("WEIGHT")
    ant1 = ctb.getcol("ANTENNA1")
    ant2 = ctb.getcol("ANTENNA2")
    time_vis_row = ctb.getcol("TIME")
    # Centroid is never used, hence it is commented out to improve efficiency
    # time_vis_row_centroid = ctb.getcol("TIME_CENTROID")
    flag = ctb.getcol("FLAG")
    flag_row = ctb.getcol("FLAG_ROW")
    scan_list = ctb.getcol("SCAN_NUMBER")
    field_ids = ctb.getcol("FIELD_ID")

    gen_info = _get_general_summary(ms_name, field_ids)

    # Here we use the median of the differences between dumps as this is a good proxy for the integration time
    if time_interval is None:
        time_interval = np.median(np.diff(np.unique(time_vis_row)))

    ctb.close()
    table_obj.close()

    map_ref_dict = _get_map_ref_dict(
        map_ant_tuple, ref_ant_per_map_ant_tuple, ant_names, ant_stations
    )

    (
        time_vis,
        vis_map_dict,
        weight_map_dict,
        flagged_mapping_antennas,
        used_samples_dict,
        scan_time_ranges,
        unq_scans,
    ) = _match_visibilities_to_time_bins(
        vis_data,
        weight,
        ant1,
        ant2,
        time_vis_row,
        flag,
        flag_row,
        ref_ant_per_map_ant_tuple,
        map_ant_tuple,
        time_interval,
        scan_list,
    )
    del vis_data, weight, ant1, ant2, time_vis_row, flag, flag_row, field_ids, scan_list

    map_ant_name_list = list(map(str, map_ant_name_tuple))

    map_ant_name_list = [undscr.join(("ant", i)) for i in map_ant_name_list]

    pnt_map_dict = _extract_pointing_chunk(
        map_ant_name_list, time_vis, pnt_mds, pointing_interpolation_method
    )

    grid_params = {}

    # The loop has been moved out of the function here making the gridding parameter auto-calculation
    # function more general use (hopefully). I honestly couldn't see a reason to keep it inside.
    for ant_index in vis_map_dict.keys():
        antenna_name = undscr.join(("ant", ant_names[ant_index]))
        telescope = get_proper_telescope(gen_info["telescope name"], antenna_name)
        n_pix, cell_size = calculate_optimal_grid_parameters(
            pnt_map_dict, antenna_name, telescope.diameter, chan_freq, ddi_id
        )
        grid_params[antenna_name] = {"n_pix": n_pix, "cell_size": cell_size}

    # ## To DO: ################## Average multiple repeated samples over_flow_protector_constant = float("%.5g" %
    # time_vis[0])  # For example 5076846059.4 -> 5076800000.0 time_vis = time_vis - over_flow_protector_constant
    # from astrohack.utils._algorithms import _average_repeated_pointings time_vis = _average_repeated_pointings(
    # vis_map_dict, weight_map_dict, flagged_mapping_antennas,time_vis,pnt_map_dict,ant_names) time_vis = time_vis +
    # over_flow_protector_constant

    _create_holog_file(
        holog_name,
        vis_map_dict,
        weight_map_dict,
        pnt_map_dict,
        time_vis,
        used_samples_dict,
        chan_freq,
        pol,
        flagged_mapping_antennas,
        map_key,
        ddi_key,
        ms_name,
        ant_names,
        grid_params,
        time_interval,
        gen_info,
        map_ref_dict,
        scan_time_ranges,
        unq_scans,
        holog_mds,
    )

    logger.info(f"Finished extracting holography chunk for DDI {ddi_id}, {map_key}.")


def _get_map_ref_dict(
    map_ant_tuple, ref_ant_per_map_ant_tuple, ant_names, ant_stations
):
    map_dict = {}
    for ii, map_id in enumerate(map_ant_tuple):
        map_name = ant_names[map_id]
        ref_list = []
        for ref_id in ref_ant_per_map_ant_tuple[ii]:
            ref_list.append(f"{ant_names[ref_id]} @ {ant_stations[ref_id]}")
        map_dict[map_name] = ref_list
    return map_dict


@njit(cache=njit_caching, nogil=True)
def _get_time_intervals(time_vis_row, scan_list, time_interval):
    unq_scans = np.unique(scan_list)
    scan_time_ranges = []
    for scan in unq_scans:
        selected_times = time_vis_row[scan_list == scan]
        min_time, max_time = np.min(selected_times), np.max(selected_times)
        scan_time_ranges.append([min_time, max_time])

    half_int = time_interval / 2
    start = np.min(time_vis_row) + half_int
    total_time = np.max(time_vis_row) - start
    n_time = int(np.ceil(total_time / time_interval)) + 1
    stop = start + n_time * time_interval
    raw_time_samples = np.linspace(start, stop, n_time + 1)

    filtered_time_samples = []
    for time_sample in raw_time_samples:
        for time_range in scan_time_ranges:
            if time_range[0] <= time_sample <= time_range[1]:
                filtered_time_samples.append(time_sample)
                break
    time_samples = np.array(filtered_time_samples)
    return time_samples, scan_time_ranges, unq_scans


@njit(cache=njit_caching, nogil=True)
def _match_visibilities_to_time_bins(
    vis_data,
    weight,
    ant1,
    ant2,
    time_vis_row,
    flag,
    flag_row,
    ref_ant_per_map_ant_tuple,
    map_ant_tuple,
    time_interval,
    scan_list,
):
    """function to extract relevant visibility data from chunk after flagging and applying weights.

    Args:
        vis_data (numpy.ndarray): Visibility data (row, channel, polarization)
        weight (numpy.ndarray): Data weight values (row, polarization)
        ant1 (numpy.ndarray): List of antenna_ids for antenna1
        ant2 (numpy.ndarray): List of antenna_ids for antenna2
        time_vis_row (numpy.ndarray): Array of full time values by row
        flag (numpy.ndarray): Array of data quality flags to apply to data
        flag_row (numpy.ndarray): Array indicating when a full row of data should be flagged
        ref_ant_per_map_ant_tuple(tuple): reference antenna per mapping antenna
        map_ant_tuple(tuple): mapping antennas?
        time_interval(float): time smoothing interval
        scan_list(list): list of valid holography scans

    Returns:
        dict: Antenna_id referenced (key) dictionary containing the visibility data selected by (time, channel,
        polarization)
    """
    time_samples, scan_time_ranges, unq_scans = _get_time_intervals(
        time_vis_row, scan_list, time_interval
    )
    n_time = len(time_samples)
    n_row, n_chan, n_pol = vis_data.shape
    half_int = time_interval / 2

    flagged_mapping_antennas = []
    vis_map_dict = {}
    sum_weight_map_dict = {}
    used_samples_dict = {}
    vis_shape = (n_time, n_chan, n_pol)
    for ant_id in map_ant_tuple:
        refant_ids = ref_ant_per_map_ant_tuple[map_ant_tuple.index(ant_id)]

        # To be uncommented if trying to debug code when unjitted.
        # ant_vis_sum = np.zeros(vis_shape, dtype=np.complex128)
        # ant_wei_sum = np.zeros(vis_shape, dtype=np.float64)
        # ant_valid_data = np.full(n_time, False)
        # ant_select = np.full(n_row, False)

        # To be commented if trying to debug code when unjitted.
        ant_vis_sum = np.zeros(vis_shape, dtype=types.complex128)
        ant_wei_sum = np.zeros(vis_shape, dtype=types.float64)
        ant_valid_data = np.full(n_time, False, dtype=types.bool)
        ant_select = np.full(n_row, False, dtype=types.bool)

        for refant_id in refant_ids:
            ant_select = np.logical_or(
                np.logical_and(ant1 == ant_id, ant2 == refant_id),
                np.logical_and(ant2 == ant_id, ant1 == refant_id),
                ant_select,
            )

        ant2_selected = ant2[ant_select]
        vis_selected = vis_data[ant_select, :, :]
        wei_selected = weight[ant_select, :]
        flag_selected = flag[ant_select, :, :]
        flag_row_selected = flag_row[ant_select]
        time_selected = time_vis_row[ant_select]

        time_index = 0
        total_wei_sum = 0.0
        for row, row_flag in enumerate(flag_row_selected):
            if row_flag:
                continue
            if time_selected[row] < time_samples[time_index] - half_int:
                continue
            else:
                time_index = _get_time_index(
                    time_selected[row], time_index, time_samples, half_int
                )
            if time_index < 0:
                break

            is_conjugate = ant2_selected[row] == ant_id
            for chan in range(n_chan):
                for pol in range(n_pol):
                    if flag_selected[row, chan, pol]:
                        pass
                    else:
                        ant_valid_data[time_index] = True
                        if is_conjugate:
                            vis_val = np.conjugate(vis_selected[row, chan, pol])
                        else:
                            vis_val = vis_selected[row, chan, pol]
                        wei_val = wei_selected[row, pol]
                        ant_vis_sum[time_index, chan, pol] += vis_val * wei_val
                        ant_wei_sum[time_index, chan, pol] += wei_val
                        total_wei_sum += wei_val

        if total_wei_sum == 0:
            flagged_mapping_antennas.append(ant_id)
            ant_vis_sum[:, :, :] = 0.0 + 0j
        else:
            for row in range(vis_shape[0]):
                for chan in range(vis_shape[1]):
                    for pol in range(vis_shape[2]):
                        if ant_wei_sum[row, chan, pol] == 0:
                            ant_vis_sum[row, chan, pol] = 0
                        else:
                            ant_vis_sum[row, chan, pol] /= ant_wei_sum[row, chan, pol]
        vis_map_dict[ant_id] = ant_vis_sum
        sum_weight_map_dict[ant_id] = ant_wei_sum
        used_samples_dict[ant_id] = ant_valid_data

    return (
        time_samples,
        vis_map_dict,
        sum_weight_map_dict,
        flagged_mapping_antennas,
        used_samples_dict,
        scan_time_ranges,
        unq_scans,
    )


def _get_time_samples(time_vis):
    """Sample three values for time vis and cooresponding indices. Values are sammpled as (first, middle, last)

    Args:
        time_vis (numpy.ndarray): a list of visibility times

    Returns:
        numpy.ndarray, list: a select subset of visibility times (first, middle, last)
    """

    n_time_vis = time_vis.shape[0]

    middle = int(n_time_vis // 2)
    indices = [0, middle, n_time_vis - 1]

    return np.take(time_vis, indices), indices


def _create_holog_file(
    holog_name,
    vis_map_dict,
    weight_map_dict,
    pnt_map_dict,
    time_vis,
    used_samples_dict,
    chan,
    pol,
    flagged_mapping_antennas,
    map_key,
    ddi_key,
    ms_name,
    ant_names,
    grid_params,
    time_interval,
    gen_info,
    map_ref_dict,
    scan_time_ranges,
    unq_scans,
    holog_mds: AstrohackHologFile,
):
    """Create holog-structured, formatted output file and save to zarr.

    Args:
        holog_name (str): holog file name.
        vis_map_dict (dict): a nested dictionary/map of weighted visibilities indexed as [antenna][time, chan, pol]; \
        mainains time ordering.
        weight_map_dict (dict): weights dictionary/map for visibilites in vis_map_dict
        pnt_map_dict (dict): pointing table map dictionary
        time_vis (numpy.ndarray): time_vis values
        chan (numpy.ndarray): channel values
        pol (numpy.ndarray): polarization values
        flagged_mapping_antennas (numpy.ndarray): list of mapping antennas that have been flagged.
        map_key(string): holog map id string
        ddi_key (string): data description id; a combination of polarization and spectral window
    """

    ctb = ctables.table("/".join((ms_name, "ANTENNA")), ack=False)
    observing_location = ctb.getcol("POSITION")
    ctb.close()

    for map_ant_index in vis_map_dict.keys():
        dataset_label = create_dataset_label(
            ant_names[map_ant_index], ddi_key.split(undscr)[1]
        )
        if map_ant_index not in flagged_mapping_antennas:
            map_ant_key = f"ant_{ant_names[map_ant_index]}"
            pnt_xds = pnt_map_dict[map_ant_key]
            vis_data = vis_map_dict[map_ant_index]
            wei_data = weight_map_dict[map_ant_index]
            valid_data = used_samples_dict[map_ant_index]
            ant_time_vis = time_vis[valid_data]

            time_vis_days = ant_time_vis / (3600 * 24)
            astro_time_vis = astropy.time.Time(time_vis_days, format="mjd")
            time_samples, indicies = _get_time_samples(astro_time_vis)
            coords = {"time": ant_time_vis, "chan": chan, "pol": pol}

            chosen_key = "DIRECTIONAL_COSINES"
            direction = np.take(
                pnt_xds[chosen_key].values,
                indicies,
                axis=0,
            )

            parallactic_samples = calculate_parallactic_angle_chunk(
                time_samples=time_samples,
                observing_location=observing_location[map_ant_index],
                direction=direction,
            )

            xds = xr.Dataset()
            xds = xds.assign_coords(coords)
            xds["VIS"] = xr.DataArray(
                vis_data[valid_data, ...],
                dims=["time", "chan", "pol"],
            )

            xds["WEIGHT"] = xr.DataArray(
                wei_data[valid_data, ...],
                dims=["time", "chan", "pol"],
            )

            xds["DIRECTIONAL_COSINES"] = xr.DataArray(
                pnt_xds[chosen_key].values[valid_data, ...],
                dims=["time", "lm"],
            )

            xds["IDEAL_DIRECTIONAL_COSINES"] = xr.DataArray(
                pnt_xds["POINTING_OFFSET"].values[valid_data, ...],
                dims=["time", "lm"],
            )

            xds.attrs["parallactic_samples"] = parallactic_samples
            xds.attrs["time_smoothing_interval"] = time_interval
            xds.attrs["scan_time_ranges"] = scan_time_ranges
            xds.attrs["scan_list"] = unq_scans

            xds.attrs["summary"] = _create_observation_summary(
                gen_info,
                grid_params,
                xds["DIRECTIONAL_COSINES"].values,
                chan,
                pnt_xds,
                valid_data,
                map_ref_dict,
            )

            holog_filename = holog_name

            logger.debug(f"Writing {dataset_label} holog data to {holog_filename}")
            holog_mds.add_node(xds, [map_ant_key, ddi_key, map_key])
        else:
            logger.warning(f"All data flagged for {dataset_label}")


def _extract_pointing_chunk(
    map_ant_ids, time_vis, pnt_ant_dict, pointing_interpolation_method
):
    """Averages pointing within the time sampling of the visibilities

    Args:
        map_ant_ids (list): list of antenna ids
        time_vis (numpy.ndarray): sorted, unique list of visibility times
        pnt_ant_dict (dict): map of pointing directional cosines with a map key based on the antenna id and indexed by
                             the MAIN table visibility time.

    Returns:
        dict:  Dictionary of directional cosine data mapped to nearest MAIN table sample times.
    """
    keys = ["DIRECTION", "DIRECTIONAL_COSINES", "ENCODER", "POINTING_OFFSET", "TARGET"]
    pnt_map_dict = {}
    coords = {"time": time_vis}
    for antenna in map_ant_ids:
        pnt_xds = pnt_ant_dict[antenna]
        y_data = []
        for key in keys:
            y_data.append(pnt_xds[key].values)
        pnt_time = pnt_xds.time.values

        resample_pnt = grid_1d_data(
            time_vis,
            pnt_time,
            y_data,
            pointing_interpolation_method,
            f"{antenna.split(undscr)[1]} pointing data",
            "visibility times",
        )

        new_pnt_xds = xr.Dataset()
        new_pnt_xds.assign_coords(coords)

        for i_key, key in enumerate(keys):
            new_pnt_xds[key] = xr.DataArray(resample_pnt[i_key], dims=("time", "az_el"))

        new_pnt_xds.attrs = pnt_xds.attrs
        pnt_map_dict[antenna] = new_pnt_xds
    return pnt_map_dict


@njit(cache=njit_caching, nogil=True)
def _get_time_index(data_time, i_time, time_axis, half_int):
    if i_time == time_axis.shape[0]:
        return -1
    while data_time > time_axis[i_time] + half_int:
        i_time += 1
        if i_time == time_axis.shape[0]:
            return -1
    return i_time


def _get_general_summary(ms_name, field_ids):
    unq_ids = np.unique(field_ids)
    field_tbl = ctables.table(
        ms_name + "::FIELD",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    i_src = int(unq_ids[0])
    src_name = field_tbl.getcol("NAME")
    phase_center_fk5 = field_tbl.getcol("PHASE_DIR")[:, 0, :]
    field_tbl.close()

    obs_table = ctables.table(
        ms_name + "::OBSERVATION",
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    time_range = casa_time_to_mjd(obs_table.getcol("TIME_RANGE")[0])
    telescope_name = obs_table.getcol("TELESCOPE_NAME")[0]
    obs_table.close()

    phase_center_fk5[:, 0] = np.where(
        phase_center_fk5[:, 0] < 0,
        phase_center_fk5[:, 0] + twopi,
        phase_center_fk5[:, 0],
    )

    gen_info = {
        "source": src_name[i_src],
        "phase center": phase_center_fk5[i_src].tolist(),
        "telescope name": telescope_name,
        "start time": time_range[0],  # start time is in MJD in days
        "stop time": time_range[-1],  # stop time is in MJD in days
        "duration": (time_range[-1] - time_range[0])
        * 86400,  # Store it in seconds rather than days
    }
    return gen_info


def _get_az_el_characteristics(pnt_map_xds, valid_data):
    az_el = pnt_map_xds["ENCODER"].values[valid_data, ...]
    lm = pnt_map_xds["DIRECTIONAL_COSINES"].values[valid_data, ...]
    mean_az_el = np.mean(az_el, axis=0)
    median_az_el = np.median(az_el, axis=0)
    lmmid = lm.shape[0] // 2
    lmquart = lmmid // 2
    ilow = lmmid - lmquart
    iupper = lmmid + lmquart + 1
    ic = np.argmin((lm[ilow:iupper, 0] ** 2 + lm[ilow:iupper, 1]) ** 2) + ilow
    center_az_el = az_el[ic]
    az_el_info = {
        "center": center_az_el.tolist(),
        "mean": mean_az_el.tolist(),
        "median": median_az_el.tolist(),
    }
    return az_el_info


def _get_freq_summary(chan_axis):
    chan_width = np.abs(chan_axis[1] - chan_axis[0])
    rep_freq = chan_axis[chan_axis.shape[0] // 2]
    freq_info = {
        "channel width": chan_width,
        "number of channels": chan_axis.shape[0],
        "frequency range": [
            chan_axis[0] - chan_width / 2,
            chan_axis[-1] + chan_width / 2,
        ],
        "rep. frequency": rep_freq,
        "rep. wavelength": clight / rep_freq,
    }

    return freq_info


def _create_observation_summary(
    obs_info,
    grid_params,
    lm,
    chan_axis,
    pnt_map_xds,
    valid_data,
    map_ref_dict,
):
    antenna_info = pnt_map_xds.attrs["antenna_info"]
    antenna_name = antenna_info["name"]
    station = antenna_info["station"]
    spw_info = _get_freq_summary(chan_axis)
    obs_info["az el info"] = _get_az_el_characteristics(pnt_map_xds, valid_data)
    obs_info["reference antennas"] = map_ref_dict[antenna_name]
    obs_info["antenna name"] = antenna_name
    obs_info["station"] = station

    l_max = np.max(lm[:, 0])
    l_min = np.min(lm[:, 0])
    m_max = np.max(lm[:, 1])
    m_min = np.min(lm[:, 1])

    beam_info = {
        "grid size": grid_params[f"ant_{antenna_name}"]["n_pix"],
        "cell size": grid_params[f"ant_{antenna_name}"]["cell_size"],
        "l extent": [l_min, l_max],
        "m extent": [m_min, m_max],
    }

    summary = {
        "spectral": spw_info,
        "beam": beam_info,
        "general": obs_info,
        "aperture": None,
    }
    return summary


def _convert_ant_name_to_id(
    ant_name_list,
    ref_ant_names,
    ref_ant_ids,
):
    """_summary_

    Args:
        ant_name_list (list): List of antennas for which to fetch ids
        ref_ant_names (list): Reference list of antenna names
        ref_ant_ids (list): Reference list of antenna ids

    Returns:
        list: List of antenna ids
    """
    if not isinstance(ant_name_list, list):
        ant_name_list = [ant_name_list]

    ant_ids_list = []
    for ant_name in ant_name_list:
        if ant_name in ref_ant_names:
            i_ids = ref_ant_names.index(ant_name)
            ant_ids_list.append(ref_ant_ids[i_ids])

    return ant_ids_list
