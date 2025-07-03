import os
import json

import numpy as np
import xarray as xr
import astropy
import toolviper.utils.logger as logger

from numba import njit
from numba.core import types

from casacore import tables as ctables

from astrohack.antenna import Telescope
from astrohack.utils import create_dataset_label
from astrohack.utils.imaging import calculate_parallactic_angle_chunk
from astrohack.utils.algorithms import calculate_optimal_grid_parameters
from astrohack.utils.conversion import casa_time_to_mjd
from astrohack.utils.constants import twopi, clight

from astrohack.utils.file import load_point_file


def process_extract_holog_chunk(extract_holog_params):
    """Perform data query on holography data chunk and get unique time and state_ids/

    Args:
        extract_holog_params: dictionary containing parameters

    Some of the parameters are:
        ms_name (str): Measurementset name
        data_column (str): Data column to extract.
        ddi (int): Data description id
        scan (int): Scan number
        map_ant_ids (numpy.narray): Array of antenna_id values corresponding to mapping data.
        ref_ant_ids (numpy.narray): Arry of antenna_id values corresponding to reference data.
        sel_state_ids (list): List pf state_ids corresponding to holography data
    """

    ms_name = extract_holog_params["ms_name"]
    pnt_name = extract_holog_params["point_name"]
    data_column = extract_holog_params["data_column"]
    ddi = extract_holog_params["ddi"]
    scans = extract_holog_params["scans"]
    ant_names = extract_holog_params["ant_names"]
    ant_station = extract_holog_params["ant_station"]
    ref_ant_per_map_ant_tuple = extract_holog_params["ref_ant_per_map_ant_tuple"]
    map_ant_tuple = extract_holog_params["map_ant_tuple"]
    map_ant_name_tuple = extract_holog_params["map_ant_name_tuple"]
    holog_map_key = extract_holog_params["holog_map_key"]
    time_interval = extract_holog_params["time_smoothing_interval"]

    # This piece of information is no longer used leaving them here commented out for completeness
    # ref_ant_per_map_ant_name_tuple = extract_holog_params["ref_ant_per_map_ant_name_tuple"]

    if len(ref_ant_per_map_ant_tuple) != len(map_ant_tuple):
        logger.error(
            "Reference antenna per mapping antenna list and mapping antenna list should have same length."
        )
        raise Exception(
            "Inconsistancy between antenna list length, see error above for more info."
        )

    sel_state_ids = extract_holog_params["sel_state_ids"]
    holog_name = extract_holog_params["holog_name"]

    chan_freq = extract_holog_params["chan_setup"]["chan_freq"]
    pol = extract_holog_params["pol_setup"]["pol"]

    table_obj = ctables.table(
        ms_name, readonly=True, lockoptions={"option": "usernoread"}, ack=False
    )
    scans = [int(scan) for scan in scans]
    if sel_state_ids:
        ctb = ctables.taql(
            "select %s, SCAN_NUMBER, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG, FIELD_ID from "
            "$table_obj WHERE DATA_DESC_ID == %s AND SCAN_NUMBER in %s AND STATE_ID in %s"
            % (data_column, ddi, scans, list(sel_state_ids))
        )
    else:
        ctb = ctables.taql(
            "select %s, SCAN_NUMBER, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG, FIELD_ID from "
            "$table_obj WHERE DATA_DESC_ID == %s AND SCAN_NUMBER in %s"
            % (data_column, ddi, scans)
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
        map_ant_tuple, ref_ant_per_map_ant_tuple, ant_names, ant_station
    )

    (
        time_vis,
        vis_map_dict,
        weight_map_dict,
        flagged_mapping_antennas,
        used_samples_dict,
    ) = _extract_holog_chunk_jit(
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

    del vis_data, weight, ant1, ant2, time_vis_row, flag, flag_row, field_ids

    map_ant_name_list = list(map(str, map_ant_name_tuple))

    map_ant_name_list = ["_".join(("ant", i)) for i in map_ant_name_list]

    pnt_ant_dict = load_point_file(pnt_name, map_ant_name_list, dask_load=False)
    pnt_map_dict = _extract_pointing_chunk(map_ant_name_list, time_vis, pnt_ant_dict)

    grid_params = {}

    # The loop has been moved out of the function here making the gridding parameter auto-calculation
    # function more general use (hopefully). I honestly couldn't see a reason to keep it inside.
    for ant_index in vis_map_dict.keys():
        antenna_name = "_".join(("ant", ant_names[ant_index]))
        n_pix, cell_size = calculate_optimal_grid_parameters(
            pnt_map_dict,
            antenna_name,
            Telescope(gen_info["telescope name"]).diam,
            chan_freq,
            ddi,
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
        holog_map_key,
        ddi,
        ms_name,
        ant_names,
        ant_station,
        grid_params,
        time_interval,
        gen_info,
        map_ref_dict,
    )

    logger.info(
        "Finished extracting holography chunk for ddi: {ddi} holog_map_key: {holog_map_key}".format(
            ddi=ddi, holog_map_key=holog_map_key
        )
    )


def _get_map_ref_dict(map_ant_tuple, ref_ant_per_map_ant_tuple, ant_names, ant_station):
    map_dict = {}
    for ii, map_id in enumerate(map_ant_tuple):
        map_name = ant_names[map_id]
        ref_list = []
        for ref_id in ref_ant_per_map_ant_tuple[ii]:
            ref_list.append(f"{ant_names[ref_id]} @ {ant_station[ref_id]}")
        map_dict[map_name] = ref_list
    return map_dict


@njit(cache=False, nogil=True)
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
    return time_samples


@njit(cache=False, nogil=True)
def _extract_holog_chunk_jit(
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
    """JIT compiled function to extract relevant visibilty data from chunk after flagging and applying weights.

    Args:
        vis_data (numpy.ndarray): Visibility data (row, channel, polarization)
        weight (numpy.ndarray): Data weight values (row, polarization)
        ant1 (numpy.ndarray): List of antenna_ids for antenna1
        ant2 (numpy.ndarray): List of antenna_ids for antenna2
        time_vis_row (numpy.ndarray): Array of full time talues by row
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

    time_samples = _get_time_intervals(time_vis_row, scan_list, time_interval)
    n_time = len(time_samples)

    n_row, n_chan, n_pol = vis_data.shape

    half_int = time_interval / 2

    vis_map_dict = {}
    sum_weight_map_dict = {}
    used_samples_dict = {}

    for antenna_id in map_ant_tuple:
        vis_map_dict[antenna_id] = np.zeros(
            (n_time, n_chan, n_pol),
            dtype=types.complex128,
        )
        sum_weight_map_dict[antenna_id] = np.zeros(
            (n_time, n_chan, n_pol),
            dtype=types.float64,
        )
        used_samples_dict[antenna_id] = np.full(n_time, False, dtype=bool)

    time_index = 0
    for row in range(n_row):
        if flag_row is False:
            continue

        # Find index of time_vis_row[row] in time_samples, assumes time_vis_row is ordered in time

        if time_vis_row[row] < time_samples[time_index] - half_int:
            continue
        else:
            time_index = _get_time_index(
                time_vis_row[row], time_index, time_samples, half_int
            )
        if time_index < 0:
            break

        ant1_id = ant1[row]
        ant2_id = ant2[row]

        if ant1_id in map_ant_tuple:
            indx = map_ant_tuple.index(ant1_id)
            conjugate = False
            ref_ant_id = ant2_id
            map_ant_id = ant1_id  # mapping antenna index
        elif ant2_id in map_ant_tuple:
            indx = map_ant_tuple.index(ant2_id)
            conjugate = True
            ref_ant_id = ant1_id
            map_ant_id = ant2_id  # mapping antenna index
        else:
            continue

        if ref_ant_id in ref_ant_per_map_ant_tuple[indx]:
            if conjugate:
                vis_baseline = np.conjugate(vis_data[row, :, :])
            else:
                vis_baseline = vis_data[row, :, :]  # n_chan x n_pol
        else:
            continue

        for chan in range(n_chan):
            for pol in range(n_pol):
                if ~(flag[row, chan, pol]):
                    # Calculate running weighted sum of visibilities
                    used_samples_dict[map_ant_id][time_index] = True
                    vis_map_dict[map_ant_id][time_index, chan, pol] = (
                        vis_map_dict[map_ant_id][time_index, chan, pol]
                        + vis_baseline[chan, pol] * weight[row, pol]
                    )

                    # Calculate running sum of weights
                    sum_weight_map_dict[map_ant_id][time_index, chan, pol] = (
                        sum_weight_map_dict[map_ant_id][time_index, chan, pol]
                        + weight[row, pol]
                    )

    flagged_mapping_antennas = []

    for map_ant_id in vis_map_dict.keys():
        sum_of_sum_weight = 0

        for time_index in range(n_time):
            for chan in range(n_chan):
                for pol in range(n_pol):
                    sum_weight = sum_weight_map_dict[map_ant_id][time_index, chan, pol]
                    sum_of_sum_weight = sum_of_sum_weight + sum_weight
                    if sum_weight == 0:
                        vis_map_dict[map_ant_id][time_index, chan, pol] = 0.0
                    else:
                        vis_map_dict[map_ant_id][time_index, chan, pol] = (
                            vis_map_dict[map_ant_id][time_index, chan, pol] / sum_weight
                        )

        if sum_of_sum_weight == 0:
            flagged_mapping_antennas.append(map_ant_id)

    return (
        time_samples,
        vis_map_dict,
        sum_weight_map_dict,
        flagged_mapping_antennas,
        used_samples_dict,
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
    holog_map_key,
    ddi,
    ms_name,
    ant_names,
    ant_station,
    grid_params,
    time_interval,
    gen_info,
    map_ref_dict,
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
        holog_map_key(string): holog map id string
        ddi (numpy.ndarray): data description id; a combination of polarization and spectral window
    """

    ctb = ctables.table("/".join((ms_name, "ANTENNA")), ack=False)
    observing_location = ctb.getcol("POSITION")
    ctb.close()

    for map_ant_index in vis_map_dict.keys():
        if map_ant_index not in flagged_mapping_antennas:
            valid_data = used_samples_dict[map_ant_index] == 1.0

            ant_time_vis = time_vis[valid_data]

            time_vis_days = ant_time_vis / (3600 * 24)
            astro_time_vis = astropy.time.Time(time_vis_days, format="mjd")
            time_samples, indicies = _get_time_samples(astro_time_vis)
            coords = {"time": ant_time_vis, "chan": chan, "pol": pol}
            map_ant_tag = (
                "ant_" + ant_names[map_ant_index]
            )  # 'ant_' + str(map_ant_index)

            direction = np.take(
                pnt_map_dict[map_ant_tag]["DIRECTIONAL_COSINES"].values,
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
                vis_map_dict[map_ant_index][valid_data, ...],
                dims=["time", "chan", "pol"],
            )

            xds["WEIGHT"] = xr.DataArray(
                weight_map_dict[map_ant_index][valid_data, ...],
                dims=["time", "chan", "pol"],
            )

            xds["DIRECTIONAL_COSINES"] = xr.DataArray(
                pnt_map_dict[map_ant_tag]["DIRECTIONAL_COSINES"].values[
                    valid_data, ...
                ],
                dims=["time", "lm"],
            )

            xds["IDEAL_DIRECTIONAL_COSINES"] = xr.DataArray(
                pnt_map_dict[map_ant_tag]["POINTING_OFFSET"].values[valid_data, ...],
                dims=["time", "lm"],
            )

            xds.attrs["holog_map_key"] = holog_map_key
            xds.attrs["ddi"] = ddi
            xds.attrs["parallactic_samples"] = parallactic_samples
            xds.attrs["time_smoothing_interval"] = time_interval

            xds.attrs["summary"] = _crate_observation_summary(
                ant_names[map_ant_index],
                ant_station[map_ant_index],
                gen_info,
                grid_params,
                xds["DIRECTIONAL_COSINES"].values,
                chan,
                pnt_map_dict[map_ant_tag],
                valid_data,
                map_ref_dict,
            )

            holog_file = holog_name

            logger.debug(
                f"Writing {create_dataset_label(ant_names[map_ant_index], ddi)} holog file to {holog_file}"
            )
            xds.to_zarr(
                os.path.join(
                    holog_file,
                    "ddi_"
                    + str(ddi)
                    + "/"
                    + str(holog_map_key)
                    + "/"
                    + "ant_"
                    + str(ant_names[map_ant_index]),
                ),
                mode="w",
                compute=True,
                consolidated=True,
            )

        else:
            logger.warning(
                "Mapping antenna {index} has no data".format(
                    index=ant_names[map_ant_index]
                )
            )


def create_holog_obs_dict(
    pnt_dict,
    baseline_average_distance,
    baseline_average_nearest,
    ant_names,
    ant_pos,
    ant_names_main,
    exclude_antennas=None,
    write_distance_matrix=False,
):
    """
    Generate holog_obs_dict.
    """

    import pandas as pd
    from scipy.spatial import distance_matrix

    mapping_scans_dict = {}
    holog_obs_dict = {}
    map_id = 0
    ant_names_set = set()

    if exclude_antennas is None:
        exclude_antennas = []
    elif isinstance(exclude_antennas, str):
        exclude_antennas = [exclude_antennas]
    else:
        pass

    for ant_name in exclude_antennas:
        prefixed = "ant_" + ant_name
        if prefixed not in pnt_dict.keys():
            logger.warning(
                f"Bad reference antenna {ant_name} is not present in the data."
            )

    # Generate {ddi: {map: {scan:[i ...], ant:{ant_map_0:[], ...}}}} structure. No reference antennas are added
    # because we first need to populate all mapping antennas.
    for ant_name, ant_ds in pnt_dict.items():
        if "ant" in ant_name:
            ant_name = ant_name.replace("ant_", "")
            if ant_name in exclude_antennas:
                pass
            else:
                if ant_name in ant_names_main:  # Check if antenna in main table.
                    ant_names_set.add(ant_name)
                    for ddi, map_dict in ant_ds.attrs["mapping_scans_obs_dict"][
                        0
                    ].items():
                        if ddi not in holog_obs_dict:
                            holog_obs_dict[ddi] = {}
                        for ant_map_id, scan_list in map_dict.items():
                            if scan_list:
                                map_key = _check_if_array_in_dict(
                                    mapping_scans_dict, scan_list
                                )
                                if not map_key:
                                    map_key = "map_" + str(map_id)
                                    mapping_scans_dict[map_key] = scan_list
                                    map_id = map_id + 1

                                if map_key not in holog_obs_dict[ddi]:
                                    holog_obs_dict[ddi][map_key] = {
                                        "scans": np.array(scan_list),
                                        "ant": {},
                                    }

                                holog_obs_dict[ddi][map_key]["ant"][ant_name] = []

    df = pd.DataFrame(ant_pos, columns=["x", "y", "z"], index=ant_names)
    df_mat = pd.DataFrame(
        distance_matrix(df.values, df.values), index=df.index, columns=df.index
    )
    logger.debug("".join(("\n", str(df_mat))))

    if write_distance_matrix:
        df_mat.to_csv(
            path_or_buf="{base}/.baseline_distance_matrix.csv".format(base=os.getcwd()),
            sep="\t",
        )
        logger.info(
            "Writing distance matrix to {base}/.baseline_distance_matrix.csv ...".format(
                base=os.getcwd()
            )
        )

    if (baseline_average_distance != "all") and (baseline_average_nearest != "all"):
        logger.error(
            "baseline_average_distance and baseline_average_nearest can not both be specified."
        )

        raise Exception("Too many baseline parameters specified.")

    # The reference antennas are then given by ref_ant_set = ant_names_set - map_ant_set.
    for ddi, ddi_dict in holog_obs_dict.items():
        for map_id, map_dict in ddi_dict.items():
            map_ant_set = set(map_dict["ant"].keys())

            # Need a copy because of del holog_obs_dict[ddi][map_id]['ant'][map_ant_key] below.
            map_ant_keys = list(map_dict["ant"].keys())

            for map_ant_key in map_ant_keys:
                ref_ant_set = ant_names_set - map_ant_set

                # Select reference antennas by distance from mapping antenna
                if baseline_average_distance != "all":
                    sub_ref_ant_set = []

                    for ref_ant in ref_ant_set:
                        if df_mat.loc[map_ant_key, ref_ant] < baseline_average_distance:
                            sub_ref_ant_set.append(ref_ant)

                    if (not sub_ref_ant_set) and ref_ant_set:
                        logger.warning(
                            "DDI "
                            + str(ddi)
                            + " and mapping antenna "
                            + str(map_ant_key)
                            + " has no reference antennas. If baseline_average_distance was specified "
                            "increase this distance. See antenna distance matrix in log by setting "
                            "debug level to DEBUG in client function."
                        )

                    ref_ant_set = sub_ref_ant_set

                # Select reference antennas by the n-closest antennas
                if baseline_average_nearest != "all":
                    sub_ref_ant_set = []
                    nearest_ant_list = (
                        df_mat.loc[map_ant_key, :]
                        .loc[list(ref_ant_set)]
                        .sort_values()
                        .index.tolist()[0:baseline_average_nearest]
                    )

                    logger.debug(nearest_ant_list)
                    for ref_ant in ref_ant_set:
                        if ref_ant in nearest_ant_list:
                            sub_ref_ant_set.append(ref_ant)

                    ref_ant_set = sub_ref_ant_set
                ##################################################

                if ref_ant_set:
                    holog_obs_dict[ddi][map_id]["ant"][map_ant_key] = np.array(
                        list(ref_ant_set)
                    )
                else:
                    del holog_obs_dict[ddi][map_id]["ant"][
                        map_ant_key
                    ]  # Don't want mapping antennas with no reference antennas.
                    logger.warning(
                        "DDI "
                        + str(ddi)
                        + " and mapping antenna "
                        + str(map_ant_key)
                        + " has no reference antennas."
                    )

    return holog_obs_dict


def _check_if_array_in_dict(array_dict, array):
    for key, val in array_dict.items():
        if np.array_equiv(val, array):
            return key
    return False


def _extract_pointing_chunk(map_ant_ids, time_vis, pnt_ant_dict):
    """Averages pointing within the time sampling of the visibilities

    Args:
        map_ant_ids (list): list of antenna ids
        time_vis (numpy.ndarray): sorted, unique list of visibility times
        pnt_ant_dict (dict): map of pointing directional cosines with a map key based on the antenna id and indexed by
                             the MAIN table visibility time.

    Returns:
        dict:  Dictionary of directional cosine data mapped to nearest MAIN table sample times.
    """

    pnt_map_dict = {}
    coords = {"time": time_vis}
    for antenna in map_ant_ids:
        pnt_xds = pnt_ant_dict[antenna]
        pnt_time = pnt_xds.time.values
        pnt_int = np.average(np.diff(pnt_time))
        vis_int = time_vis[1] - time_vis[0]

        if pnt_int < vis_int:
            avg_dir, avg_dir_cos, avg_enc, avg_pnt_off, avg_tgt = (
                _time_avg_pointing_jit(
                    time_vis,
                    pnt_xds.time.values,
                    pnt_xds["DIRECTION"].values,
                    pnt_xds["DIRECTIONAL_COSINES"].values,
                    pnt_xds["ENCODER"].values,
                    pnt_xds["POINTING_OFFSET"].values,
                    pnt_xds["TARGET"].values,
                )
            )
        else:
            avg_dir, avg_dir_cos, avg_enc, avg_pnt_off, avg_tgt = _interpolate_pointing(
                time_vis,
                pnt_xds.time.values,
                pnt_xds["DIRECTION"].values,
                pnt_xds["DIRECTIONAL_COSINES"].values,
                pnt_xds["ENCODER"].values,
                pnt_xds["POINTING_OFFSET"].values,
                pnt_xds["TARGET"].values,
            )

        new_pnt_xds = xr.Dataset()
        new_pnt_xds.assign_coords(coords)

        new_pnt_xds["DIRECTION"] = xr.DataArray(avg_dir, dims=("time", "az_el"))
        new_pnt_xds["DIRECTIONAL_COSINES"] = xr.DataArray(
            avg_dir_cos, dims=("time", "az_el")
        )
        new_pnt_xds["ENCODER"] = xr.DataArray(avg_enc, dims=("time", "az_el"))
        new_pnt_xds["POINTING_OFFSET"] = xr.DataArray(
            avg_pnt_off, dims=("time", "az_el")
        )
        new_pnt_xds["TARGET"] = xr.DataArray(avg_tgt, dims=("time", "az_el"))
        new_pnt_xds.attrs = pnt_xds.attrs
        pnt_map_dict[antenna] = new_pnt_xds
    return pnt_map_dict


@njit(cache=False, nogil=True)
def _time_avg_pointing_jit(time_vis, pnt_time, dire, dir_cos, enc, pnt_off, tgt):
    half_int = (time_vis[1] - time_vis[0]) / 2
    n_samples = time_vis.shape[0]
    the_shape = (n_samples, 2)
    n_row = pnt_time.shape[0]

    avg_dir = np.zeros(the_shape)
    avg_dir_cos = np.zeros(the_shape)
    avg_enc = np.zeros(the_shape)
    avg_pnt_off = np.zeros(the_shape)
    avg_tgt = np.zeros(the_shape)
    avg_wgt = np.zeros(the_shape)

    i_time = 0
    for i_row in range(n_row):
        if pnt_time[i_row] < time_vis[i_time] - half_int:
            continue
        else:
            i_time = _get_time_index(pnt_time[i_row], i_time, time_vis, half_int)
            if i_time < 0:
                break

        avg_dir[i_time] += dire[i_row]
        avg_dir_cos[i_time] += dir_cos[i_row]
        avg_enc[i_time] += enc[i_row]
        avg_pnt_off[i_time] += pnt_off[i_row]
        avg_tgt[i_time] += tgt[i_row]
        avg_wgt[i_time] += 1

    avg_dir /= avg_wgt
    avg_dir_cos /= avg_wgt
    avg_enc /= avg_wgt
    avg_pnt_off /= avg_wgt
    avg_tgt /= avg_wgt

    return avg_dir, avg_dir_cos, avg_enc, avg_pnt_off, avg_tgt


@njit(cache=False, nogil=True)
def _interpolate_pointing(time_vis, pnt_time, dire, dir_cos, enc, pnt_off, tgt):
    n_samples = time_vis.shape[0]
    the_shape = (n_samples, 2)
    n_row = pnt_time.shape[0]
    pnt_int = np.average(np.diff(pnt_time))
    half_int = (time_vis[1] - time_vis[0]) / 2

    avg_dir = np.zeros(the_shape)
    avg_dir_cos = np.zeros(the_shape)
    avg_enc = np.zeros(the_shape)
    avg_pnt_off = np.zeros(the_shape)
    avg_tgt = np.zeros(the_shape)
    avg_wgt = np.zeros(the_shape)

    for i_time in range(n_samples):
        i_row = int(np.floor((time_vis[i_time] - half_int - pnt_time[0]) / pnt_int))

        avg_dir[i_time] += dire[i_row]
        avg_dir_cos[i_time] += dir_cos[i_row]
        avg_enc[i_time] += enc[i_row]
        avg_pnt_off[i_time] += pnt_off[i_row]
        avg_tgt[i_time] += tgt[i_row]
        avg_wgt[i_time] += 1

    return avg_dir, avg_dir_cos, avg_enc, avg_pnt_off, avg_tgt


@njit(cache=False, nogil=True)
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


def _crate_observation_summary(
    antenna_name,
    station,
    obs_info,
    grid_params,
    lm,
    chan_axis,
    pnt_map_xds,
    valid_data,
    map_ref_dict,
):
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


def create_holog_json(holog_file, holog_dict):
    """Save holog file meta information to json file with the transformation
        of the ordering (ddi, holog_map, ant) --> (ant, ddi, holog_map).

    Args:
        input_params ():
        holog_file (str): holog file name.
        holog_dict (dict): Dictionary containing msdx data.
    """

    ant_holog_dict = {}

    for ddi, map_dict in holog_dict.items():
        if "ddi_" in ddi:
            for mapping, ant_dict in map_dict.items():
                if "map_" in mapping:
                    for ant, xds in ant_dict.items():
                        if "ant_" in ant:
                            if ant not in ant_holog_dict:
                                ant_holog_dict[ant] = {ddi: {mapping: {}}}
                            elif ddi not in ant_holog_dict[ant]:
                                ant_holog_dict[ant][ddi] = {mapping: {}}

                            ant_holog_dict[ant][ddi][mapping] = xds.to_dict(data=False)

    output_meta_file = "{name}/{ext}".format(name=holog_file, ext=".holog_json")

    try:
        with open(output_meta_file, "w") as json_file:
            json.dump(ant_holog_dict, json_file)

    except Exception as error:
        logger.error(f"{error}")

        raise Exception(error)
