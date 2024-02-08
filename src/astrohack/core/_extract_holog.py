import os
import json
import numpy as np
import xarray as xr
import astropy
import astrohack
import graphviper.utils.logger as logger

from numba import njit
from numba.core import types
from datetime import date

from casacore import tables as ctables
from astrohack._utils._imaging import _calculate_parallactic_angle_chunk
from astrohack._utils._algorithms import calculate_optimal_grid_parameters, _significant_digits
from astrohack._utils._plot_commons import _create_figure_and_axes, _scatter_plot, _close_figure
from astrohack._utils._conversion import _convert_unit

from astrohack._utils._dio import _load_point_file


def _extract_holog_chunk(extract_holog_params):
    """Perform data query on holography data chunk and get unique time and state_ids/

    Args:
        ms_name (str): Measurementset name
        data_column (str): Data column to extract.
        ddi (int): Data description id
        scan (int): Scan number
        map_ant_ids (numpy.narray): Array of antenna_id values corresponding to mapping data.
        ref_ant_ids (numpy.narray): Arry of antenna_id values corresponding to reference data.
        sel_state_ids (list): List pf state_ids corresponding to holography data/
    """

    ms_name = extract_holog_params["ms_name"]
    pnt_name = extract_holog_params["point_name"]
    data_column = extract_holog_params["data_column"]
    ddi = extract_holog_params["ddi"]
    scans = extract_holog_params["scans"]
    ant_names = extract_holog_params["ant_names"]
    ref_ant_per_map_ant_tuple = extract_holog_params["ref_ant_per_map_ant_tuple"]
    map_ant_tuple = extract_holog_params["map_ant_tuple"]
    map_ant_name_tuple = extract_holog_params["map_ant_name_tuple"]
    holog_map_key = extract_holog_params["holog_map_key"]
    time_interval = extract_holog_params['time_smoothing_interval']
    telescope_name = extract_holog_params["telescope_name"]

    # This piece of information is no longer used leaving them here commented out for completeness
    # ref_ant_per_map_ant_name_tuple = extract_holog_params["ref_ant_per_map_ant_name_tuple"]

    if len(ref_ant_per_map_ant_tuple) != len(map_ant_tuple):
        logger.error("Reference antenna per mapping antenna list and mapping antenna list should have same length.")
        raise Exception("Inconsistancy between antenna list length, see error above for more info.")

    sel_state_ids = extract_holog_params["sel_state_ids"]
    holog_name = extract_holog_params["holog_name"]

    chan_freq = extract_holog_params["chan_setup"]["chan_freq"]
    pol = extract_holog_params["pol_setup"]["pol"]
    table_obj = ctables.table(ms_name, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)

    if sel_state_ids:
        ctb = ctables.taql(
            "select %s, SCAN_NUMBER, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG from $table_obj "
            "WHERE DATA_DESC_ID == %s AND SCAN_NUMBER in %s AND STATE_ID in %s"
            % (data_column, ddi, list(scans), list(sel_state_ids))
        )
    else:
        ctb = ctables.taql(
            "select %s, SCAN_NUMBER, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG from $table_obj "
            "WHERE DATA_DESC_ID == %s AND SCAN_NUMBER in %s"
            % (data_column, ddi, list(scans))
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

    # Here we use the median of the differences between dumps as this is a good proxy for the integration time
    if time_interval is None:
        time_interval = np.median(np.diff(np.unique(time_vis_row)))

    ctb.close()
    table_obj.close()

    time_vis, vis_map_dict, weight_map_dict, flagged_mapping_antennas, used_samples_dict = _extract_holog_chunk_jit(
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
        scan_list
    )

    del vis_data, weight, ant1, ant2, time_vis_row, flag, flag_row

    map_ant_name_list = list(map(str, map_ant_name_tuple))

    map_ant_name_list = ["_".join(("ant", i)) for i in map_ant_name_list]

    pnt_ant_dict = _load_point_file(pnt_name, map_ant_name_list, dask_load=False)
    pnt_map_dict = _extract_pointing_chunk(map_ant_name_list, time_vis, pnt_ant_dict)

    grid_params = {}

    # The loop has been moved out of the function here making the gridding parameter auto-calculation
    # function more general use (hopefully). I honestly couldn't see a reason to keep it inside.
    for ant_index in vis_map_dict.keys():
        antenna_name = "_".join(("ant", ant_names[ant_index]))
        n_pix, cell_size = calculate_optimal_grid_parameters(pnt_map_dict, antenna_name, telescope_name, chan_freq)

        grid_params[antenna_name] = {
            "n_pix": n_pix,
            "cell_size": cell_size
        }

    # ## To DO: ################## Average multiple repeated samples over_flow_protector_constant = float("%.5g" %
    # time_vis[0])  # For example 5076846059.4 -> 5076800000.0 time_vis = time_vis - over_flow_protector_constant
    # from astrohack._utils._algorithms import _average_repeated_pointings time_vis = _average_repeated_pointings(
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
        grid_params,
        time_interval
    )

    logger.info("Finished extracting holography chunk for ddi: {ddi} holog_map_key: {holog_map_key}".format(
        ddi=ddi, holog_map_key=holog_map_key)
    )


@njit(cache=False, nogil=True)
def _get_time_intervals(time_vis_row, scan_list, time_interval):
    unq_scans = np.unique(scan_list)
    scan_time_ranges = []
    for scan in unq_scans:
        selected_times = time_vis_row[scan_list == scan]
        min_time, max_time = np.min(selected_times), np.max(selected_times)
        scan_time_ranges.append([min_time, max_time])

    half_int = time_interval/2
    start = np.min(time_vis_row)+half_int
    total_time = np.max(time_vis_row)-start
    n_time = int(np.ceil(total_time/time_interval))+1
    stop = start + n_time*time_interval
    raw_time_samples = np.linspace(start, stop, n_time+1)

    filtered_time_samples = []
    for time_sample in raw_time_samples:
        for time_range in scan_time_ranges:
            if time_range[0] <= time_sample <= time_range[1]:
                filtered_time_samples.append(time_sample)
                break
    return np.array(filtered_time_samples)


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
        scan_list
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

    half_int = time_interval/2

    vis_map_dict = {}
    sum_weight_map_dict = {}
    used_samples_dict = {}

    for antenna_id in map_ant_tuple:
        vis_map_dict[antenna_id] = np.zeros(
            (n_time, n_chan, n_pol), dtype=types.complex64
        )
        sum_weight_map_dict[antenna_id] = np.zeros(
            (n_time, n_chan, n_pol), dtype=types.float64
        )
        used_samples_dict[antenna_id] = np.full(n_time, False, dtype=bool)

    time_index = 0
    for row in range(n_row):
        if flag_row is False:
            continue
        # Find index of time_vis_row[row] in time_samples, assumes time_vis_row is ordered in time
        if time_vis_row[row] < time_samples[time_index] - half_int:
            continue
        elif time_vis_row[row] > time_samples[time_index] + half_int:
            time_index += 1
        if time_index == n_time:
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
                    sum_weight = sum_weight_map_dict[map_ant_id][
                        time_index, chan, pol
                    ]
                    sum_of_sum_weight = sum_of_sum_weight + sum_weight
                    if sum_weight == 0:
                        vis_map_dict[map_ant_id][time_index, chan, pol] = 0.0
                    else:
                        vis_map_dict[map_ant_id][time_index, chan, pol] = (
                                vis_map_dict[map_ant_id][time_index, chan, pol]
                                / sum_weight
                        )

        if sum_of_sum_weight == 0:
            flagged_mapping_antennas.append(map_ant_id)

    return time_samples, vis_map_dict, sum_weight_map_dict, flagged_mapping_antennas, used_samples_dict


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
        grid_params,
        time_interval
):
    """Create holog-structured, formatted output file and save to zarr.

    Args:
        holog_name (str): holog file name.
        vis_map_dict (dict): a nested dictionary/map of weighted visibilities indexed as [antenna][time, chan, pol]; mainains time ordering.
        weight_map_dict (dict): weights dictionary/map for visibilites in vis_map_dict
        pnt_map_dict (dict): pointing table map dictionary
        time_vis (numpy.ndarray): time_vis values
        chan (numpy.ndarray): channel values
        pol (numpy.ndarray): polarization values
        flagged_mapping_antennas (numpy.ndarray): list of mapping antennas that have been flagged.
        holog_map_key(string): holog map id string
        ddi (numpy.ndarray): data description id; a combination of polarization and spectral window
    """

    ctb = ctables.table("/".join((ms_name, "ANTENNA")))
    observing_location = ctb.getcol("POSITION")

    ctb = ctables.table("/".join((ms_name, "OBSERVATION")))
    telescope_name = ctb.getcol("TELESCOPE_NAME")[0]

    ctb.close()

    for map_ant_index in vis_map_dict.keys():
        if map_ant_index not in flagged_mapping_antennas:
            valid_data = used_samples_dict[map_ant_index] == 1.

            ant_time_vis = time_vis[valid_data]

            time_vis_days = ant_time_vis / (3600 * 24)
            astro_time_vis = astropy.time.Time(time_vis_days, format="mjd")
            time_samples, indicies = _get_time_samples(astro_time_vis)
            coords = {"time": ant_time_vis, "chan": chan, "pol": pol}
            map_ant_tag = 'ant_' + ant_names[map_ant_index]  # 'ant_' + str(map_ant_index)

            direction = np.take(pnt_map_dict[map_ant_tag]["DIRECTIONAL_COSINES"].values, indicies, axis=0)

            parallactic_samples = _calculate_parallactic_angle_chunk(
                time_samples=time_samples,
                observing_location=observing_location[map_ant_index],
                direction=direction
            )

            xds = xr.Dataset()
            xds = xds.assign_coords(coords)
            xds["VIS"] = xr.DataArray(
                vis_map_dict[map_ant_index][valid_data, ...], dims=["time", "chan", "pol"]
            )

            xds["WEIGHT"] = xr.DataArray(
                weight_map_dict[map_ant_index][valid_data, ...], dims=["time", "chan", "pol"]
            )

            xds["DIRECTIONAL_COSINES"] = xr.DataArray(
                pnt_map_dict[map_ant_tag]["DIRECTIONAL_COSINES"].values[valid_data, ...], dims=["time", "lm"]
            )

            xds["IDEAL_DIRECTIONAL_COSINES"] = xr.DataArray(
                pnt_map_dict[map_ant_tag]["POINTING_OFFSET"].values[valid_data, ...], dims=["time", "lm"]
            )

            xds.attrs["holog_map_key"] = holog_map_key
            xds.attrs["ddi"] = ddi
            xds.attrs["parallactic_samples"] = parallactic_samples
            xds.attrs["telescope_name"] = telescope_name
            xds.attrs["antenna_name"] = ant_names[map_ant_index]

            xds.attrs["l_max"] = np.max(xds["DIRECTIONAL_COSINES"][:, 0].values)
            xds.attrs["l_min"] = np.min(xds["DIRECTIONAL_COSINES"][:, 0].values)
            xds.attrs["m_max"] = np.max(xds["DIRECTIONAL_COSINES"][:, 1].values)
            xds.attrs["m_min"] = np.min(xds["DIRECTIONAL_COSINES"][:, 1].values)

            xds.attrs["grid_params"] = grid_params[map_ant_tag]
            xds.attrs["time_smoothing_interval"] = time_interval

            holog_file = holog_name

            logger.info(
                "Writing holog file to {file}".format(file=holog_file)
            )
            xds.to_zarr(
                os.path.join(
                    holog_file,
                    'ddi_' + str(ddi) + "/" + str(holog_map_key) + "/" + "ant_" + str(ant_names[map_ant_index])
                ),
                mode="w",
                compute=True,
                consolidated=True,
            )

        else:
            logger.warning("Mapping antenna {index} has no data".format(index=ant_names[map_ant_index]))


def _create_holog_obs_dict(
        pnt_dict,
        baseline_average_distance,
        baseline_average_nearest,
        ant_names,
        ant_pos,
        ant_names_main,
        write_distance_matrix=False
):
    '''
    Generate holog_obs_dict.
    '''

    import pandas as pd
    from scipy.spatial import distance_matrix

    mapping_scans_dict = {}
    holog_obs_dict = {}
    map_id = 0
    ant_names_set = set()

    # Generate {ddi: {map: {scan:[i ...], ant:{ant_map_0:[], ...}}}} structure. No reference antenas are added
    # because we first need to populate all mapping antennas.
    for ant_name, ant_ds in pnt_dict.items():
        if 'ant' in ant_name:
            ant_name = ant_name.replace('ant_', '')
            if ant_name in ant_names_main:  # Check if antenna in main table.
                ant_names_set.add(ant_name)
                for ddi, map_dict in ant_ds.attrs['mapping_scans_obs_dict'][0].items():
                    if ddi not in holog_obs_dict:
                        holog_obs_dict[ddi] = {}
                    for ant_map_id, scan_list in map_dict.items():
                        if scan_list:
                            map_key = _check_if_array_in_dict(mapping_scans_dict, scan_list)
                            if not map_key:
                                map_key = 'map_' + str(map_id)
                                mapping_scans_dict[map_key] = scan_list
                                map_id = map_id + 1

                            if map_key not in holog_obs_dict[ddi]:
                                holog_obs_dict[ddi][map_key] = {'scans': np.array(scan_list), 'ant': {}}

                            holog_obs_dict[ddi][map_key]['ant'][ant_name] = []

    df = pd.DataFrame(ant_pos, columns=['x', 'y', 'z'], index=ant_names)
    df_mat = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
    logger.debug("".join(("\n", str(df_mat))))

    if write_distance_matrix:
        df_mat.to_csv(path_or_buf="{base}/.baseline_distance_matrix.csv".format(base=os.getcwd()), sep="\t")
        logger.info("Writing distance matrix to {base}/.baseline_distance_matrix.csv ...".format(base=os.getcwd()))

    if (baseline_average_distance != 'all') and (baseline_average_nearest != 'all'):
        logger.error('baseline_average_distance and baseline_average_nearest can not both be specified.')

        raise Exception("Too many baseline parameters specified.")

    # The reference antennas are then given by ref_ant_set = ant_names_set - map_ant_set.
    for ddi, ddi_dict in holog_obs_dict.items():
        for map_id, map_dict in ddi_dict.items():
            map_ant_set = set(map_dict['ant'].keys())

            # Need a copy because of del holog_obs_dict[ddi][map_id]['ant'][map_ant_key] below.
            map_ant_keys = list(map_dict['ant'].keys())

            for map_ant_key in map_ant_keys:
                ref_ant_set = ant_names_set - map_ant_set

                # Select reference antennas by distance from mapping antenna
                if baseline_average_distance != 'all':
                    sub_ref_ant_set = []

                    for ref_ant in ref_ant_set:
                        if df_mat.loc[map_ant_key, ref_ant] < baseline_average_distance:
                            sub_ref_ant_set.append(ref_ant)

                    if (not sub_ref_ant_set) and ref_ant_set:
                        logger.warning('DDI ' + str(ddi) + ' and mapping antenna ' + str(
                            map_ant_key) + ' has no reference antennas. If baseline_average_distance was specified '
                                           'increase this distance. See antenna distance matrix in log by setting '
                                           'debug level to DEBUG in client function.')

                    ref_ant_set = sub_ref_ant_set

                # Select reference antennas by the n-closest antennas
                if baseline_average_nearest != 'all':
                    sub_ref_ant_set = []
                    nearest_ant_list = df_mat.loc[map_ant_key, :].loc[list(ref_ant_set)].sort_values().index.tolist()[
                                       0:baseline_average_nearest]

                    logger.debug(nearest_ant_list)
                    for ref_ant in ref_ant_set:
                        if ref_ant in nearest_ant_list:
                            sub_ref_ant_set.append(ref_ant)

                    ref_ant_set = sub_ref_ant_set
                ##################################################

                if ref_ant_set:
                    holog_obs_dict[ddi][map_id]['ant'][map_ant_key] = np.array(list(ref_ant_set))
                else:
                    del holog_obs_dict[ddi][map_id]['ant'][
                        map_ant_key]  # Don't want mapping antennas with no reference antennas.
                    logger.warning(
                        'DDI ' + str(ddi) + ' and mapping antenna ' + str(map_ant_key) + ' has no reference antennas.')

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
        avg_dir, avg_dir_cos, avg_enc, avg_pnt_off, avg_tgt =\
            _time_avg_pointing_jit(time_vis,
                                   pnt_xds.time.values,
                                   pnt_xds['DIRECTION'].values,
                                   pnt_xds['DIRECTIONAL_COSINES'].values,
                                   pnt_xds['ENCODER'].values,
                                   pnt_xds['POINTING_OFFSET'].values,
                                   pnt_xds['TARGET'].values,
                                   )

        new_pnt_xds = xr.Dataset()
        new_pnt_xds.assign_coords(coords)

        new_pnt_xds["DIRECTION"] = xr.DataArray(avg_dir, dims=("time", "az_el"))
        new_pnt_xds["DIRECTIONAL_COSINES"] = xr.DataArray(avg_dir_cos, dims=("time", "az_el"))
        new_pnt_xds["ENCODER"] = xr.DataArray(avg_enc, dims=("time", "az_el"))
        new_pnt_xds["POINTING_OFFSET"] = xr.DataArray(avg_pnt_off, dims=("time", "az_el"))
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
        if pnt_time[i_row] > time_vis[i_time] + half_int:
            if i_time == n_samples-1:
                break
            else:
                i_time += 1
        elif pnt_time[i_row] < time_vis[i_time] - half_int:
            continue
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


def _create_holog_meta_data(holog_file, holog_dict, input_params):
    """Save holog file meta information to json file with the transformation
        of the ordering (ddi, holog_map, ant) --> (ant, ddi, holog_map).

    Args:
        holog_name (str): holog file name.
        holog_dict (dict): Dictionary containing msdx data.
    """

    ant_holog_dict = {}
    cell_sizes = []
    n_pixs = []
    telescope_names = []

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

                            cell_sizes.append(xds.attrs["grid_params"]["cell_size"])
                            n_pixs.append(xds.attrs["grid_params"]["n_pix"])
                            telescope_names.append(xds.attrs['telescope_name'])

    cell_sizes_sigfigs = _significant_digits(cell_sizes, digits=3)

    meta_data = {
        'cell_size': np.mean(cell_sizes),
        'n_pix': n_pixs[0],
        'telescope_name': telescope_names[0]
    }

    if not (len(set(cell_sizes_sigfigs)) == 1):
        logger.warning('Cell size not consistent: ' + str(cell_sizes))
        logger.warning('Calculating suggested cell size ...')

        meta_data["cell_size"] = \
            astrohack._utils._algorithms._calculate_suggested_grid_parameter(parameter=np.array(cell_sizes))

        logger.info("The suggested cell size is calculated to be: {cell_size}".format(cell_size=meta_data["cell_size"]))

    if not (len(set(n_pixs)) == 1):
        logger.warning('Number of pixels not consistent: ' + str(n_pixs))
        logger.warning('Calculating suggested number of pixels ...')

        meta_data['n_pix'] = int(
            astrohack._utils._algorithms._calculate_suggested_grid_parameter(parameter=np.array(n_pixs)))

        logger.info("The suggested number of pixels is calculated to be: {n_pix} (grid: {points} x {points})".format(
            n_pix=meta_data["n_pix"], points=int(np.sqrt(meta_data["n_pix"]))
        ))

    if not (len(set(telescope_names)) == 1):
        logger.error('Telescope name not consistent: ' + str(telescope_names))
        meta_data['telescope_name'] = None

    output_meta_file = "{name}/{ext}".format(name=holog_file, ext=".holog_json")

    try:
        with open(output_meta_file, "w") as json_file:
            json.dump(ant_holog_dict, json_file)

    except Exception as error:
        logger.error(f"{error}")

        raise Exception(error)

    meta_data.update(input_params)

    return meta_data


def _plot_lm_coverage(param_dict):
    data = param_dict['xds_data']
    angle_fact = _convert_unit('rad', param_dict['angle_unit'], 'trigonometric')
    real_lm = data['DIRECTIONAL_COSINES'] * angle_fact
    ideal_lm = data['IDEAL_DIRECTIONAL_COSINES'] * angle_fact
    time = data.time.values
    time -= time[0]
    time *= _convert_unit('sec', param_dict['time_unit'], 'time')
    param_dict['l_label'] = f'L [{param_dict["angle_unit"]}]'
    param_dict['m_label'] = f'M [{param_dict["angle_unit"]}]'
    param_dict['time_label'] = f'Time from observation start [{param_dict["time_unit"]}]'

    param_dict['marker'] = '.'
    param_dict['linestyle'] = '-'
    param_dict['color'] = 'blue'

    _plot_lm_coverage_sub(time, real_lm, ideal_lm, param_dict)

    if param_dict['plot_correlation'] is None or param_dict['plot_correlation'] == 'None':
        pass
    else:
        param_dict['linestyle'] = ''
        visi = np.average(data["VIS"].values, axis=1)
        weights = np.average(data["WEIGHT"].values, axis=1)
        pol_axis = data.pol.values
        if isinstance(param_dict['plot_correlation'], (list, tuple)):
            for correlation in param_dict['plot_correlation']:
                _plot_correlation(visi, weights, correlation, pol_axis, time, real_lm, param_dict)
        else:
            if param_dict['plot_correlation'] == 'all':
                for correlation in pol_axis:
                    _plot_correlation(visi, weights, correlation, pol_axis, time, real_lm, param_dict)
            else:
                _plot_correlation(visi, weights, param_dict['plot_correlation'], pol_axis, time, real_lm, param_dict)


def _plot_correlation(visi, weights, correlation, pol_axis, time, lm, param_dict):
    if correlation in pol_axis:
        ipol = pol_axis == correlation
        loc_vis = visi[:, ipol]
        loc_wei = weights[:, ipol]
        if param_dict['complex_split'] == 'polar':
            y_data = [np.absolute(loc_vis)]
            y_label = [f'{correlation} Amplitude [arb. units]']
            title = ['Amplitude']
            y_data.append(np.angle(loc_vis) * _convert_unit('rad', param_dict["phase_unit"], 'trigonometric'))
            y_label.append(f'{correlation} Phase [{param_dict["phase_unit"]}]')
            title.append('Phase')
        else:
            y_data = [loc_vis.real]
            y_label = [f'Real {correlation} [arb. units]']
            title = ['real part']
            y_data.append(loc_vis.imag)
            y_label.append(f'Imaginary {correlation} [arb. units]')
            title.append('imaginary part')

        y_data.append(loc_wei)
        y_label.append(f'{correlation} weights [arb. units]')
        title.append('weights')

        fig, ax = _create_figure_and_axes(param_dict['figure_size'], [3, 3])
        for isplit in range(3):
            _scatter_plot(ax[isplit, 0], time, param_dict['time_label'], y_data[isplit], y_label[isplit],
                          f'Time vs {correlation} {title[isplit]}', data_marker=param_dict['marker'],
                          data_linestyle=param_dict['linestyle'], data_color=param_dict['color'])
            _scatter_plot(ax[isplit, 1], lm[:, 0], param_dict['l_label'], y_data[isplit], y_label[isplit],
                          f'L vs {correlation} {title[isplit]}', data_marker=param_dict['marker'],
                          data_linestyle=param_dict['linestyle'], data_color=param_dict['color'])
            _scatter_plot(ax[isplit, 2], lm[:, 1], param_dict['m_label'], y_data[isplit], y_label[isplit],
                          f'M vs {correlation} {title[isplit]}', data_marker=param_dict['marker'],
                          data_linestyle=param_dict['linestyle'], data_color=param_dict['color'])

        plotfile = (f'{param_dict["destination"]}/holog_directional_cosines_{correlation}_{param_dict["this_map"]}_'
                    f'{param_dict["this_ant"]}_{param_dict["this_ddi"]}.png')
        _close_figure(fig, f'Channel averaged {correlation} vs Directional Cosines', plotfile, param_dict['dpi'],
                      param_dict['display'])
    else:

        logger.warning(
            f'Correlation {correlation} is not present for {param_dict["this_ant"]} {param_dict["this_ddi"]} '
            f'{param_dict["this_map"]}, skipping...')
    return


def _plot_lm_coverage_sub(time, real_lm, ideal_lm, param_dict):
    fig, ax = _create_figure_and_axes(param_dict['figure_size'], [2, 2])
    _scatter_plot(ax[0, 0], time, param_dict['time_label'], real_lm[:, 0], param_dict['l_label'], 'Time vs Real L',
                  data_marker=param_dict['marker'], data_linestyle=param_dict['linestyle'], data_color=
                  param_dict['color'])
    _scatter_plot(ax[0, 1], time, param_dict['time_label'], real_lm[:, 1], param_dict['m_label'], 'Time vs Real M',
                  data_marker=param_dict['marker'], data_linestyle=param_dict['linestyle'], data_color=
                  param_dict['color'])
    _scatter_plot(ax[1, 0], real_lm[:, 0], param_dict['l_label'], real_lm[:, 1], param_dict['m_label'], 'Real L and M',
                  data_marker=param_dict['marker'], data_linestyle=param_dict['linestyle'], data_color=
                  param_dict['color'])
    _scatter_plot(ax[1, 1], ideal_lm[:, 0], param_dict['l_label'], ideal_lm[:, 1], param_dict['m_label'],
                  'Ideal L and M', data_marker=param_dict['marker'], data_linestyle=param_dict['linestyle'],
                  data_color=param_dict['color'])
    plotfile = f'{param_dict["destination"]}/holog_directional_cosines_{param_dict["this_map"]}_' \
               f'{param_dict["this_ant"]}_{param_dict["this_ddi"]}.png'
    _close_figure(fig, 'Directional Cosines', plotfile, param_dict['dpi'], param_dict['display'])


def _export_to_aips(param_dict):
    xds_data = param_dict['xds_data']
    stokes = 'I'
    stokes_vis = _compute_average_stokes_visibilities(xds_data, stokes)
    filename = f'{param_dict["destination"]}/holog_visibilities_{param_dict["this_map"]}_{param_dict["this_ant"]}_' \
               f'{param_dict["this_ddi"]}.txt'
    ant_num = xds_data.attrs['antenna_name'].split('a')[1]
    cmt = '#! '
    spc = 6 * ' '
    today = date.today().strftime("%y%m%d")
    outstr = cmt + f"RefAnt = ** Antenna = {ant_num} Stokes = '{stokes}_' Freq =  {stokes_vis.attrs['frequency']:.9f}" \
                   f" DATE-OBS = '{today}'\n"
    outstr += cmt + "MINsamp =   0  Npoint =   1\n"
    outstr += cmt + "IFnumber =   2   Channel =    32.0\n"
    outstr += cmt + "TimeRange = -99,  0,  0,  0,  999,  0,  0,  0\n"
    outstr += cmt + "Averaged Ref-Ants = 10, 15,\n"
    outstr += cmt + "DOCAL = T  DOPOL =-1\n"
    outstr += cmt + "BCHAN=     4 ECHAN=    60 CHINC=  1 averaged\n"
    outstr += cmt + "   LL             MM             AMPLITUDE      PHASE         SIGMA(AMP)   SIGMA(PHASE)\n"
    lm = xds_data['DIRECTIONAL_COSINES'].values
    amp = stokes_vis['AMPLITUDE'].values
    pha = stokes_vis['PHASE'].values
    sigma_amp = stokes_vis['SIGMA_AMP']
    sigma_pha = stokes_vis['SIGMA_PHA']
    for i_time in range(len(xds_data.time)):
        if np.isfinite(sigma_amp[i_time]):
            outstr += f"{lm[i_time, 0]:15.7f}{lm[i_time, 1]:15.7f}{amp[i_time]:15.7f}{pha[i_time]:15.7f}" \
                      f"{sigma_amp[i_time]:15.7f}{sigma_pha[i_time]:15.7f}\n"
    outstr += f"{cmt}Average number samples per point =   1.000"
    with open(filename, 'w') as outfile:
        outfile.write(outstr)
    return


def _compute_average_stokes_visibilities(vis, stokes):
    n_chan = len(vis.chan)
    chan_ave_vis = vis.mean(dim='chan', skipna=True)
    amp, pha, sigma_amp, sigma_pha = _compute_stokes(chan_ave_vis['VIS'].values, n_chan * chan_ave_vis['WEIGHT'].values,
                                                     chan_ave_vis.pol)
    coords = {'time': chan_ave_vis.time,
              'pol': ['I', 'Q', 'U', 'V']}
    xds = xr.Dataset()
    xds = xds.assign_coords(coords)
    xds["AMPLITUDE"] = xr.DataArray(amp, dims=["time", 'pol'], coords=coords)
    xds["PHASE"] = xr.DataArray(pha, dims=["time", 'pol'], coords=coords)
    xds['SIGMA_AMP'] = xr.DataArray(sigma_amp, dims=["time", 'pol'], coords=coords)
    xds['SIGMA_PHA'] = xr.DataArray(sigma_amp, dims=["time", 'pol'], coords=coords)
    xds.attrs['frequency'] = np.mean(vis.chan) / 1e9  # in GHz
    return xds.sel(pol=stokes)


def _compute_stokes(data, weight, pol_axis):
    stokes_data = np.zeros_like(data)
    weight[weight == 0] = np.nan
    sigma = np.sqrt(1 / weight)
    sigma_amp = np.zeros_like(weight)
    if 'RR' in pol_axis:
        stokes_data[:, 0] = (data[:, 0] + data[:, 3]) / 2
        sigma_amp[:, 0] = (sigma[:, 0] + sigma[:, 3]) / 2
        stokes_data[:, 1] = (data[:, 1] + data[:, 2]) / 2
        sigma_amp[:, 1] = (sigma[:, 1] + sigma[:, 2]) / 2
        stokes_data[:, 2] = 1j * (data[:, 1] - data[:, 2]) / 2
        sigma_amp[:, 2] = sigma_amp[:, 1]
        stokes_data[:, 3] = (data[:, 0] - data[:, 3]) / 2
        sigma_amp[:, 0] = (sigma[:, 0] + sigma[:, 3]) / 2
    elif 'XX' in pol_axis:
        stokes_data[:, 0] = (data[:, 0] + data[:, 3]) / 2
        sigma_amp[:, 0] = (sigma[:, 0] + sigma[:, 3]) / 2
        stokes_data[:, 1] = (data[:, 0] - data[:, 3]) / 2
        sigma_amp[:, 1] = sigma_amp[:, 0]
        stokes_data[:, 2] = (data[:, 1] + data[:, 2]) / 2
        sigma_amp[:, 2] = (sigma[:, 1] + sigma[:, 2]) / 2
        stokes_data[:, 3] = 1j * (data[:, 1] - data[:, 2]) / 2
        sigma_amp[:, 3] = sigma_amp[:, 2]
    else:
        raise Exception("Pol not supported " + str(pol_axis))
    stokes_amp = np.absolute(stokes_data)
    stokes_pha = np.angle(stokes_data, deg=True)
    sigma_amp[~np.isfinite(sigma_amp)] = np.nan
    sigma_amp[sigma_amp == 0] = np.nan
    snr = stokes_amp / sigma_amp
    cst = np.sqrt(9 / (2 * np.pi ** 3))
    # Both sigmas here are probably wrong because of the uncertainty of how weights are stored.
    sigma_pha = np.pi / np.sqrt(3) * (1 - cst * snr)
    sigma_pha = np.where(snr > 2.5, 1 / snr, sigma_pha)
    sigma_pha *= _convert_unit('rad', 'deg', 'trigonometric')
    return stokes_amp, stokes_pha, sigma_amp, sigma_pha
