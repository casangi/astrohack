import os
import json
import numpy as np
import xarray as xr
import astropy
import astrohack
import skriba.logger

from numba import njit
from numba.core import types

from casacore import tables as ctables
from astrohack._utils._imaging import _calculate_parallactic_angle_chunk
from astrohack._utils._algorithms import _get_grid_parms, _significant_digits
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
    logger = skriba.logger.get_logger(logger_name="astrohack")

    ms_name = extract_holog_params["ms_name"]
    pnt_name = extract_holog_params["point_name"]
    data_column = extract_holog_params["data_column"]
    ddi = extract_holog_params["ddi"]
    scans = extract_holog_params["scans"]
    ant_names = extract_holog_params["ant_names"]
    ref_ant_per_map_ant_tuple = extract_holog_params["ref_ant_per_map_ant_tuple"]
    map_ant_tuple = extract_holog_params["map_ant_tuple"]
    ref_ant_per_map_ant_name_tuple = extract_holog_params["ref_ant_per_map_ant_name_tuple"]
    map_ant_name_tuple = extract_holog_params["map_ant_name_tuple"]

    holog_map_key = extract_holog_params["holog_map_key"]
    telescope_name = extract_holog_params["telescope_name"]

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
            "select %s, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG from $table_obj WHERE "
            "DATA_DESC_ID == %s AND SCAN_NUMBER in %s AND STATE_ID in %s"
            % (data_column, ddi, list(scans), list(sel_state_ids))
        )
    else:
        ctb = ctables.taql(
            "select %s, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG from $table_obj WHERE "
            "DATA_DESC_ID == %s AND SCAN_NUMBER in %s"
            % (data_column, ddi, list(scans))
        )

    vis_data = ctb.getcol(data_column)
    weight = ctb.getcol("WEIGHT")
    ant1 = ctb.getcol("ANTENNA1")
    ant2 = ctb.getcol("ANTENNA2")
    time_vis_row = ctb.getcol("TIME")
    time_vis_row_centroid = ctb.getcol("TIME_CENTROID")
    flag = ctb.getcol("FLAG")
    flag_row = ctb.getcol("FLAG_ROW")
    ctb.close()
    table_obj.close()

    time_vis, unique_index = np.unique(
        time_vis_row, return_index=True
    )  # Note that values are sorted.

    vis_map_dict, weight_map_dict, flagged_mapping_antennas = _extract_holog_chunk_jit(
        vis_data,
        weight,
        ant1,
        ant2,
        time_vis_row,
        time_vis,
        flag,
        flag_row,
        ref_ant_per_map_ant_tuple,
        map_ant_tuple,
    )

    del vis_data, weight, ant1, ant2, time_vis_row, flag, flag_row

    map_ant_name_list = list(map(str, map_ant_name_tuple))

    map_ant_name_list = ['ant_' + i for i in map_ant_name_list]

    pnt_ant_dict = _load_point_file(pnt_name, map_ant_name_list, dask_load=False)
    pnt_map_dict = _extract_pointing_chunk(map_ant_name_list, time_vis, pnt_ant_dict)
    grid_parms = _get_grid_parms(vis_map_dict, pnt_map_dict, ant_names)

    ### To DO:
    ################### Average multiple repeated samples
    # over_flow_protector_constant = float("%.5g" % time_vis[0])  # For example 5076846059.4 -> 5076800000.0
    # time_vis = time_vis - over_flow_protector_constant
    # from astrohack._utils._algorithms import _average_repeated_pointings
    # time_vis = _average_repeated_pointings(vis_map_dict, weight_map_dict, flagged_mapping_antennas,time_vis,pnt_map_dict,ant_names)
    # time_vis = time_vis + over_flow_protector_constant

    _create_holog_file(
        holog_name,
        vis_map_dict,
        weight_map_dict,
        pnt_map_dict,
        time_vis,
        chan_freq,
        pol,
        flagged_mapping_antennas,
        holog_map_key,
        ddi,
        ms_name,
        ant_names,
        grid_parms
    )

    logger.info("Finished extracting holography chunk for ddi: {ddi} holog_map_key: {holog_map_key}".format(
            ddi=ddi, holog_map_key=holog_map_key)
    )


@njit(cache=False, nogil=True)
def _extract_holog_chunk_jit(
        vis_data,
        weight,
        ant1,
        ant2,
        time_vis_row,
        time_vis,
        flag,
        flag_row,
        ref_ant_per_map_ant_tuple,
        map_ant_tuple,
):
    """JIT copiled function to extract relevant visibilty data from chunk after flagging and applying weights.

    Args:
        vis_data (numpy.ndarray): Visibility data (row, channel, polarization)
        weight (numpy.ndarray): Data weight values (row, polarization)
        ant1 (numpy.ndarray): List of antenna_ids for antenna1
        ant2 (numpy.ndarray): List of antenna_ids for antenna2
        time_vis_row (numpy.ndarray): Array of full time talues by row
        time_vis (numpy.ndarray): Array of unique time values from time_vis_row
        flag (numpy.ndarray): Array of data quality flags to apply to data
        flag_row (numpy.ndarray): Array indicating when a full row of data should be flagged

    Returns:
        dict: Antenna_id referenced (key) dictionary containing the visibility data selected by (time, channel, polarization)
    """

    n_row, n_chan, n_pol = vis_data.shape
    n_time = len(time_vis)

    vis_map_dict = {}
    sum_weight_map_dict = {}

    for antenna_id in map_ant_tuple:
        vis_map_dict[antenna_id] = np.zeros(
            (n_time, n_chan, n_pol), dtype=types.complex64
        )
        sum_weight_map_dict[antenna_id] = np.zeros(
            (n_time, n_chan, n_pol), dtype=types.float64
        )

    for row in range(n_row):

        if flag_row is False:
            continue

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

        # Find index of time_vis_row[row] in time_vis that maintains the value ordering
        time_index = np.searchsorted(time_vis, time_vis_row[row])

        for chan in range(n_chan):
            for pol in range(n_pol):
                if ~(flag[row, chan, pol]):
                    # Calculate running weighted sum of visibilities
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

    return vis_map_dict, sum_weight_map_dict, flagged_mapping_antennas


def _get_time_samples(time_vis):
    """Sample three values for time vis and cooresponding indicies. Values are sammpled as (first, middle, last)

    Args:
        time_vis (numpy.ndarray): a list of visibility times

    Returns:
        numpy.ndarray, list: a select subset of visibility times (first, middle, last)
    """

    n_time_vis = time_vis.shape[0]

    middle = int(n_time_vis // 2)
    indicies = [0, middle, n_time_vis - 1]

    return np.take(time_vis, indicies), indicies


def _create_holog_file(
        holog_name,
        vis_map_dict,
        weight_map_dict,
        pnt_map_dict,
        time_vis,
        chan,
        pol,
        flagged_mapping_antennas,
        holog_map_key,
        ddi,
        ms_name,
        ant_names,
        grid_parms,
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

    logger = skriba.logger.get_logger(logger_name="astrohack")

    ctb = ctables.table("/".join((ms_name, "ANTENNA")))
    observing_location = ctb.getcol("POSITION")

    ctb = ctables.table("/".join((ms_name, "OBSERVATION")))
    telescope_name = ctb.getcol("TELESCOPE_NAME")[0]

    ctb.close()

    time_vis_days = time_vis / (3600 * 24)
    astro_time_vis = astropy.time.Time(time_vis_days, format="mjd")
    time_samples, indicies = _get_time_samples(astro_time_vis)

    coords = {"time": time_vis, "chan": chan, "pol": pol}

    for map_ant_index in vis_map_dict.keys():
        if map_ant_index not in flagged_mapping_antennas:
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
                vis_map_dict[map_ant_index], dims=["time", "chan", "pol"]
            )

            xds["WEIGHT"] = xr.DataArray(
                weight_map_dict[map_ant_index], dims=["time", "chan", "pol"]
            )

            xds["DIRECTIONAL_COSINES"] = xr.DataArray(
                pnt_map_dict[map_ant_tag]["DIRECTIONAL_COSINES"].values, dims=["time", "lm"]
            )

            xds["IDEAL_DIRECTIONAL_COSINES"] = xr.DataArray(
                pnt_map_dict[map_ant_tag]["POINTING_OFFSET"].values, dims=["time", "lm"]
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

            xds.attrs["grid_parms"] = grid_parms[map_ant_tag]

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
            logger.warning("Mapping antenna index {index}".format(index=ant_names[map_ant_index]))


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

    logger = skriba.logger.get_logger(logger_name="astrohack")
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
                            map_ant_key) + 'has no reference antennas. If baseline_average_distance was specified '
                                           'increase this distance. See antenna distance matrix in log by setting '
                                           'debug level to DEBUG in client function.')

                    ref_ant_set = sub_ref_ant_set

                # Select reference antennas by n closest antennas
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
    """Extract nearest MAIN table time indexed pointing map

    Args:
        map_ant_ids (dict): list of antenna ids
        time_vis (numpy.ndarray): sorted, unique list of visibility times
        pnt_ant_dict (dict): map of pointing directional cosines with a map key based on the antenna id and indexed by
                             the MAIN table visibility time.

    Returns:
        dict:  Dictionary of directional cosine data mapped to nearest MAIN table sample times.
    """

    n_time_vis = time_vis.shape[0]

    pnt_map_dict = {}

    for antenna in map_ant_ids:
        pnt_map_dict[antenna] = np.zeros((n_time_vis, 2))

        pnt_map_dict[antenna] = (
            pnt_ant_dict[antenna]
            .interp(time=time_vis, method="nearest")
        )

    return pnt_map_dict


def _create_holog_meta_data(holog_file, holog_dict, input_params):
    """Save holog file meta information to json file with the transformation
        of the ordering (ddi, holog_map, ant) --> (ant, ddi, holog_map).

    Args:
        holog_name (str): holog file name.
        holog_dict (dict): Dictionary containing msdx data.
    """
    logger = skriba.logger.get_logger(logger_name="astrohack")

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

                            cell_sizes.append(xds.attrs["grid_parms"]["cell_size"])
                            n_pixs.append(xds.attrs["grid_parms"]["n_pix"])
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
            astrohack._utils._algorithms._calculate_suggested_grid_paramater(parameter=
                                                                                                  np.array(cell_sizes))

        logger.info("The suggested cell size is calculated to be: {cell_size}".format(cell_size=meta_data["cell_size"]))

    if not (len(set(n_pixs)) == 1):
        logger.warning('Number of pixels not consistent: ' + str(n_pixs))
        logger.warning('Calculating suggested number of pixels ...')

        meta_data['n_pix'] = int(
            astrohack._utils._algorithms._calculate_suggested_grid_paramater(parameter=np.array(n_pixs)))

        logger.info("The suggested number of pixels is calculated to be: {n_pix}".format(n_pix=meta_data["n_pix"]))

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
    real_lm = data['DIRECTIONAL_COSINES']*angle_fact
    ideal_lm = data['IDEAL_DIRECTIONAL_COSINES']*angle_fact
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
        pol_axis = data.pol.values
        if isinstance(param_dict['plot_correlation'], (list, tuple)):
            for correlation in param_dict['plot_correlation']:
                _plot_correlation(visi, correlation, pol_axis, time, real_lm, param_dict)
        else:
            if param_dict['plot_correlation'] == 'all':
                for correlation in pol_axis:
                    _plot_correlation(visi, correlation, pol_axis, time, real_lm, param_dict)
            else:
                _plot_correlation(visi, param_dict['plot_correlation'], pol_axis, time, real_lm, param_dict)


def _plot_correlation(visi, correlation, pol_axis, time, lm, param_dict):
    if correlation in pol_axis:
        ipol = pol_axis == correlation
        loc_vis = visi[:, ipol]
        if param_dict['complex_split'] == 'polar':
            y_data = [np.absolute(loc_vis)]
            y_label = [f'{correlation} Amplitude [arb. units]']
            title = ['Amplitude']
            y_data.append(np.angle(loc_vis)*_convert_unit('rad', param_dict["phase_unit"], 'trigonometric'))
            y_label.append(f'{correlation} Phase [{param_dict["phase_unit"]}]')
            title.append('Phase')
        else:
            y_data = [loc_vis.real]
            y_label = [f'Real {correlation} [arb. units]']
            title = ['real part']
            y_data.append(loc_vis.imag)
            y_label.append(f'Imaginary {correlation} [arb. units]')
            title.append('imaginary part')

        fig, ax = _create_figure_and_axes(param_dict['figure_size'], [2, 3])
        for isplit in range(2):
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
        logger = _get_astrohack_logger()
        logger.warning(f'Correlation {correlation} is not present for {param_dict["this_ant"]} {param_dict["this_ddi"]} '
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












