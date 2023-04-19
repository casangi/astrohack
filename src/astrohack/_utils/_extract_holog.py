import os
import json
import zarr
import copy
import numpy as np
import xarray as xr
import astropy
from astropy.io import fits
from scipy import spatial
from numba import njit
from numba.core import types
from numba.typed import Dict

from casacore import tables as ctables
from astrohack._utils._imaging import _calculate_parallactic_angle_chunk
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

from astrohack._utils._io import _load_point_file

def _extract_holog_chunk(extract_holog_params):
    """Perform data query on holography data chunk and get unique time and state_ids/

    Args:
        ms_name (str): Measurementset name
        data_col (str): Data column to extract.
        ddi (int): Data description id
        scan (int): Scan number
        map_ant_ids (numpy.narray): Array of antenna_id values corresponding to mapping data.
        ref_ant_ids (numpy.narray): Arry of antenna_id values corresponding to reference data.
        sel_state_ids (list): List pf state_ids corresponding to holography data/
    """
    logger = _get_astrohack_logger()

    ms_name = extract_holog_params["ms_name"]
    pnt_name = extract_holog_params["point_name"]
    data_col = extract_holog_params["data_col"]
    ddi = extract_holog_params["ddi"]
    scans = extract_holog_params["scans"]
    ant_names = extract_holog_params["ant_names"]
    ref_ant_per_map_ant_tuple = extract_holog_params["ref_ant_per_map_ant_tuple"]
    map_ant_tuple = extract_holog_params["map_ant_tuple"]
    ref_ant_per_map_ant_name_tuple = extract_holog_params["ref_ant_per_map_ant_name_tuple"]
    map_ant_name_tuple = extract_holog_params["map_ant_name_tuple"]
    
    holog_map_key = extract_holog_params["holog_map_key"]
    telescope_name = extract_holog_params["telescope_name"]
    
    assert len(ref_ant_per_map_ant_tuple) == len(map_ant_tuple), "ref_ant_per_map_ant_tuple and map_ant_tuple should have same length."
    
    sel_state_ids = extract_holog_params["sel_state_ids"]
    holog_name = extract_holog_params["holog_name"]
    overwrite = extract_holog_params["overwrite"]

    chan_freq = extract_holog_params["chan_setup"]["chan_freq"]
    pol = extract_holog_params["pol_setup"]["pol"]
    
    table_obj = ctables.table(ms_name, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)

    if sel_state_ids:
        ctb = ctables.taql(
            "select %s, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG from $table_obj WHERE DATA_DESC_ID == %s AND SCAN_NUMBER in %s AND STATE_ID in %s"
            % (data_col, ddi, list(scans), list(sel_state_ids))
        )
    else:
        ctb = ctables.taql(
            "select %s, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG from $table_obj WHERE DATA_DESC_ID == %s AND SCAN_NUMBER in %s"
            % (data_col, ddi, list(scans))
        )
        
    vis_data = ctb.getcol(data_col)
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
    
    #print(vis_data.shape,weight.shape,ant1.shape,ant2.shape,time_vis_row.shape,time_vis.shape,flag.shape,flag_row.shape,ref_ant_per_map_ant_tuple,map_ant_tuple,)
    
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
    
    ''' Removing for now. Code struggles with VLA pointing errors
    ### To DO:
    ################### Average multiple repeated samples
    if telescope_name != "ALMA":
        over_flow_protector_constant = float("%.5g" % time_vis[0])  # For example 5076846059.4 -> 5076800000.0
        time_vis = time_vis - over_flow_protector_constant

        
        time_vis = _average_repeated_pointings(vis_map_dict, weight_map_dict, flagged_mapping_antennas,time_vis,pnt_map_dict)
        
        time_vis = time_vis + over_flow_protector_constant
    '''
    
    holog_dict = _create_holog_file(
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
        overwrite=overwrite,
    )

    logger.info(
        "Finished extracting holography chunk for ddi: {ddi} holog_map_key: {holog_map_key}".format(
            ddi=ddi, holog_map_key=holog_map_key
        )
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
    overwrite,
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
    logger = _get_astrohack_logger()

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
            map_ant_tag = 'ant_' + ant_names[map_ant_index] #'ant_' + str(map_ant_index)

            direction = np.take(pnt_map_dict[map_ant_tag], indicies, axis=0)

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
                pnt_map_dict[map_ant_tag], dims=["time", "lm"]
            )

            xds.attrs["holog_map_key"] = holog_map_key
            #xds.attrs["ant_id"] = map_ant_tag
            xds.attrs["ddi"] = ddi
            xds.attrs["parallactic_samples"] = parallactic_samples
            xds.attrs["telescope_name"] = telescope_name
            xds.attrs["antenna_name"] = ant_names[map_ant_index]
            
            xds.attrs["l_max"] = np.max(xds["DIRECTIONAL_COSINES"][:,0].values)
            xds.attrs["l_min"] = np.min(xds["DIRECTIONAL_COSINES"][:,0].values)
            xds.attrs["m_max"] = np.max(xds["DIRECTIONAL_COSINES"][:,1].values)
            xds.attrs["m_min"] = np.min(xds["DIRECTIONAL_COSINES"][:,1].values)
            
            #print(xds)

            holog_file = holog_name

            if overwrite is False:
                if os.path.exists(holog_file):
                    logger.error(
                        "Holog file {file} exists. To overwite set the overwrite=True option in extract_holog or remove current file.".format(file=holog_file))
                    raise

            logger.info(
                "Writing holog file to {file}".format(file=holog_file)
            )
            xds.to_zarr(
                os.path.join(
                    holog_file, 'ddi_' + str(ddi) + "/" + str(holog_map_key) + "/" + "ant_" + str(ant_names[map_ant_index])
                ),
                mode="w",
                compute=True,
                consolidated=True,
            )

        else:
            logger.warning(
                "[FLAGGED DATA] mapping antenna index {index}".format(index=map_ant_index
                )
            )











### To DO
def _create_holog_obs_dict():
    '''
    Code to automatically create holog_obs_dict
    VLA datasets causeing issues.
    if holog_obs_dict is None:
        ant_names_list = []
        #Create mapping antennas
        holog_obs_dict = {}
        for ant_id,pnt_xds in pnt_dict.items():
            ant_name = pnt_xds.attrs['ant_name']
            mapping_scans = pnt_xds.attrs['mapping_scans']
            ant_names_list.append(ant_name)
            for ddi,scans in  mapping_scans.items():
                ddi = int(ddi)
                for s in scans:
                    try:
                        holog_obs_dict[ddi][s]['map'].append(ant_name)
                    except:
                        holog_obs_dict.setdefault(ddi,{})[s] = {'map':[ant_name]}#dict(zip(scans, [ant_name]*len(scans)))
       
       
        #Create reference antennas
        
        ant_names_set = set(ant_names_list)
        for ddi,scan in holog_obs_dict.items():
            for scan_id,ant_types in scan.items():
                holog_obs_dict[ddi][scan_id]['ref'] = ant_names_set - set(holog_obs_dict[ddi][scan_id]['map'])
            
    '''
    
    return 0


def _extract_pointing_chunk(map_ant_ids, time_vis, pnt_ant_dict):
    """Extract nearest MAIN table time indexed pointing map

    Args:
        map_ant_ids (dict): list of antenna ids
        time_vis (numpy.ndarray): sorted, unique list of visibility times
        pnt_ant_dict (dict): map of pointing directional cosines with a map key based on the antenna id and indexed by the MAIN table visibility time.

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
            .DIRECTIONAL_COSINES.values
        )

    return pnt_map_dict
