import dask
import os
import json
import zarr
import copy
import numbers

import numpy as np

import xarray as xr
import dask.array as da

import astropy
import astropy.units as u
import astropy.coordinates as coord

from numba import njit
from numba.core import types
from numba.typed import Dict
from datetime import datetime

from casacore import tables as ctables

from astrohack._utils import _system_message as console
from astrohack._utils._parallactic_angle import _calculate_parallactic_angle_chunk


DIMENSION_KEY = "_ARRAY_DIMENSIONS"

jit_cache = False


def _read_dimensions_meta_data(hack_file, ddi, ant_id):
    """Reads dimensional data from hack meta file.

    Args:
        ant_id (int): Antenna id
        hack_file (str): Hack file name.

    Returns:
        dict: dictionary containing dimension data.
    """
    try:
        with open(
            "{name}/{ddi}/{file}".format(name=hack_file, ddi=ddi, file="/.hack_attr")
        ) as json_file:
            json_dict = json.load(json_file)

    except Exception as error:
        console.error("[_read_dimensions_meta_data] {error}".format(error=error))

    return json_dict[str(ant_id)]


def _read_data_from_hack_meta(hack_file, hack_dict, ant_id):
    """Read hack file meta data and extract antenna based xds information for each (ddi, scan)

    Args:
        hack_file (str): Hack file name.
        hack_dict (dict): Hack file dictionary containing msxds data.
        ant_id (int): Antenna id

    Returns:
        nested dict: nested dictionary (ddi, scan, xds) with xds data embedded in it.
    """

    ant_id_str = str(ant_id)

    hack_meta_data = "/".join((hack_file, ".hack_json"))

    try:
        with open(hack_meta_data, "r") as json_file:
            hack_json = json.load(json_file)

    except Exception as error:
        console.error("[_read_data_from_hack_meta] {error}".format(error=error))

    ant_data_dict = {}

    for ddi in hack_json[ant_id_str].keys():
        for scan in hack_json[ant_id_str][ddi].keys():
            ant_data_dict.setdefault(int(ddi), {})[int(scan)] = hack_dict[int(ddi)][
                int(scan)
            ][int(ant_id)]

    return ant_data_dict


def _create_hack_meta_data(hack_file, hack_dict):
    """Save hack file meta information to json file with the transformation
        of the ordering (ddi, scan, ant) --> (ant, ddi, scan).

    Args:
        hack_name (str): Hack file name.
        hack_dict (dict): Dictionary containing msdx data.
    """

    for ddi, scan_dict in hack_dict.items():
        if isinstance(ddi, numbers.Number):
            ant_sub_dict = {}
            ant_hack_dict = {}
            data_extent = {}
            max_extent = {}
            dims_meta_data = {}
            lm_extent = {"l": {"min": [], "max": []}, "m": {"min": [], "max": []}}

            for scan, ant_dict in scan_dict.items():
                for ant, xds in ant_dict.items():
                    ant_sub_dict.setdefault(ddi, {})
                    ant_hack_dict.setdefault(ant, ant_sub_dict)[ddi][
                        scan
                    ] = xds.to_dict(data=False)
                    ant_sub_dict = {}

                    # Find the average (l, m) extent for each antenna, over (ddi, scan) and write the meta data to file.
                    dims = xds.dims
                    # max_vis = np.max(xds.DIRECTIONAL_COSINES.values)
                    lm_extent["l"]["min"].append(
                        np.min(xds.DIRECTIONAL_COSINES.values[:, 0])
                    )
                    lm_extent["l"]["max"].append(
                        np.max(xds.DIRECTIONAL_COSINES.values[:, 0])
                    )

                    lm_extent["m"]["min"].append(
                        np.min(xds.DIRECTIONAL_COSINES.values[:, 1])
                    )
                    lm_extent["m"]["max"].append(
                        np.max(xds.DIRECTIONAL_COSINES.values[:, 1])
                    )

                    data_extent.setdefault(ant, np.array([]))
                    data_extent[ant] = np.append(data_extent[ant], dims["time"])

                    dims_meta_data.setdefault(
                        ant,
                        {
                            "time": dims["time"],
                            "chan": dims["chan"],
                            "pol": dims["pol"],
                        },
                    )

        for ant, values in data_extent.items():
            max_value = np.max(values)

            max_extent[ant] = {
                "n_time": max_value,
                "time": dims_meta_data[ant]["time"],
                "pol": dims_meta_data[ant]["pol"],
                "chan": dims_meta_data[ant]["chan"],
                "extent": {
                    "l": {
                        "min": np.array(lm_extent["l"]["min"]).mean(),
                        "max": np.array(lm_extent["l"]["max"]).mean(),
                    },
                    "m": {
                        "min": np.array(lm_extent["m"]["min"]).mean(),
                        "max": np.array(lm_extent["m"]["max"]).mean(),
                    },
                },
            }

        output_attr_file = "{name}/{ddi}/{ext}".format(
            name=hack_file, ddi=ddi, ext=".hack_attr"
        )

        try:
            with open(output_attr_file, "w") as json_file:
                json.dump(max_extent, json_file)

        except Exception as error:
            console.error("[_create_hack_meta_data] {error}".format(error=error))

    output_meta_file = "{name}/{ext}".format(name=hack_file, ext=".hack_json")

    try:
        with open(output_meta_file, "w") as json_file:
            json.dump(ant_hack_dict, json_file)

    except Exception as error:
        console.error("[_create_hack_meta_data] {error}".format(error=error))


def _get_attrs(zarr_obj):
    """Get attributes of zarr obj (groups or arrays)

    Args:
        zarr_obj (zarr): a zarr_group object

    Returns:
        dict: a group of zarr attibutes
    """
    return {k: v for k, v in zarr_obj.attrs.asdict().items() if not k.startswith("_NC")}


def _open_no_dask_zarr(zarr_name, slice_dict={}):
    """
    Alternative to xarray open_zarr where the arrays are not Dask Arrays.

    slice_dict: A dictionary of slice objects for which values to read form a dimension.
                For example silce_dict={'time':slice(0,10)} would select the first 10 elements in the time dimension.
                If a dim is not specified all values are returned.
    return:
        xarray.Dataset()
    """

    zarr_group = zarr.open_group(store=zarr_name, mode="r")
    group_attrs = _get_attrs(zarr_group)

    slice_dict_complete = copy.deepcopy(slice_dict)
    coords = {}
    xds = xr.Dataset()

    for var_name, var in zarr_group.arrays():
        var_attrs = _get_attrs(var)

        for dim in var_attrs[DIMENSION_KEY]:
            if dim not in slice_dict_complete:
                slice_dict_complete[dim] = slice(None)  # No slicing.

        if (var_attrs[DIMENSION_KEY][0] == var_name) and (
            len(var_attrs[DIMENSION_KEY]) == 1
        ):
            coords[var_name] = var[
                slice_dict_complete[var_attrs[DIMENSION_KEY][0]]
            ]  # Dimension coordinates.
        else:
            # Construct slicing
            slicing_list = []
            for dim in var_attrs[DIMENSION_KEY]:
                slicing_list.append(slice_dict_complete[dim])
            slicing_tuple = tuple(slicing_list)
            xds[var_name] = xr.DataArray(
                var[slicing_tuple], dims=var_attrs[DIMENSION_KEY]
            )

    xds = xds.assign_coords(coords)

    xds.attrs = group_attrs
    return xds


#### Pointing Table Conversion ####
def _load_pnt_dict(file, ant_list=None, dask_load=True):
    """Load pointing dictionary from disk.

    Args:
        file (zarr): Input zarr file containing pointing dictionary.

    Returns:
        dict: Pointing dictionary
    """
    pnt_dict = {}

    for f in os.listdir(file):
        if f.isnumeric():
            if (ant_list is None) or (int(f) in ant_list):
                if dask_load:
                    pnt_dict[int(f)] = xr.open_zarr(os.path.join(file, f))
                else:
                    pnt_dict[int(f)] = _open_no_dask_zarr(os.path.join(file, f))

    return pnt_dict


def _make_ant_pnt_xds_chunk(ms_name, ant_id, pnt_name):
    """Extract subset of pointing table data into a dictionary of xarray data arrays. This is written to disk as a zarr file.
            This function processes a chunk the overalll data and is managed by Dask.

    Args:
        ms_name (str): Measurement file name.
        ant_id (int): Antenna id
        pnt_name (str): Name of output poitning dictinary file name.
    """

    tb = ctables.taql(
        "select DIRECTION, TIME, TARGET, ENCODER, ANTENNA_ID, POINTING_OFFSET from %s WHERE ANTENNA_ID == %s"
        % (os.path.join(ms_name, "POINTING"), ant_id)
    )

    ### NB: Add check if directions refrence frame is Azemuth Elevation (AZELGEO)
    direction = tb.getcol("DIRECTION")[:, 0, :]
    target = tb.getcol("TARGET")[:, 0, :]
    encoder = tb.getcol("ENCODER")
    direction_time = tb.getcol("TIME")
    pointing_offset = tb.getcol("POINTING_OFFSET")[:, 0, :]

    tb.close()

    """Using CASA table tool
    tb = table()
    tb.open(os.path.join(ms_name,"POINTING"), nomodify=True, lockoptions={'option': 'usernoread'})
    pt_ant_table = tb.taql('select DIRECTION,TIME,TARGET,ENCODER,ANTENNA_ID,POINTING_OFFSET from %s WHERE ANTENNA_ID == %s' % (os.path.join(ms_name,"POINTING"),ant_id))
    
    ### NB: Add check if directions refrence frame is Azemuth Elevation (AZELGEO)
    
    direction = np.swapaxes(pt_ant_table.getcol('DIRECTION')[:,0,:],0,1)
    target = np.swapaxes(pt_ant_table.getcol('TARGET')[:,0,:],0,1)
    encoder = np.swapaxes(pt_ant_table.getcol('ENCODER'),0,1)
    direction_time = pt_ant_table.getcol('TIME')
    pointing_offset = np.swapaxes(pt_ant_table.getcol('POINTING_OFFSET')[:,0,:],0,1)
    tb.close()
    """

    pnt_xds = xr.Dataset()
    coords = {"time": direction_time}
    pnt_xds = pnt_xds.assign_coords(coords)

    # Measurement set v2 definition: https://drive.google.com/file/d/1IapBTsFYnUT1qPu_UK09DIFGM81EIZQr/view?usp=sharing
    # DIRECTION: Antenna pointing direction
    pnt_xds["DIRECTION"] = xr.DataArray(direction, dims=("time", "az_el"))

    # ENCODER: The current encoder values on the primary axes of the mount type for the antenna, expressed as a Direction
    # Measure.
    pnt_xds["ENCODER"] = xr.DataArray(encoder, dims=("time", "az_el"))

    # TARGET: This is the true expected position of the source, including all coordinate corrections such as precession,
    # nutation etc.
    pnt_xds["TARGET"] = xr.DataArray(target, dims=("time", "az_el"))

    # POINTING_OFFSET: The a priori pointing corrections applied by the telescope in pointing to the DIRECTION position,
    # optionally expressed as polynomial coefficients.
    pnt_xds["POINTING_OFFSET"] = xr.DataArray(pointing_offset, dims=("time", "az_el"))

    # Calculate directional cosines (l,m) which are used as the gridding locations.
    # See equations 8,9 in https://library.nrao.edu/public/memos/evla/EVLAM_212.pdf.
    # TARGET: A_s, E_s (target source position)
    # DIRECTION: A_a, E_a (Antenna's pointing direction)

    ### NB: Is VLA's definition of Azimuth the same for ALMA, MeerKAT, etc.? (positive for a clockwise rotation from north, viewed from above)
    ### NB: Compare with calulation using WCS in astropy.
    l = np.cos(target[:, 1]) * np.sin(target[:, 0] - direction[:, 0])
    m = np.sin(target[:, 1]) * np.cos(direction[:, 1]) - np.cos(target[:, 1]) * np.sin(
        direction[:, 1]
    ) * np.cos(target[:, 0] - direction[:, 0])

    pnt_xds["DIRECTIONAL_COSINES"] = xr.DataArray(
        np.array([l, m]).T, dims=("time", "ra_dec")
    )

    console.info(
        "[_make_ant_pnt_xds_chunk] Writing pointing xds to {file}".format(
            file=os.path.join(pnt_name, str(ant_id))
        )
    )
    pnt_xds.to_zarr(
        os.path.join(pnt_name, str(ant_id)), mode="w", compute=True, consolidated=True
    )


def _make_ant_pnt_dict(ms_name, pnt_name, parallel=True):
    """Top level function to extract subset of pointing table data into a dictionary of xarray dataarrays.

    Args:
        ms_name (str): Measurement file name.
        pnt_name (str): Output pointing dictionary file name.
        parallel (bool, optional): Process in parallel. Defaults to True.

    Returns:
        dict: pointing dictionary of xarray dataarrays
    """

    ctb = ctables.table(
        os.path.join(ms_name, "ANTENNA"),
        readonly=True,
        lockoptions={"option": "usernoread"},
    )

    antenna_name = ctb.getcol("NAME")
    antenna_id = np.arange(len(antenna_name))

    ctb.close()

    if parallel:
        delayed_pnt_list = []
        for id in antenna_id:
            delayed_pnt_list.append(
                dask.delayed(_make_ant_pnt_xds_chunk)(
                    dask.delayed(ms_name), dask.delayed(id), dask.delayed(pnt_name)
                )
            )
        dask.compute(delayed_pnt_list)
    else:
        for id in antenna_id:
            _make_ant_pnt_xds_chunk(ms_name, id, pnt_name)

    return _load_pnt_dict(pnt_name)


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


@njit(cache=jit_cache, nogil=True)
def _extract_holog_chunk_jit(
    vis_data,
    weight,
    ant1,
    ant2,
    time_vis_row,
    time_vis,
    flag,
    flag_row,
    map_ant_ids,
    ref_ant_ids,
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
        map_ant_ids (numpy.ndarray): Array of antenna_ids for mapping data
        ref_ant_ids (numpy.ndarray): Array of antenna_ids for reference data

    Returns:
        dict: Antenna_id referenced (key) dictionary containing the visibility data selected by (time, channel, polarization)
    """

    n_row, n_chan, n_pol = vis_data.shape
    n_time = len(time_vis)

    vis_map_dict = {}
    sum_weight_map_dict = {}

    for antenna_id in map_ant_ids:
        vis_map_dict[antenna_id] = np.zeros(
            (n_time, n_chan, n_pol), dtype=types.complex64
        )
        sum_weight_map_dict[antenna_id] = np.zeros(
            (n_time, n_chan, n_pol), dtype=types.complex64
        )

    for row in range(n_row):

        if flag_row is False:
            continue

        ant1_index = ant1[row]
        ant2_index = ant2[row]

        if (ant1_index in map_ant_ids) and (ant2_index in ref_ant_ids):
            vis_baseline = vis_data[row, :, :]  # n_chan x n_pol
            map_ant_index = ant1_index  # mapping antenna index

        elif (ant2_index in map_ant_ids) and (
            ant1_index not in ref_ant_ids
        ):  # conjugate
            vis_baseline = np.conjugate(vis_data[row, :, :])
            map_ant_index = ant2_index

        else:
            continue

        # Find index of time_vis_row[row] in time_vis that maintains the value ordering
        time_index = np.searchsorted(time_vis, time_vis_row[row])

        for chan in range(n_chan):
            for pol in range(n_pol):
                if ~(flag[row, chan, pol]):
                    # Calculate running weighted sum of visibilities
                    vis_map_dict[map_ant_index][time_index, chan, pol] = (
                        vis_map_dict[map_ant_index][time_index, chan, pol]
                        + vis_baseline[chan, pol] * weight[row, pol]
                    )

                    # Calculate running sum of weights
                    sum_weight_map_dict[map_ant_index][time_index, chan, pol] = (
                        sum_weight_map_dict[map_ant_index][time_index, chan, pol]
                        + weight[row, pol]
                    )

    flagged_mapping_antennas = []

    for map_ant_index in vis_map_dict.keys():
        sum_of_sum_weight = 0

        for time_index in range(n_time):
            for chan in range(n_chan):
                for pol in range(n_pol):
                    sum_weight = sum_weight_map_dict[map_ant_index][
                        time_index, chan, pol
                    ]
                    sum_of_sum_weight = sum_of_sum_weight + sum_weight
                    if sum_weight == 0:
                        vis_map_dict[map_ant_index][time_index, chan, pol] = 0.0
                    else:
                        vis_map_dict[map_ant_index][time_index, chan, pol] = (
                            vis_map_dict[map_ant_index][time_index, chan, pol]
                            / sum_weight
                        )

        if sum_of_sum_weight == 0:
            flagged_mapping_antennas.append(map_ant_index)

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


def _create_hack_file(
    hack_name,
    vis_map_dict,
    weight_map_dict,
    pnt_map_dict,
    time_vis,
    chan,
    pol,
    flagged_mapping_antennas,
    scan,
    ddi,
    ms_name,
    overwrite,
):
    """Create hack-structured, formatted output file and save to zarr.

    Args:
        hack_name (str): Hack file name.
        vis_map_dict (dict): a nested dictionary/map of weighted visibilities indexed as [antenna][time, chan, pol]; mainains time ordering.
        weight_map_dict (dict): weights dictionary/map for visibilites in vis_map_dict
        pnt_map_dict (dict): pointing table map dictionary
        time_vis (numpy.ndarray): time_vis values
        chan (numpy.ndarray): channel values
        pol (numpy.ndarray): polarization values
        flagged_mapping_antennas (numpy.ndarray): list of mapping antennas that have been flagged.
        scan (numpy.ndarray): scan number
        ddi (numpy.ndarray): data description id; a combination of polarization and spectral window
    """

    ctb = ctables.table("/".join((ms_name, "ANTENNA")))
    observing_location = ctb.getcol("POSITION")
    ctb.close()

    time_vis_days = time_vis / (3600 * 24)
    astro_time_vis = astropy.time.Time(time_vis_days, format="mjd")
    time_samples, indicies = _get_time_samples(astro_time_vis)

    coords = {"time": time_vis, "chan": chan, "pol": pol}

    for map_ant_index in vis_map_dict.keys():
        if map_ant_index not in flagged_mapping_antennas:

            parallactic_samples = _calculate_parallactic_angle_chunk(
                time_samples=time_samples,
                observing_location=observing_location[map_ant_index],
                direction=pnt_map_dict[map_ant_index],
                indicies=indicies,
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
                pnt_map_dict[map_ant_index], dims=["time", "lm"]
            )
            xds.attrs["scan"] = scan
            xds.attrs["ant_id"] = map_ant_index
            xds.attrs["ddi"] = ddi
            xds.attrs["parallactic_samples"] = parallactic_samples

            hack_file = "{base}.{suffix}".format(base=hack_name, suffix="holog.zarr")

            if overwrite is False:
                if os.path.exists(hack_file):
                    console.warning(
                        "[_create_hack_file] Hack file {file} exists. To overwite set the overwrite=True option in extract_holog or remove current file.".format(
                            file=hack_file
                        )
                    )

            console.info(
                "[_create_hack_file] Writing hack file to {file}".format(file=hack_file)
            )
            xds.to_zarr(
                os.path.join(
                    hack_file, str(ddi) + "/" + str(scan) + "/" + str(map_ant_index)
                ),
                mode="w",
                compute=True,
                consolidated=True,
            )

        else:
            console.warning(
                "[_create_hack_file] [FLAGGED DATA] scan: {scan} mapping antenna index {index}".format(
                    scan=scan, index=map_ant_index
                )
            )


def _extract_holog_chunk(extract_holog_parms):
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

    ms_name = extract_holog_parms["ms_name"]
    pnt_name = extract_holog_parms["pnt_name"]
    data_col = extract_holog_parms["data_col"]
    ddi = extract_holog_parms["ddi"]
    scan = extract_holog_parms["scan"]
    map_ant_ids = extract_holog_parms["map_ant_ids"]
    ref_ant_ids = extract_holog_parms["ref_ant_ids"]
    sel_state_ids = extract_holog_parms["sel_state_ids"]
    hack_name = extract_holog_parms["hack_name"]
    overwrite = extract_holog_parms["overwrite"]

    chan_freq = extract_holog_parms["chan_setup"]["chan_freq"]
    pol = extract_holog_parms["pol_setup"]["pol"]

    ctb = ctables.taql(
        "select %s, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG, STATE_ID from %s WHERE DATA_DESC_ID == %s AND SCAN_NUMBER == %s AND STATE_ID in %s"
        % (data_col, ms_name, ddi, scan, sel_state_ids)
    )

    vis_data = ctb.getcol("DATA")
    weight = ctb.getcol("WEIGHT")
    ant1 = ctb.getcol("ANTENNA1")
    ant2 = ctb.getcol("ANTENNA2")
    time_vis_row = ctb.getcol("TIME")
    time_vis_row_centroid = ctb.getcol("TIME_CENTROID")
    flag = ctb.getcol("FLAG")
    flag_row = ctb.getcol("FLAG_ROW")
    state_ids_row = ctb.getcol("STATE_ID")

    ctb.close()

    time_vis, unique_index = np.unique(
        time_vis_row, return_index=True
    )  # Note that values are sorted.
    state_ids = state_ids_row[unique_index]

    vis_map_dict, weight_map_dict, flagged_mapping_antennas = _extract_holog_chunk_jit(
        vis_data,
        weight,
        ant1,
        ant2,
        time_vis_row,
        time_vis,
        flag,
        flag_row,
        map_ant_ids,
        ref_ant_ids,
    )

    del vis_data, weight, ant1, ant2, time_vis_row, flag, flag_row

    pnt_ant_dict = _load_pnt_dict(pnt_name, map_ant_ids, dask_load=False)

    pnt_map_dict = _extract_pointing_chunk(map_ant_ids, time_vis, pnt_ant_dict)

    hack_dict = _create_hack_file(
        hack_name,
        vis_map_dict,
        weight_map_dict,
        pnt_map_dict,
        time_vis,
        chan_freq,
        pol,
        flagged_mapping_antennas,
        scan,
        ddi,
        ms_name,
        overwrite=overwrite,
    )

    console.info(
        "Finished extracting holography chunk for ddi: {ddi} scan: {scan}".format(
            ddi=ddi, scan=scan
        )
    )
