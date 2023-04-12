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
from astrohack._utils._conversion import convert_dict_from_numba
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

DIMENSION_KEY = "_ARRAY_DIMENSIONS"


def check_if_file_exists(file):
    logger = _get_astrohack_logger()

    if os.path.exists(file) is False:
        logger.error(
            " File " + file + " does not exists."
            )
        raise FileNotFoundError


def check_if_file_will_be_overwritten(file,overwrite):
    logger = _get_astrohack_logger()
    if (os.path.exists(file) is True) and (overwrite is False):
        logger.error(
            " {file} already exists. To overwite set the overwrite=True option in extract_holog or remove current file.".format(
                file=file
            )
        )
        raise FileExistsError
    elif  (os.path.exists(file) is True) and (overwrite is True):
        logger.warning(
            file + " will be overwritten."
        )


def _load_panel_file(file=None, panel_dict=None, dask_load=True):
    """ Open panel file.

    Args:
        file (str, optional): Path to panel file. Defaults to None.

    Returns:
        bool: Nested dictionary containing panel data xds.
    """
    logger = _get_astrohack_logger()
    panel_data_dict = {}

    if panel_dict is not None:
            panel_data_dict = panel_dict
    
    ant_list =  [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]
    
    try:
        for ant in ant_list:
            if 'ant' in ant:
                ddi_list =  [dir_name for dir_name in os.listdir(file + "/" + str(ant)) if os.path.isdir(file + "/" + str(ant))]
                panel_data_dict[ant] = {}
                
                for ddi in ddi_list:
                    if 'ddi' in ddi:
                        if dask_load:
                            panel_data_dict[ant][ddi] = xr.open_zarr("{name}/{ant}/{ddi}/xds.zarr".format(name=file, ant=ant, ddi=ddi))
                        else:
                            panel_data_dict[ant][ddi] = _open_no_dask_zarr("{name}/{ant}/{ddi}/xds.zarr".format(name=file, ant=ant, ddi=ddi))
    
    except Exception as e:
            logger.error(str(e))
            raise

    
    return panel_data_dict


def _load_image_file(file=None, image_dict=None,  dask_load=True):
        """ Open hologgraphy file.

        Args:s
            file (str, optional): Path to holography file. Defaults to None.

        Returns:
            bool: bool describing whether the file was opened properly
        """
        logger = _get_astrohack_logger()
        ant_data_dict = {}

        if image_dict is not None:
            ant_data_dict = image_dict

        ant_list =  [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]
        
        try:
            for ant in ant_list:
                if 'ant' in ant:
                    ddi_list =  [dir_name for dir_name in os.listdir(file + "/" + str(ant)) if os.path.isdir(file + "/" + str(ant))]
                    ant_data_dict[ant] = {}
                    
                    for ddi in ddi_list:
                        if 'ddi' in ddi:
                            if dask_load:
                                ant_data_dict[ant][ddi] = xr.open_zarr("{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))
                            else:
                                ant_data_dict[ant][ddi] = _open_no_dask_zarr("{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))

        except Exception as e:
            logger.error(str(e))
            raise

        return ant_data_dict


def _load_holog_file(holog_file, dask_load=True, load_pnt_dict=True, ant_id=None, ddi_id=None, holog_dict=None):
    """Loads holog file from disk

    Args:
        holog_name (str): holog file name

    Returns:

    """
    
    logger = _get_astrohack_logger()
    
    if holog_dict is None:
        holog_dict = {}

    if load_pnt_dict == True:
        logger.info("Loading pointing dictionary to holog ...")
        holog_dict["pnt_dict"] = _load_pnt_dict(file=holog_file, ant_list=None, dask_load=dask_load)

    for ddi in os.listdir(holog_file):
        if "ddi_" in ddi:
            
            if ddi_id is None:
                if ddi not in holog_dict:
                    holog_dict[ddi] = {}
            else:
                if (ddi == ddi_id):
                    holog_dict[ddi] = {}
                else:
                    continue
                
            for holog_map in os.listdir(os.path.join(holog_file, ddi)):
                if "map_" in holog_map:
                    if holog_map not in holog_dict[ddi]:
                        holog_dict[ddi][holog_map] = {}
                    for ant in os.listdir(os.path.join(holog_file, ddi + "/" + holog_map)):
                        if "ant_" in ant:
                            if (ant_id is None) or (ant_id in ant):
                                mapping_ant_vis_holog_data_name = os.path.join(
                                    holog_file, ddi + "/" + holog_map + "/" + ant
                                )
                                

                                if dask_load:
                                    holog_dict[ddi][holog_map][ant] = xr.open_zarr(
                                        mapping_ant_vis_holog_data_name
                                    )
                                else:
                                    holog_dict[ddi][holog_map][ant] = _open_no_dask_zarr(mapping_ant_vis_holog_data_name)

    
    if ant_id is None:
        return holog_dict
        
    return holog_dict, _read_data_from_holog_json(holog_file=holog_file, holog_dict=holog_dict, ant_id=ant_id, ddi_id=ddi_id)


def _read_fits(filename):
    """
    Reads a square FITS file and do sanity checks on its dimensionality
    Args:
        filename: a string containing the FITS file name/path

    Returns:
    The FITS header and the associated data array
    """
    hdul = fits.open(filename)
    head = hdul[0].header
    data = hdul[0].data[0, 0, :, :]
    hdul.close()
    if head["NAXIS"] != 1:
        if head["NAXIS"] < 1:
            raise Exception(filename + " is not bi-dimensional")
        elif head["NAXIS"] > 1:
            for iax in range(2, head["NAXIS"]):
                if head["NAXIS" + str(iax + 1)] != 1:
                    raise Exception(filename + " is not bi-dimensional")
    if head["NAXIS1"] != head["NAXIS2"]:
        raise Exception(filename + " does not have the same amount of pixels in the x and y axes")
    return head, data


def _write_fits(head, data, filename):
    """
    Write a dictionary and a dataset to a FITS file
    Args:
        head: The dictionary containing the header
        data: The dataset
        filename: The name of the output file
    """
    hdu = fits.PrimaryHDU(data)
    hdu.header = head
    hdu.header["ORIGIN"] = "Astrohack"
    hdu.writeto(filename, overwrite=True)
    return


def _aips_holog_to_xds(ampname, devname):
    """
    Read amplitude and deviation FITS files onto a common Xarray dataset
    Args:
        ampname: Name of the amplitude FITS file
        devname: Name of the deviation FITS file

    Returns:
    Xarray dataset
    """
    amphead, ampdata = _read_fits(ampname)
    devhead, devdata = _read_fits(devname)
    ampdata = np.flipud(ampdata)
    devdata = np.flipud(devdata)

    if amphead["NAXIS1"] != devhead["NAXIS1"]:
        raise Exception(ampname+' and '+devname+' have different dimensions')
    if amphead["CRPIX1"] != devhead["CRPIX1"] or amphead["CRVAL1"] != devhead["CRVAL1"] \
            or amphead["CDELT1"] != devhead["CDELT1"]:
        raise Exception(ampname+' and '+devname+' have different axes descriptions')

    npoint, wavelength = _get_aips_headpars(devhead)
    u = np.arange(-amphead["CRPIX1"], amphead["NAXIS1"] - amphead["CRPIX1"]) * amphead["CDELT1"]
    v = np.arange(-amphead["CRPIX2"], amphead["NAXIS2"] - amphead["CRPIX2"]) * amphead["CDELT2"]

    xds = xr.Dataset()
    xds.attrs['npix'] = amphead["NAXIS1"]
    xds.attrs['cell_size'] = amphead["CDELT1"]
    xds.attrs['ref_pixel'] = amphead["CRPIX1"]
    xds.attrs['ref_value'] = amphead["CRVAL1"]
    xds.attrs['npoint'] = npoint
    xds.attrs['wavelength'] = wavelength
    xds.attrs['amp_unit'] = amphead["BUNIT"].strip()
    xds.attrs['AIPS'] = True
    xds.attrs['ant_name'] = amphead["TELESCOP"].strip()
    xds['AMPLITUDE'] = xr.DataArray(ampdata, dims=["u", "v"])
    xds['DEVIATION'] = xr.DataArray(devdata, dims=["u", "v"])
    coords = {"u": u, "v": v}
    xds = xds.assign_coords(coords)
    return xds


def _get_aips_headpars(head):
    """
    Fetch number of points used in holography and wavelength stored by AIPS on a FITS header
    Args:
        head: AIPS FITS header

    Returns:
    npoint, wavelength
    """
    npoint = np.nan
    wavelength = np.nan
    for line in head["HISTORY"]:
        wrds = line.split()
        if wrds[1] == "Visibilities":
            npoint = np.sqrt(int(wrds[-1]))
        elif wrds[1] == "Observing":
            wavelength = float(wrds[-2])
    return npoint, wavelength


def _load_image_xds(file_stem, ant, ddi, dask_load=True):
    """ Load specific image xds

    Args:
        file_name (str): File directory
        ant (int): Antenna ID
        ddi (int): DDI 

    Raises:
        FileNotFoundError: FileNotFoundError

    Returns:
        zarr: zarr image file
    """

    image_path = "{image}/{ant}/{ddi}".format(image=file_stem, ant=ant, ddi=ddi)

    if os.path.isdir(image_path):
        if dask_load:
            return xr.open_zarr(image_path)
        else:
            return _open_no_dask_zarr(image_path)
    else:
        raise FileNotFoundError("Image file: {} not found".format(image_path))


def _read_meta_data(holog_file):
    """Reads dimensional data from holog meta file.

    Args:
        holog_file (str): holog file name.

    Returns:
        dict: dictionary containing dimension data.
    """
    logger = _get_astrohack_logger()
    try:
        with open("{name}/{file}".format(name=holog_file, file="/.holog_attr")) as json_file:
            json_dict = json.load(json_file)

    except Exception as error:
        logger.error(str(error))
        raise

    return json_dict


def _read_data_from_holog_json(holog_file, holog_dict, ant_id, ddi_id=None):
    """Read holog file meta data and extract antenna based xds information for each (ddi, holog_map)

    Args:
        holog_file (str): holog file name.
        holog_dict (dict): holog file dictionary containing msxds data.
        ant_id (int): Antenna id

    Returns:
        nested dict: nested dictionary (ddi, holog_map, xds) with xds data embedded in it.
    """
    logger = _get_astrohack_logger()
    ant_id_str = str(ant_id)
    

    holog_meta_data = "/".join((holog_file, ".holog_json"))

    try:
        with open(holog_meta_data, "r") as json_file:
            holog_json = json.load(json_file)

    except Exception as error:
        logger.error(str(error))
        raise

    ant_data_dict = {}

    for ddi in holog_json[ant_id_str].keys():
        if "ddi_" in ddi:
            if (ddi_id is not None) and (ddi != ddi_id):
                continue
                
            for holog_map in holog_json[ant_id_str][ddi].keys():
                if "map_" in holog_map:
                    ant_data_dict.setdefault(ddi, {})[holog_map] = holog_dict[ddi][holog_map][ant_id]


    return ant_data_dict


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


def _load_pnt_dict(file, ant_list=None, dask_load=True):
    """Load pointing dictionary from disk.

    Args:
        file (zarr): Input zarr file containing pointing dictionary.

    Returns:
        dict: Pointing dictionary
    """
    pnt_dict = {}

    for ant in os.listdir(file):
        if "ant_" in ant:
            if (ant_list is None) or (ant in ant_list):
                if dask_load:
                    pnt_dict[ant] = xr.open_zarr(os.path.join(file, ant))
                else:
                    pnt_dict[ant] = _open_no_dask_zarr(os.path.join(file, ant))

    return pnt_dict


def _make_ant_pnt_chunk(ms_name, pnt_parms):
    """Extract subset of pointing table data into a dictionary of xarray data arrays. This is written to disk as a zarr file.
            This function processes a chunk the overalll data and is managed by Dask.

    Args:
        ms_name (str): Measurement file name.
        ant_id (int): Antenna id
        pnt_name (str): Name of output poitning dictinary file name.
    """
    logger = _get_astrohack_logger()
    
    ant_id = pnt_parms['ant_id']
    ant_name = pnt_parms['ant_name']
    pnt_name = pnt_parms['pnt_name']
    scan_time_dict = pnt_parms['scan_time_dict']
    
    table_obj = ctables.table(os.path.join(ms_name, "POINTING"), readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    tb = ctables.taql(
        "select DIRECTION, TIME, TARGET, ENCODER, ANTENNA_ID, POINTING_OFFSET from $table_obj WHERE ANTENNA_ID == %s"
        % (ant_id)
    )

    ### NB: Add check if directions refrence frame is Azemuth Elevation (AZELGEO)
    try:
        direction = tb.getcol("DIRECTION")[:, 0, :]
        target = tb.getcol("TARGET")[:, 0, :]
        encoder = tb.getcol("ENCODER")
        direction_time = tb.getcol("TIME")
        pointing_offset = tb.getcol("POINTING_OFFSET")[:, 0, :]
    except Exception as e:
        tb.close()
        logger.warning("Skipping antenna " + str(ant_id) + " no pointing info")

        return 0
    tb.close()
    table_obj.close()
    
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
    
    ###############

    mapping_scans = {}
    time_tree = spatial.KDTree(direction_time[:,None]) #Use for nearest interpolation
    
    for ddi_id, ddi in scan_time_dict.items():
        scan_list = []
        for scan_id, scan_time in ddi.items():
            _, time_index = time_tree.query(scan_time[:,None])
            sub_lm = np.abs(pnt_xds["DIRECTIONAL_COSINES"].isel(time=slice(time_index[0],time_index[1]))).mean()
            
            if sub_lm > 10**-12: #Antenna is mapping since lm is non-zero
                scan_list.append(scan_id)
        mapping_scans[ddi_id] = scan_list
            
    pnt_xds.attrs['mapping_scans'] = [mapping_scans]
    ###############

    pnt_xds.attrs['ant_name'] = pnt_parms['ant_name']
    
    
    logger.info(
        "Writing pointing xds to {file}".format(
            file=os.path.join(pnt_name, "ant_" + str(ant_name))
        )
    )
    
    pnt_xds.to_zarr(os.path.join(pnt_name, "ant_{}".format(str(ant_name)) ), mode="w", compute=True, consolidated=True)

@convert_dict_from_numba
@njit(cache=False, nogil=True)
def _extract_scan_time_dict(time, scan_ids, ddi_ids):
    d1 = Dict.empty(
        key_type=types.int64,
        value_type=np.zeros(2, dtype=types.float64),
    )
    
    scan_time_dict = Dict.empty(
        key_type=types.int64,
        value_type=d1, 
    )
    
    for i, s in enumerate(scan_ids):
        s = types.int64(s)
        t = time[i]
        ddi = ddi_ids[i]
        if ddi in scan_time_dict:
            if s in scan_time_dict[ddi]:
                if  scan_time_dict[ddi][s][0] > t:
                    scan_time_dict[ddi][s][0] = t
                if  scan_time_dict[ddi][s][1] < t:
                    scan_time_dict[ddi][s][1] = t
            else:
                scan_time_dict[ddi][s] = np.array([t,t])
        else:
            scan_time_dict[ddi] = {s: np.array([t,t])}

    return scan_time_dict

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
                "[FLAGGED DATA] scan: {scan} mapping antenna index {index}".format(
                    scan=scan, index=map_ant_index
                )
            )


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

    pnt_ant_dict = _load_pnt_dict(pnt_name, map_ant_name_list, dask_load=False)

    pnt_map_dict = _extract_pointing_chunk(map_ant_name_list, time_vis, pnt_ant_dict)
    
    ''' Removing for now. Code struggles with VLA pointing errors
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

def _get_attrs(zarr_obj):
    """Get attributes of zarr obj (groups or arrays)

    Args:
        zarr_obj (zarr): a zarr_group object

    Returns:
        dict: a group of zarr attibutes
    """
    return {
        k: v for k, v in zarr_obj.attrs.asdict().items() if not k.startswith("_NC")
    }
