import os
import json
import zarr
import copy
import datetime
import shutil
import inspect
import pathlib

import numpy as np
import xarray as xr

import graphviper.utils.logger as logger

from astropy.io import fits
from astrohack import __version__ as code_version

from astrohack._utils._tools import _add_prefix
from astrohack._utils._tools import NumpyEncoder

DIMENSION_KEY = "_ARRAY_DIMENSIONS"


def _check_if_file_exists(file):
    

    if pathlib.Path(file).exists() is False:
        logger.error('File {file} does not exists.')
        raise FileNotFoundError


def _check_if_file_will_be_overwritten(file, overwrite):
    
    path = pathlib.Path(file)

    if (path.exists() is True) and (overwrite is False):
        logger.error(f'{file} already exists. To overwrite set overwrite to True, or remove current file.')

        raise FileExistsError("{file} exists.".format(file=file))

    elif (path.exists() is True) and (overwrite is True):
        if file.endswith(".zarr"):
            logger.warning(f'{file} will be overwritten.')
            shutil.rmtree(file)
        else:
            logger.warning(f'{file} may not be valid hack file. Check the file name again.')
            raise Exception(f"IncorrectFileType: {file}")


def _load_panel_file(file=None, panel_dict=None, dask_load=True):
    """ Open panel file.

    Args:
        file (str, optional): Path to panel file. Defaults to None.

    Returns:
        bool: Nested dictionary containing panel data xds.
    """
    
    panel_data_dict = {}

    if panel_dict is not None:
        panel_data_dict = panel_dict

    ant_list = [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]

    try:
        for ant in ant_list:
            if 'ant' in ant:
                ddi_list = [dir_name for dir_name in os.listdir(file + "/" + str(ant)) if
                            os.path.isdir(file + "/" + str(ant))]
                panel_data_dict[ant] = {}

                for ddi in ddi_list:
                    if 'ddi' in ddi:
                        if dask_load:
                            panel_data_dict[ant][ddi] = xr.open_zarr(
                                "{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))
                        else:
                            panel_data_dict[ant][ddi] = _open_no_dask_zarr(
                                "{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))

    except Exception as e:
        logger.error(str(e))
        raise
    return panel_data_dict


def _load_image_file(file=None, image_dict=None, dask_load=True):
    """ Open hologgraphy file.

    Args:s
        file (str, optional): Path to holography file. Defaults to None.

    Returns:
        bool: bool describing whether the file was opened properly
    """
    
    ant_data_dict = {}

    if image_dict is not None:
        ant_data_dict = image_dict

    ant_list = [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]

    try:
        for ant in ant_list:
            if 'ant' in ant:
                ddi_list = [dir_name for dir_name in os.listdir(file + "/" + str(ant)) if
                            os.path.isdir(file + "/" + str(ant))]
                ant_data_dict[ant] = {}

                for ddi in ddi_list:
                    if 'ddi' in ddi:
                        if dask_load:
                            ant_data_dict[ant][ddi] = xr.open_zarr(
                                "{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))
                        else:
                            ant_data_dict[ant][ddi] = _open_no_dask_zarr(
                                "{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))

    except Exception as e:
        logger.error(str(e))
        raise

    return ant_data_dict


def _load_locit_file(file=None, locit_dict=None, dask_load=True):
    """ Open Antenna position (locit) file.

    Args:
        file (str, optional): Path to holography file. Defaults to None.


    Returns:
        bool: bool describing whether the file was opened properly
    """
    
    ant_data_dict = {}

    if locit_dict is not None:
        ant_data_dict = locit_dict

    ant_list = [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]

    ant_data_dict['obs_info'] = _read_meta_data(f'{file}/.observation_info')
    ant_data_dict['ant_info'] = {}
    try:
        for ant in ant_list:
            if 'ant' in ant:
                ddi_list = [dir_name for dir_name in os.listdir(file + "/" + str(ant)) if
                            os.path.isdir(file + "/" + str(ant))]
                ant_data_dict[ant] = {}
                ant_data_dict['ant_info'][ant] = _read_meta_data(f'{file}/{ant}/.antenna_info')
                for ddi in ddi_list:
                    if 'ddi' in ddi:
                        if dask_load:
                            ant_data_dict[ant][ddi] = xr.open_zarr(
                                "{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))
                        else:
                            ant_data_dict[ant][ddi] = _open_no_dask_zarr(
                                "{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))
    except Exception as e:
        logger.error(str(e))
        raise
    return ant_data_dict


def _load_position_file(file=None, position_dict=None, dask_load=True, combine=False):
    """ Open position file.

    Args:
        file (str, optional): Path to holography file. Defaults to None.


    Returns:
        bool: bool describing whether the file was opened properly
    """
    
    ant_data_dict = {}

    if position_dict is not None:
        ant_data_dict = position_dict

    ant_list = [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]

    try:
        if combine:
            for ant in ant_list:
                if 'ant' in ant:
                    if dask_load:
                        ant_data_dict[ant] = xr.open_zarr(f'{file}/{ant}')
                    else:
                        ant_data_dict[ant] = _open_no_dask_zarr(f'{file}/{ant}')
        else:
            for ant in ant_list:
                if 'ant' in ant:
                    ddi_list = [dir_name for dir_name in os.listdir(file + "/" + str(ant)) if
                                os.path.isdir(file + "/" + str(ant))]
                    ant_data_dict[ant] = {}
                    for ddi in ddi_list:
                        if 'ddi' in ddi:
                            if dask_load:
                                ant_data_dict[ant][ddi] = xr.open_zarr(
                                    "{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))
                            else:
                                ant_data_dict[ant][ddi] = _open_no_dask_zarr(
                                    "{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))
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

    

    if holog_dict is None:
        holog_dict = {}

    if load_pnt_dict:
        logger.info("Loading pointing dictionary to holog ...")
        holog_dict["pnt_dict"] = _load_point_file(file=holog_file, ant_list=None, dask_load=dask_load)

    for ddi in os.listdir(holog_file):
        if "ddi_" in ddi:

            if ddi_id is None:
                if ddi not in holog_dict:
                    holog_dict[ddi] = {}
            else:
                if ddi == ddi_id:
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
                                    holog_dict[ddi][holog_map][ant] = _open_no_dask_zarr(
                                        mapping_ant_vis_holog_data_name)

    if ant_id is None:
        return holog_dict

    return holog_dict, _read_data_from_holog_json(holog_file=holog_file, holog_dict=holog_dict, ant_id=ant_id,
                                                  ddi_id=ddi_id)


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


def _write_fits(header, imagetype, data, filename, unit, origin):
    """
    Write a dictionary and a dataset to a FITS file
    Args:
        header: The dictionary containing the header
        imagetype: Type to be added to FITS header
        data: The dataset
        filename: The name of the output file
        unit: to be set to bunit
        origin: Which astrohack mds has created the FITS being written
    """

    header['BUNIT'] = unit
    header['TYPE'] = imagetype
    header['ORIGIN'] = f'Astrohack v{code_version}: {origin}'
    header['DATE'] = datetime.datetime.now().strftime('%b %d %Y, %H:%M:%S')

    hdu = fits.PrimaryHDU(_reorder_axes_for_fits(data))
    for key in header.keys():
        hdu.header.set(key, header[key])
    hdu.writeto(_add_prefix(filename, origin), overwrite=True)
    return


def _reorder_axes_for_fits(data: np.ndarray):
    carta_dim_order = (1, 0, 2, 3, )
    shape = data.shape
    n_dim = len(shape)
    if n_dim == 5:
        # Ignore the time dimension and flip polarization and frequency axes
        transpo = np.transpose(data[0, ...], carta_dim_order)
        # Flip vertical axis
        return np.flip(transpo, 2)
    elif n_dim == 2:
        return np.flipud(data)


def _create_destination_folder(destination):
    """
    Try to create a folder if it already exists raise a warning
    Args:
        destination: the folder to be created
    """
    

    try:
        os.mkdir(destination)
    except FileExistsError:
        logger.warning(f'Destination folder already exists, results may be overwritten')


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
        raise Exception(ampname + ' and ' + devname + ' have different dimensions')
    if amphead["CRPIX1"] != devhead["CRPIX1"] or amphead["CRVAL1"] != devhead["CRVAL1"] \
            or amphead["CDELT1"] != devhead["CDELT1"]:
        raise Exception(ampname + ' and ' + devname + ' have different axes descriptions')

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


def _read_meta_data(file_name):
    """Reads dimensional data from holog meta file.

    Args:
        file_name (str): astorhack file name.

    Returns:
        dict: dictionary containing dimension data.
    """
    

    try:
        with open(file_name) as json_file:
            json_dict = json.load(json_file)

    except Exception as error:
        logger.error(str(error))
        raise Exception

    return json_dict


def _write_meta_data(file_name, input_dict):
    """
    Creates a metadata dictionary that is compatible with JSON and writes it to a file
    Args:
        file_name: Output json file name
        input_dict: Dictionary to be included in the metadata
    """

    calling_function = 1
    

    meta_data = copy.deepcopy(input_dict)

    meta_data.update({
        'version': code_version,
        'origin': inspect.stack()[calling_function].function
    })

    try:
        with open(file_name, "w") as json_file:
            json.dump(meta_data, json_file, cls=NumpyEncoder)

    except Exception as error:
        logger.error(f'{error}')


def _read_data_from_holog_json(holog_file, holog_dict, ant_id, ddi_id=None):
    """Read holog file meta data and extract antenna based xds information for each (ddi, holog_map)

    Args:
        holog_file (str): holog file name.
        holog_dict (dict): holog file dictionary containing msxds data.
        ant_id (int): Antenna id

    Returns:
        nested dict: nested dictionary (ddi, holog_map, xds) with xds data embedded in it.
    """
    
    ant_id_str = str(ant_id)

    holog_meta_data = str(pathlib.Path(holog_file).joinpath(".holog_json"))

    try:
        with open(holog_meta_data, "r") as json_file:
            holog_json = json.load(json_file)

    except Exception as error:
        logger.error(str(error))
        raise Exception

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


def _load_point_file(file, ant_list=None, dask_load=True, pnt_dict=None, diagnostic=False):
    """Load pointing dictionary from disk.

    Args:
        file (zarr): Input zarr file containing pointing dictionary.

    Returns:
        dict: Pointing dictionary
    """
    if pnt_dict is None:
        pnt_dict = {}

    pnt_dict['point_meta_ds'] = xr.open_zarr(file)

    for ant in os.listdir(file):
        if "ant_" in ant:
            if (ant_list is None) or (ant in ant_list):
                if dask_load:
                    pnt_dict[ant] = xr.open_zarr(os.path.join(file, ant))
                else:
                    pnt_dict[ant] = _open_no_dask_zarr(os.path.join(file, ant))

    if diagnostic:
        _check_time_axis_consistency(pnt_dict)

    return pnt_dict


def _check_time_axis_consistency(pnt_dict):
    
    variable_length = {}

    for ant in pnt_dict.keys():
        if ant != "point_meta_ds":
            variable_length[ant] = pnt_dict[ant].POINTING_OFFSET.values.shape[0]

    # Calculate a fractional error
    std = np.std(list(variable_length.values()))
    mean = np.mean(list(variable_length.values()))
    fractional_error = std / mean

    if fractional_error >= 0.01:
        logger.warning(
            "There is an inconsistency in the data length along the time axis in one or more of the antennas "
            "POINTING_OFFSET; for more info run the logger in debug mode.")

        logger.debug("Pointing offset time axis length per antenna")
        logger.debug(variable_length)


def _get_attrs(zarr_obj):
    """Get attributes of zarr obj (groups or arrays)

    Args:
        zarr_obj (zarr): a zarr_group object

    Returns:
        dict: a group of zarr attributes
    """
    return {
        k: v for k, v in zarr_obj.attrs.asdict().items() if not k.startswith("_NC")
    }


def _reshape(array, columns):
    size = len(array)
    rows = int(size / columns)
    if rows <= 0:
        return 1, 0
    else:
        remainder = size - (rows * columns)

        return rows, remainder


def _print_array(array, columns, indent=4):
    rows, remainder = _reshape(array, columns)

    if columns > len(array):
        columns = len(array)

    str_line = ""

    for i in range(rows):
        temp = []
        for j in range(columns):
            k = columns * i + j
            if j == 0:
                temp.append("{:>3}".format(array[k]).rjust(indent, ' '))
            else:
                temp.append("{:>3}".format(array[k]))

        str_line += ", ".join(temp)
        str_line += "\n"

    temp = []
    if remainder > 0:
        for i in range(remainder):
            index = columns * rows + i

            if i == 0:
                temp.append("{:>3}".format(array[index]).rjust(indent, ' '))
            else:
                temp.append("{:>3}".format(array[index]))

        str_line += ", ".join(temp)

    print(str_line)
