import os
import json
import zarr
import copy
import numpy as np
import xarray as xr
import datetime

from astropy.io import fits
from astrohack import __version__ as code_version
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._tools import _numpy_to_json

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
    
    ant_list = [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]
    
    try:
        for ant in ant_list:
            if 'ant' in ant:
                ddi_list =  [dir_name for dir_name in os.listdir(file + "/" + str(ant)) if os.path.isdir(file + "/" + str(ant))]
                panel_data_dict[ant] = {}
                
                for ddi in ddi_list:
                    if 'ddi' in ddi:
                        if dask_load:
                            panel_data_dict[ant][ddi] = xr.open_zarr("{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))
                        else:
                            panel_data_dict[ant][ddi] = _open_no_dask_zarr("{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))
    
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
        holog_dict["pnt_dict"] = _load_point_file(file=holog_file, ant_list=None, dask_load=dask_load)

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


def _write_fits(header, imagetype, data, filename, unit, origin):
    """
    Write a dictionary and a dataset to a FITS file
    Args:
        header: The dictionary containing the header
        imagetype: Type to be added to FITS header
        data: The dataset
        filename: The name of the output file
        unit: to be set to bunit
    """

    header['BUNIT'] = unit
    header['TYPE'] = imagetype
    header['ORIGIN'] = f'Astrohack v{code_version}: {origin}'
    header['DATE'] = datetime.datetime.now().strftime('%b %d %Y, %H:%M:%S')
    hdu = fits.PrimaryHDU(data)
    for key in header.keys():
        hdu.header.set(key, header[key])
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


def _read_meta_data(file_name, file_type, origin):
    """Reads dimensional data from holog meta file.

    Args:
        file_name (str): astorhack file name.
        file_type (str): astrohack file type
        origin (str, list): Astrohack expected origin(s)

    Returns:
        dict: dictionary containing dimension data.
    """
    logger = _get_astrohack_logger()
    try:
        with open(f'{file_name}/.{file_type}_attr') as json_file:
            json_dict = json.load(json_file)

    except Exception as error:
        logger.error(str(error))
        raise

    try:
        metadataorigin = json_dict['origin']
    except KeyError:
        logger.error("[_read_meta_data]: Badly formatted metadata in input file")
        raise Exception('Bad metadata')
    if isinstance(origin, str):
        if metadataorigin != origin:
            logger.error(f"[_read_meta_data]: Input file is not an Astrohack {file_type} file")
            logger.error(f"Expected origin was {origin} but got {metadataorigin}")
            raise TypeError('Incorrect file type')
    elif isinstance(origin, (list, tuple)):
        if metadataorigin not in origin:
            logger.error(f"[_read_meta_data]: Input file is not an Astrohack {file_type} file")
            logger.error(f"Expected origin was {origin} but got {metadataorigin}")
            raise TypeError('Incorrect file type')

    return json_dict


def _check_mds_origin(file_name, file_type):
    """

    Args:
        file_name(str): astrohack file name.
        file_type(str, list): accepted astrohack file type

    Returns: origin(str) the origin of the mds file

    """
    logger = _get_astrohack_logger()
    if isinstance(file_type, str):
        file_type = [file_type]

    for ftype in file_type:
        try:
            with open(f'{file_name}/.{ftype}_attr') as json_file:
                json_dict = json.load(json_file)
                found = True
        except FileNotFoundError:
            found = False
        if found:
            break
    if not found:
        logger.error("[_check_mds_origin]: metadata not found")
        raise Exception('Metadata not found')

    try:
        metadataorigin = json_dict['origin']
    except KeyError:
        logger.error("[_check_mds_origin]: Badly formatted metadata in input file")
        raise Exception('Bad metadata')

    return metadataorigin


def _write_meta_data(origin, file_name, input_dict):
    """
    Creates a metadata dictionary that is compatible with JSON and writes it to a file
    Args:
        origin: Which function created the mds
        file_name: Output json file name
        input_dict: Dictionary to be included in the metadata
    """
    logger = _get_astrohack_logger()
    metadata = {'version': code_version,
                'origin': origin}
    for key in input_dict.keys():
        if type(input_dict[key]) == np.ndarray:
            try:
                for item in range(len(input_dict[key])):
                    newkey = f'{key}_{item}'
                    metadata[newkey] = _numpy_to_json(input_dict[key][item])
            except TypeError:
                if len(input_dict['grid_size'].shape) == 0:
                    metadata[key] = "None"
                else:
                    metadata[key] = input_dict[key]
        elif input_dict[key] is None:
            metadata[key] = "None"
        else:
            metadata[key] = input_dict[key]
    try:
        with open(file_name, "w") as json_file:
            json.dump(metadata, json_file)
    except Exception as error:
        logger.error("[_write_meta_data] {error}".format(error=error))


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


def _load_point_file(file, ant_list=None, dask_load=True, pnt_dict=None):
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

    return pnt_dict


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
