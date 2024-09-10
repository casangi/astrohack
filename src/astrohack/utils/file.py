import os
import json
import copy
import zarr
import shutil
import pathlib

import numpy as np
import xarray as xr

import toolviper.utils.logger as logger
from toolviper.utils.console import Colorize

from astrohack.utils.data import read_meta_data

DIMENSION_KEY = "_ARRAY_DIMENSIONS"
colorize = Colorize()


def load_panel_file(file=None, panel_dict=None, dask_load=True):
    """ Open panel file.

    Args:
        dask_load ():
        panel_dict ():
        file (str, optional): Path to panel file. Defaults to None.

    Returns:
        bool: Nested dictionary containing panel data xds.
    """

    panel_data_dict = {}

    if panel_dict is not None:
        panel_data_dict = panel_dict

    ant_list = [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]

    if not pathlib.Path(file).exists():
        logger.error("Requested file {} doesn't exist ...".format(colorize.blue(file)))

        raise FileNotFoundError

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

    return panel_data_dict


def load_image_file(file=None, image_dict=None, dask_load=True):
    """ Open holography file.

    Args:
        file (str, optional): Path to holography file. Defaults to None.
        image_dict ():
        dask_load ():

    Returns:
        bool: bool describing whether the file was opened properly

    """

    ant_data_dict = {}

    if image_dict is not None:
        ant_data_dict = image_dict

    ant_list = [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]

    if not pathlib.Path(file).exists():
        logger.error("Requested file {} doesn't exist ...".format(colorize.blue(file)))

        raise FileNotFoundError

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

    return ant_data_dict


def load_locit_file(file=None, locit_dict=None, dask_load=True):
    """ Open Antenna position (locit) file.

    Args:
        dask_load ():
        locit_dict ():
        file (str, optional): Path to holography file. Defaults to None.


    Returns:
        bool: bool describing whether the file was opened properly
    """

    ant_data_dict = {}

    if locit_dict is not None:
        ant_data_dict = locit_dict

    ant_list = [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]

    ant_data_dict['observation_info'] = read_meta_data(f'{file}/.observation_info')
    ant_data_dict['antenna_info'] = {}

    if not pathlib.Path(file).exists():
        logger.error("Requested file {} doesn't exist ...".format(colorize.blue(file)))

        raise FileNotFoundError

    for ant in ant_list:
        if 'ant' in ant:
            ddi_list = [
                dir_name for dir_name in os.listdir(file + "/" + str(ant)) if os.path.isdir(file + "/" + str(ant))
            ]

            ant_data_dict[ant] = {}
            ant_data_dict['antenna_info'][ant] = read_meta_data(f'{file}/{ant}/.antenna_info')

            for ddi in ddi_list:
                if 'ddi' in ddi:
                    if dask_load:
                        ant_data_dict[ant][ddi] = xr.open_zarr(
                            "{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))
                    else:
                        ant_data_dict[ant][ddi] = _open_no_dask_zarr(
                            "{name}/{ant}/{ddi}".format(name=file, ant=ant, ddi=ddi))

    return ant_data_dict


def load_position_file(file=None, position_dict=None, dask_load=True, combine=False):
    """ Open position file.

    Args:
        combine ():
        dask_load ():
        position_dict ():
        file (str, optional): Path to holography file. Defaults to None.


    Returns:
        bool: bool describing whether the file was opened properly
    """

    ant_data_dict = {}

    if position_dict is not None:
        ant_data_dict = position_dict

    ant_list = [dir_name for dir_name in os.listdir(file) if os.path.isdir(file)]

    if not pathlib.Path(file).exists():
        logger.error("Requested file {} doesn't exist ...".format(colorize.blue(file)))

        raise FileNotFoundError

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

    return ant_data_dict


def load_holog_file(file, dask_load=True, load_pnt_dict=True, ant_id=None, ddi_id=None, holog_dict=None):
    """Loads holog file from disk

    Args:
        holog_dict ():
        ddi_id ():
        ant_id ():
        load_pnt_dict ():
        dask_load ():
        file ():


    Returns:

    """

    if holog_dict is None:
        holog_dict = {}

    if not pathlib.Path(file).exists():
        logger.error("Requested file {} doesn't exist ...".format(colorize.blue(file)))

        raise FileNotFoundError

    if load_pnt_dict:
        logger.info("Loading pointing dictionary to holog ...")
        holog_dict["pnt_dict"] = load_point_file(file=file, dask_load=dask_load)

    for ddi in os.listdir(file):
        if "ddi_" in ddi:

            if ddi_id is None:
                if ddi not in holog_dict:
                    holog_dict[ddi] = {}
            else:
                if ddi == ddi_id:
                    holog_dict[ddi] = {}
                else:
                    continue

            for holog_map in os.listdir(os.path.join(file, ddi)):
                if "map_" in holog_map:
                    if holog_map not in holog_dict[ddi]:
                        holog_dict[ddi][holog_map] = {}
                    for ant in os.listdir(os.path.join(file, ddi + "/" + holog_map)):
                        if "ant_" in ant:
                            if (ant_id is None) or (ant_id in ant):
                                mapping_ant_vis_holog_data_name = os.path.join(
                                    file, ddi + "/" + holog_map + "/" + ant
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

    return holog_dict, _read_data_from_holog_json(
        holog_file=file,
        holog_dict=holog_dict,
        ant_id=ant_id,
        ddi_id=ddi_id
    )


def overwrite_file(file, overwrite):
    path = pathlib.Path(file)

    if (path.exists() is True) and (overwrite is False):
        logger.error(f'{file} already exists. To overwrite set overwrite to True, or remove current file.')

        raise FileExistsError("{file} exists.".format(file=file))

    elif (path.exists() is True) and (overwrite is True):
        if file.endswith(".zarr"):
            logger.warning(f'{file} will be overwritten.')
            shutil.rmtree(file)
        else:
            logger.warning(f'{file} may not be valid astrohack file. Check the file name again.')
            raise Exception(f"IncorrectFileType: {file}")


def load_image_xds(file_stem, ant, ddi, dask_load=True):
    """ Load specific image xds

    Args:
        dask_load ():
        file_stem (str): File directory
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


def load_point_file(file, ant_list=None, dask_load=True, pnt_dict=None, diagnostic=False):
    """Load pointing dictionary from disk.

        Args:
            file (str): Input zarr file containing pointing dictionary.
            diagnostic (bool):
            pnt_dict (dict):
            dask_load (bool):
            ant_list (list):

        Returns:
            dict: Pointing dictionary
        """
    if pnt_dict is None:
        pnt_dict = {}

    if not pathlib.Path(file).exists():
        logger.error("Requested file {} doesn't exist ...".format(colorize.blue(file)))

        raise FileNotFoundError

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


def _read_data_from_holog_json(holog_file, holog_dict, ant_id, ddi_id=None):
    """Read holog file metadata and extract antenna based xds information for each (ddi, holog_map)

        Args:
            ddi_id ():
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


def _open_no_dask_zarr(zarr_name, slice_dict=None):
    """
        Alternative to xarray open_zarr where the arrays are not Dask Arrays.

        slice_dict: A dictionary of slice objects for which values to read form a dimension.
                    For example silce_dict={'time':slice(0,10)} would select the first 10 elements in the time dimension.
                    If a dim is not specified all values are returned.
        return:
            xarray.Dataset()
        """

    if slice_dict is None:
        slice_dict = {}

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
        logger.debug(str(variable_length))


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
