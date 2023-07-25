import os
import dask
import json
import copy
import inspect
import numbers

import numpy as np

from astropy.time import Time

from pprint import pformat

from casacore import tables as ctables

from astrohack._utils._constants import pol_str

from astrohack._utils._conversion import _convert_ant_name_to_id
from astrohack._utils._extract_holog import _create_holog_meta_data
from astrohack._utils._extract_point import _extract_pointing

from astrohack._utils._dio import _load_point_file
from astrohack._utils._dio import  _check_if_file_will_be_overwritten, _check_if_file_exists
from astrohack._utils._dio import _load_holog_file
from astrohack._utils._dio import _write_meta_data

from astrohack._utils._extract_holog import _extract_holog_chunk

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

from astrohack._utils._param_utils._check_parms import _check_parms, _parm_check_passed

from astrohack._utils._tools import _remove_suffix
from astrohack._utils._tools import _jsonify

from astrohack.mds import AstrohackPointFile

# Added for clarity when inspecting stacktrace
CURRENT_FUNCTION=0

def extract_pointing(
    ms_name,
    point_name=None,
    parallel=False,
    overwrite=False,
):
    """ Extract pointing data, from measurement set. Creates holography output file.

    :param ms_name: Name of input measurement file name.
    :type ms_name: str

    :param point_name: Name of *<point_name>.point.zarr* file to create. Defaults to measurement set name with *point.zarr* extension.
    :type point_name: str, optional

    :param parallel: Boolean for whether to process in parallel. Defaults to False
    :type parallel: bool, optional

    :param overwrite: Overwrite pointing file on disk, defaults to False
    :type overwrite: bool, optional

    :return: Holography point object.
    :rtype: AstrohackPointFile

    .. _Description:

    **AstrohackPointFile**

    Point object allows the user to access point data via dictionary keys with values `ant`. The point object also provides a `summary()` helper function to list available keys for each file. 

    """
    
    # Returns the current local variables in dictionary form
    extract_pointing_params = locals()

    logger = _get_astrohack_logger()
    
    # Pull latest function fromt he stack, this is dynamic and preferred to hardcoding.
    function_name = inspect.stack()[CURRENT_FUNCTION].function

    ######### Parameter Checking #########
    extract_pointing_params = _check_extract_pointing_params(function_name=function_name, extract_point_params=extract_pointing_params)

    input_params = extract_pointing_params.copy()
    
    try:
        _check_if_file_exists(extract_pointing_params['ms_name'])
        _check_if_file_will_be_overwritten(extract_pointing_params['point_name'], extract_pointing_params['overwrite'])
    
    
        pnt_dict = _extract_pointing(
            ms_name=extract_pointing_params['ms_name'], 
            pnt_name=extract_pointing_params['point_name'], 
            parallel=extract_pointing_params['parallel']
        )
    
        # Calling this directly since it is so simple it doesn't need a "_create_{}" function.
        _write_meta_data(
            file_name="{name}/{ext}".format(name=extract_pointing_params['point_name'], ext=".point_attr"), 
            input_dict=extract_pointing_params
        )

        logger.info(f"[{function_name}]: Finished processing")
        point_dict = _load_point_file(file=extract_pointing_params["point_name"], dask_load=True)
    
        pointing_mds = AstrohackPointFile(extract_pointing_params['point_name'])
        pointing_mds._open()

        return pointing_mds
    
    except Exception as exception:
        logger.error("There was an error, exiting:: Exception: {exception}".format(exception=exception))
        
        return

def _check_extract_pointing_params(function_name, extract_point_params):

    #### Parameter Checking ####
    logger = _get_astrohack_logger()
    params_passed = True
    
    params_passed = params_passed and _check_parms(function_name, extract_point_params, 'ms_name', [str], default=None)

    base_name = _remove_suffix(extract_point_params['ms_name'], '.ms')
    params_passed = params_passed and _check_parms(function_name, extract_point_params,'point_name', [str],
                                                 default=base_name+'.point.zarr')

    point_base_name = _remove_suffix(extract_point_params['point_name'], '.point.zarr')
    params_passed = params_passed and _check_parms(function_name, extract_point_params, 'point_name', [str],
                                                 default=point_base_name+'.point.zarr')
    
        
    params_passed = params_passed and _check_parms(function_name, extract_point_params, 'parallel', [bool], default=False)

    params_passed = params_passed and _check_parms(function_name, extract_point_params, 'overwrite', [bool],default=False)

    _parm_check_passed(function_name, params_passed)

    return extract_point_params