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
from astrohack._utils._dio import  check_if_file_will_be_overwritten, check_if_file_exists
from astrohack._utils._dio import _load_holog_file
from astrohack._utils._dio import _write_meta_data

from astrohack._utils._extract_holog import _extract_holog_chunk

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

from astrohack._utils._parm_utils._check_parms import _check_parms, _parm_check_passed

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
    extract_pointing_parms = locals()

    logger = _get_astrohack_logger()
    
    # Pull latest function fromt he stack, this is dynamic and preferred to hardcoding.
    function_name = inspect.stack()[CURRENT_FUNCTION].function

    ######### Parameter Checking #########
#    extract_pointing_parms = _check_extract_pointing_parms(fname,
#                                                     ms_name,
#                                                     point_name,
#                                                     parallel,
#                                                     overwrite)

#    input_params = extract_pointing_parms.copy()
    
    check_if_file_exists(function_name, extract_pointing_parms['ms_name'])
    check_if_file_will_be_overwritten(function_name, extract_pointing_parms['point_name'], extract_pointing_parms['overwrite'])
    
    
    pnt_dict = _extract_pointing(
        ms_name=extract_pointing_parms['ms_name'], 
        pnt_name=extract_pointing_parms['point_name'], 
        parallel=extract_pointing_parms['parallel']
    )
    
    # Calling this directly since it is so simple it doesn't need a "_create_{}" function.
    _write_meta_data(
        origin=function_name, 
        file_name="{name}/{ext}".format(name=extract_pointing_parms['point_name'], ext=".point_attr"), 
        input_dict=extract_pointing_parms
    )

    logger.info(f"[{function_name}]: Finished processing")
    point_dict = _load_point_file(file=extract_pointing_parms["point_name"], dask_load=True)
    
    pointing_mds = AstrohackPointFile(extract_pointing_parms['point_name'])
    pointing_mds._open()

    return pointing_mds

'''
def _check_extract_holog_params(fname,
                               ms_name,
                               point_name,
                               parallel,
                               overwrite):

    extract_holog_params = {"ms_name": ms_name, "holog_name": holog_name, "ddi_sel": ddi_sel, "point_name": point_name,
                           "data_column": data_column, "parallel": parallel, "overwrite": overwrite,
                           "reuse_point_zarr": reuse_point_zarr, "baseline_average_distance": baseline_average_distance,
                           "baseline_average_nearest": baseline_average_nearest}

    #### Parameter Checking ####
    logger = _get_astrohack_logger()
    parms_passed = True
    
    parms_passed = parms_passed and _check_parms(fname, extract_holog_params, 'ms_name', [str], default=None)

    base_name = _remove_suffix(ms_name, '.ms')
    parms_passed = parms_passed and _check_parms(fname, extract_holog_params,'holog_name', [str],
                                                 default=base_name+'.holog.zarr')

    point_base_name = _remove_suffix(extract_holog_params['holog_name'], '.holog.zarr')
    parms_passed = parms_passed and _check_parms(fname, extract_holog_params, 'point_name', [str],
                                                 default=point_base_name+'.point.zarr')
    
    parms_passed = parms_passed and _check_parms(fname, extract_holog_params, 'ddi_sel', [list, int],
                                                 list_acceptable_data_types=[int], default='all')
  
    #To Do: special function needed to check holog_obs_dict.
    parm_check = isinstance(holog_obs_dict,dict) or (holog_obs_dict is None)
    parms_passed = parms_passed and parm_check
    if not parm_check:
        logger.error(f'[{fname}]: Parameter holog_obs_dict must be of type {str(dict)}.')
        
    parms_passed = parms_passed and _check_parms(fname, extract_holog_params, 'baseline_average_distance',
                                                 [numbers.Number], default='all')
    
    parms_passed = parms_passed and _check_parms(fname, extract_holog_params, 'baseline_average_nearest', [int],
                                                 default='all')
    
    if (extract_holog_params['baseline_average_distance'] != 'all') and \
            (extract_holog_params['baseline_average_nearest'] != 'all'):
        logger.error(f'[{fname}]: baseline_average_distance: {str(baseline_average_distance)} and '
                     f'baseline_average_nearest: {str(baseline_average_distance)} can not both be specified.')
        parms_passed = False
 
    parms_passed = parms_passed and _check_parms(fname, extract_holog_params, 'data_column', [str],
                                                 default='CORRECTED_DATA')

    parms_passed = parms_passed and _check_parms(fname, extract_holog_params, 'parallel', [bool], default=False)
    
    parms_passed = parms_passed and _check_parms(fname, extract_holog_params, 'reuse_point_zarr', [bool], default=False)

    parms_passed = parms_passed and _check_parms(fname, extract_holog_params, 'overwrite', [bool],default=False)

    _parm_check_passed(fname, parms_passed)

    return extract_holog_params
    '''