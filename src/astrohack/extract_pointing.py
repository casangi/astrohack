import os
import dask
import json
import copy
import numpy as np
import numbers
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


def extract_pointing(
    ms_name,
    point_name=None,
    parallel=False,
    overwrite=False,
):
    """
    Extract holography and optionally pointing data, from measurement set. Creates holography output file.

    :param ms_name: Name of input measurement file name.
    :type ms_name: str
    :param holog_obs_dict: The *holog_obs_dict* describes which scan and antenna data to extract from the measurement set. As detailed below, this compound dictionary also includes important meta data needed for preprocessing and extraction of the holography data from the measurement set. If not specified holog_obs_dict will be generated. For auto generation of the holog_obs_dict the assumtion is made that the same antanna beam is not mapped twice in a row (alternating sets of antennas is fine).
    :type holog_obs_dict: dict, optional
    :param ddi:  DDI(s) that should be extracted from the measurement set. Defaults to all DDI's in the ms.
    :type ddi: int numpy.ndarray | int list, optional
    :param baseline_average_distance: To increase the signal to noise for a mapping antenna mutiple reference antennas can be used. The baseline_average_distance is the acceptable distance (in meters) between a mapping antenna and a reference antenna. The baseline_average_distance is only used if the holog_obs_dict is not specified. If no distance is specified all reference antennas will be used. baseline_average_distance and baseline_average_nearest can not be used together.
    :type baseline_average_distance: float, optional
    :param baseline_average_nearest: To increase the signal to noise for a mapping antenna mutiple reference antennas can be used. The baseline_average_nearest is the number of nearest reference antennas to use. The baseline_average_nearest is only used if the holog_obs_dict is not specified.  baseline_average_distance and baseline_average_nearest can not be used together.
    :type baseline_average_nearest: int, optional
    :param holog_name: Name of *<holog_name>.holog.zarr* file to create. Defaults to measurement set name with *holog.zarr* extension.
    :type holog_name: str, optional
    :param point_name: Name of *<point_name>.point.zarr* file to create. Defaults to measurement set name with *point.zarr* extension.
    :type point_name: str, optional
    :param data_column: Determines the data column to pull from the measurement set. Defaults to "CORRECTED_DATA".
    :type data_column: str, optional, ex. DATA, CORRECTED_DATA
    :param parallel: Boolean for whether to process in parallel, defaults to False.
    :type parallel: bool, optional
    :param overwrite: Boolean for whether to overwrite current holog.zarr and point.zarr files, defaults to False.
    :type overwrite: bool, optional

    :return: Holography point object.
    :rtype: AstrohackPointFile

    .. _Description:

    **AstrohackPointFile**

    Holog object allows the user to access holog data via compound dictionary keys with values, in order of depth, `ddi` -> `map` -> `ant`. The holog object also provides a `summary()` helper function to list available keys for each file. An outline of the holog object structure is show below:

    .. parsed-literal::
        holog_mds = 
        {
            ddi_0:{
                map_0:{
                 ant_0: holog_ds,
                          ⋮
                 ant_n: holog_ds
                },
                ⋮
                map_p: …
            },
            ⋮
            ddi_m: …
        }

    **Additional Information**

        This function extracts the holography related information from the given measurement file. The data is restructured into an astrohack file format and saved into a file in the form of *<holog_name>.holog.zarr*. The extension *.holog.zarr* is used for all holography files. In addition, the pointing information is recorded into a holography file of format *<pointing_name>.point.zarr*. The extension *.point.zarr* is used for all holography pointing files. 

        **holog_obs_dict[holog_mapping_id] (dict):**
        *holog_mapping_id* is a unique, arbitrary, user-defined integer assigned to the data that describes a single complete mapping of the beam.
        
        .. rubric:: This is needed for two reasons:
        * A complete mapping of the beam can be done over more than one scan (for example the VLA data). 
        * A measurement set can contain more than one mapping of the beam (for example the ALMA data).
    
        **holog_obs_dict[holog_mapping_id][scans] (int | numpy.ndarray | list):**
        All the scans in the measurement set the *holog_mapping_id*.
    
        **holog_obs_dict[holog_mapping_id][ant] (dict):**
        The dictionary keys are the mapping antenna names and the values a list of the reference antennas. See example below.

        The below example shows how the *holog_obs_description* dictionary should be laid out. For each *holog_mapping_id* the relevant scans 
        and antennas must be provided. For the `ant` key, an entry is required for each mapping antenna and the accompanying reference antenna(s).
    
        .. parsed-literal::
            holog_obs_description = {
                'map_0' :{
                    'scans':[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                    'ant':{
                        'DA44':[
                            'DV02', 'DV03', 'DV04', 
                            'DV11', 'DV12', 'DV13', 
                            'DV14', 'DV15', 'DV16', 
                            'DV17', 'DV18', 'DV19', 
                            'DV20', 'DV21', 'DV22', 
                            'DV23', 'DV24', 'DV25'
                        ]
                    }
                }
            }

    """
    extract_pointing_parms = locals()

    logger = _get_astrohack_logger()
    
    fname = 'extract_pointing'
    ######### Parameter Checking #########
#    extract_pointing_parms = _check_extract_pointing_parms(fname,
#                                                     ms_name,
#                                                     point_name,
#                                                     parallel,
#                                                     overwrite)

       
    #input_params = extract_pointing_parms.copy()
    
    check_if_file_exists(fname, extract_pointing_parms['ms_name'])
    check_if_file_will_be_overwritten(fname, extract_pointing_parms['point_name'], extract_pointing_parms['overwrite'])
    
    
    pnt_dict = _extract_pointing(
        ms_name=extract_pointing_parms['ms_name'], 
        pnt_name=extract_pointing_parms['point_name'], 
        parallel=extract_pointing_parms['parallel']
    )
    
    # Calling this directly since it is so simple it doesn't need a "_create_{}" function.
    _write_meta_data(
        origin='extract_pointing', 
        file_name="{name}/{ext}".format(name=extract_pointing_parms['point_name'], ext=".point_attr"), 
        input_dict=extract_pointing_parms
    )

    logger.info(f"[{fname}]: Finished processing")
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