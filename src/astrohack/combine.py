import os
import dask
import numpy as np

from astrohack._utils._combine import _combine_chunk
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._parm_utils._check_parms import _check_parms
from astrohack._utils._utils import _remove_suffix
from astrohack._utils._io import check_if_file_will_be_overwritten, check_if_file_exists
from astrohack._utils._dio import AstrohackImageFile


def combine_holog_ddi(image_name, combine_name, ant_list, ddi_list, parallel, overwrite):
    logger = _get_astrohack_logger()

    combine_params = _check_combine_parms(image_name, combine_name, ant_list, ddi_list, parallel,  overwrite)

    check_if_file_exists(combine_params['image_file'])
    check_if_file_will_be_overwritten(combine_params['combine_file'], combine_params['overwrite'])

    if combine_params['ant_list'] == 'all':
        antennae = os.listdir(combine_params['image_file'])
    else:
        antennae = combine_params['ant_list']

    delayed_list = []
    combine_chunk_params = combine_params
    for antenna in antennae:
        combine_chunk_params['antenna'] = antenna
        logger.info("Processing ant_id: " + str(antenna))
        if parallel:
            delayed_list.append(dask.delayed(_combine_chunk)(dask.delayed(combine_chunk_params)))
        else:
            _combine_chunk(combine_chunk_params)

    if parallel:
        dask.compute(delayed_list)

    combine_mds = AstrohackImageFile(combine_chunk_params['combine_file'])
    combine_mds.open()

    return combine_mds


def _check_combine_parms(image_name, combine_name, ant_list, ddi_list, parallel,  overwrite):

    combine_params = {"image_file": image_name,
                      "combine_file": combine_name,
                      "ant_list": ant_list,
                      "ddi_list": ddi_list,
                      "parallel": parallel,
                      "overwrite": overwrite}

    #### Parameter Checking ####
    logger = _get_astrohack_logger()

    parms_passed = _check_parms(combine_params, 'image_file', [str], default=None)
    base_name = _remove_suffix(combine_params['combine_file'], '.image.zarr')
    parms_passed = parms_passed and _check_parms(combine_params, 'combine_name', [str], default=base_name+'.combine.zarr')
    parms_passed = parms_passed and _check_parms(combine_params, 'ant_list', [list, np.ndarray],
                                                 list_acceptable_data_types=[str], default='all')
    parms_passed = parms_passed and _check_parms(combine_params, 'ddi_list', [list, np.ndarray],
                                                 list_acceptable_data_types=[str], default='all')
    parms_passed = parms_passed and _check_parms(combine_params, 'parallel', [bool], default=False)
    parms_passed = parms_passed and _check_parms(combine_params, 'overwrite', [bool], default=False)

    if not parms_passed:
        logger.error("extract_combine parameter checking failed.")
        raise Exception("extract_combine parameter checking failed.")
    #### End Parameter Checking ####
    return combine_params
