import os
import dask
import numpy as np

from astrohack._utils._combine import _combine_chunk
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._parm_utils._check_parms import _check_parms
from astrohack._utils._tools import _remove_suffix
from astrohack._utils._io import check_if_file_will_be_overwritten, check_if_file_exists, _write_meta_data
from astrohack._utils._dio import AstrohackImageFile


def combine_image_ddi(image_name, combine_name=None, ant_list=None, ddi_list=None, weighted=False, parallel=False,
                      overwrite=False):
    """Combine DDIs in a Holography image to increase SNR

    :param image_name: Input holography data file name. Accepted data format is the output from ``astrohack.holog.holog``
    :type image_name: str
    :param combine_name: Name of output file; File name will be appended with suffix *.combine.zarr*. Defaults to *basename* of input file plus holography panel file suffix.
    :type combine_name: str, optional
    :param ant_list: List of Antennae to be processed. None will use all antennae. Defaults to None
    :type ant_list: list, optional, ex. [ant_ea25 ... ant_ea04]
    :param ddi_list: List of DDIs to be combined. None will use all DDIs. Defaults to None.
    :type ddi_list: list, optional, ex. [ddi_0 ... ddi_N]
    :param weighted: Weight phases by the corresponding amplitudes
    :type weighted: bool, optional
    :param parallel: Run in parallel. Defaults to False.
    :type parallel: bool, optional
    :param overwrite: Overwrite files on disk. Defaults to False.
    :type overwrite: bool, optional

    :return: Holography image object.
    :rtype: AstrohackImageFile

    .. _Description:
    **AstrohackImageFile**

    Image object allows the user to access image data via compound dictionary keys with values, in order of depth, `ant` -> `ddi`. The image object also provides a `summary()` helper function to list available keys for each file. An outline of the image object structure is show below:

    .. parsed-literal::
        image_mds =
            {
            ant_0:{
                ddi_0: image_ds,
                 ⋮
                ddi_m: image_ds
            },
            ⋮
            ant_n: …
        }
    """
    logger = _get_astrohack_logger()

    combine_params = _check_combine_parms(image_name, combine_name, ant_list, ddi_list, weighted, parallel,  overwrite)
    input_params = combine_params.copy()

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

    output_attr_file = "{name}/{ext}".format(name=combine_params['combine_file'], ext=".image_attr")
    _write_meta_data('combine', output_attr_file, input_params)

    combine_mds = AstrohackImageFile(combine_chunk_params['combine_file'])
    combine_mds.open()

    return combine_mds


def _check_combine_parms(image_name, combine_name, ant_list, ddi_list, weighted, parallel,  overwrite):

    combine_params = {"image_file": image_name,
                      "combine_file": combine_name,
                      "ant_list": ant_list,
                      "ddi_list": ddi_list,
                      "weighted": weighted,
                      "parallel": parallel,
                      "overwrite": overwrite}

    #### Parameter Checking ####
    logger = _get_astrohack_logger()

    parms_passed = _check_parms(combine_params, 'image_file', [str], default=None)
    base_name = _remove_suffix(combine_params['image_file'], '.image.zarr')
    parms_passed = parms_passed and _check_parms(combine_params, 'combine_file', [str], default=base_name+'.combine.zarr')
    parms_passed = parms_passed and _check_parms(combine_params, 'ant_list', [list, np.ndarray],
                                                 list_acceptable_data_types=[str], default='all')
    parms_passed = parms_passed and _check_parms(combine_params, 'ddi_list', [list, np.ndarray],
                                                 list_acceptable_data_types=[str], default='all')
    parms_passed = parms_passed and _check_parms(combine_params, 'parallel', [bool], default=False)
    parms_passed = parms_passed and _check_parms(combine_params, 'weighted', [bool], default=False)
    parms_passed = parms_passed and _check_parms(combine_params, 'overwrite', [bool], default=False)

    if not parms_passed:
        logger.error("extract_combine parameter checking failed.")
        raise Exception("extract_combine parameter checking failed.")
    #### End Parameter Checking ####
    return combine_params
