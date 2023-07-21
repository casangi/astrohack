import numpy as np

from astrohack._utils._combine import _combine_chunk
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._param_utils._check_parms import _check_parms, _parm_check_passed
from astrohack._utils._tools import _remove_suffix
from astrohack._utils._dio import _check_if_file_will_be_overwritten, _check_if_file_exists, _write_meta_data
from astrohack.mds import AstrohackImageFile
from astrohack._utils._dask_graph_tools import _dask_general_compute


def combine(image_name, combine_name=None, ant_id=None, ddi=None, weighted=False, parallel=False, overwrite=False):
    """Combine DDIs in a Holography image to increase SNR

    :param image_name: Input holography data file name. Accepted data format is the output from ``astrohack.holog.holog``
    :type image_name: str
    :param combine_name: Name of output file; File name will be appended with suffix *.combine.zarr*. Defaults to *basename* of input file plus holography panel file suffix.
    :type combine_name: str, optional
    :param ant_id: List of Antennae to be processed. None will use all antennae. Defaults to None, ex. ea25.
    :type ant_id: list or str, optional
    :param ddi: List of DDIs to be combined. None will use all DDIs. Defaults to None, ex. [0, ..., 8].
    :type ddi: list of int, optional
    :param weighted: Weight phases by the corresponding amplitudes.
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
    fname = 'combine'
    combine_params = _check_combine_parms(fname, image_name, combine_name, ant_id, ddi, weighted, parallel, overwrite)
    input_params = combine_params.copy()

    _check_if_file_exists(combine_params['image_file'])
    _check_if_file_will_be_overwritten(combine_params['combine_file'], combine_params['overwrite'])

    image_mds = AstrohackImageFile(combine_params['image_file'])
    image_mds._open()
    combine_params['image_mds'] = image_mds

    if _dask_general_compute(fname, image_mds, _combine_chunk, combine_params, ['ant'], parallel=parallel):
        logger.info(f"[{fname}]: Finished processing")
        output_attr_file = "{name}/{ext}".format(name=combine_params['combine_file'], ext=".image_attr")
        _write_meta_data(output_attr_file, input_params)
        combine_mds = AstrohackImageFile(combine_params['combine_file'])
        combine_mds._open()
        return combine_mds
    else:
        logger.warning(f"[{fname}]: No data to process")
        return None


def _check_combine_parms(fname, image_name, combine_name, ant_id, ddi_list, weighted, parallel, overwrite):

    combine_params = {"image_file": image_name,
                      "combine_file": combine_name,
                      "ant": ant_id,
                      "ddi": ddi_list,
                      "weighted": weighted,
                      "parallel": parallel,
                      "overwrite": overwrite}

    #### Parameter Checking ####
    parms_passed = _check_parms(fname, combine_params, 'image_file', [str], default=None)
    base_name = _remove_suffix(combine_params['image_file'], '.image.zarr')
    parms_passed = parms_passed and _check_parms(fname, combine_params, 'combine_file', [str],
                                                 default=base_name+'.combine.zarr')
    parms_passed = parms_passed and _check_parms(fname, combine_params, 'ant', [str, list],
                                                 list_acceptable_data_types=[str], default='all')
    parms_passed = parms_passed and _check_parms(fname, combine_params, 'ddi', [int, list],
                                                 list_acceptable_data_types=[int], default='all')
    parms_passed = parms_passed and _check_parms(fname, combine_params, 'parallel', [bool], default=False)
    parms_passed = parms_passed and _check_parms(fname, combine_params, 'weighted', [bool], default=False)
    parms_passed = parms_passed and _check_parms(fname, combine_params, 'overwrite', [bool], default=False)

    _parm_check_passed(fname, parms_passed)
    #### End Parameter Checking ####
    return combine_params
