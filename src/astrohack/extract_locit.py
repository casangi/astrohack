import numpy

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._dio import  check_if_file_will_be_overwritten, check_if_file_exists
from astrohack._utils._parm_utils._check_parms import _check_parms, _parm_check_passed
from astrohack._utils._tools import _remove_suffix
from astrohack._utils._extract_locit import _extract_antenna_data, _extract_pointing_data


def extract_locit(ms_name, locit_name=None, parallel=False, overwrite=False):
    """
    Extract Antenna position determination data from an MS and stores it in a locit output file.

    :param ms_name: Name of input measurement file name.
    :type ms_name: str
    :param locit_name: Name of *<locit_name>.locit.zarr* file to create. Defaults to measurement set name with *locit.zarr* extension.
    :type locit_name: str, optional
    :param parallel: Boolean for whether to process in parallel, defaults to False.
    :type parallel: bool, optional
    :param overwrite: Boolean for whether to overwrite current locit.zarr file, defaults to False.
    :type overwrite: bool, optional

    :return: Antenna position locit object.
    :rtype: AstrohackLocitFile

    .. _Description:

    **AstrohackLocitFile**


    **Additional Information**

    """
    logger = _get_astrohack_logger()

    fname = 'extract_locit'
    ######### Parameter Checking #########
    extract_locit_parms = _check_extract_locit_parms(fname, ms_name, locit_name, parallel, overwrite)
    input_params = extract_locit_parms.copy()

    check_if_file_exists(fname, extract_locit_parms['ms_name'])
    check_if_file_will_be_overwritten(fname, extract_locit_parms['locit_name'], extract_locit_parms['overwrite'])

    ant_dict = _extract_antenna_data(fname, extract_locit_parms['ms_name'])
    pnt_dict = _extract_pointing_data(fname, ms_name, ant_dict)


def _check_extract_locit_parms(fname, ms_name, locit_name, parallel, overwrite):

    extract_locit_parms = {"ms_name": ms_name, "locit_name": locit_name, "parallel": parallel, "overwrite": overwrite}

    #### Parameter Checking ####
    logger = _get_astrohack_logger()
    parms_passed = True

    parms_passed = parms_passed and _check_parms(fname, extract_locit_parms, 'ms_name', [str], default=None)

    base_name = _remove_suffix(ms_name, '.ms')
    parms_passed = parms_passed and _check_parms(fname, extract_locit_parms, 'locit_name', [str],
                                                 default=base_name+'.locit.zarr')

    parms_passed = parms_passed and _check_parms(fname, extract_locit_parms, 'parallel', [bool], default=False)

    parms_passed = parms_passed and _check_parms(fname, extract_locit_parms, 'overwrite', [bool],default=False)

    _parm_check_passed(fname, parms_passed)

    return extract_locit_parms
