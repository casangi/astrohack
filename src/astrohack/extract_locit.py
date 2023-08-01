from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._dio import _check_if_file_will_be_overwritten, _check_if_file_exists
from astrohack._utils._param_utils._check_parms import _check_parms, _parm_check_passed
from astrohack._utils._tools import _remove_suffix
from astrohack._utils._dio import _write_meta_data
from astrohack._utils._extract_locit import _extract_antenna_data, _extract_spectral_info
from astrohack._utils._extract_locit import _extract_source_and_telescope, _extract_antenna_phase_gains
from astrohack.mds import AstrohackLocitFile


def extract_locit(cal_table, locit_name=None, ant_id=None, ddi=None, overwrite=False):
    """
    Extract Antenna position determination data from an MS and stores it in a locit output file.

    :param cal_table: Name of input measurement file name.
    :type cal_table: str
    :param locit_name: Name of *<locit_name>.locit.zarr* file to create. Defaults to measurement set name with *locit.zarr* extension.
    :type locit_name: str, optional
    :param ant_id: List of antennae/antenna to be extracted, defaults to "all" when None, ex. ea25
    :type ant_id: list or str, optional
    :param ddi: List of ddis/ddi to be extracted, defaults to "all" when None, ex. 0
    :type ddi: list or int, optional
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
    extract_locit_parms = _check_extract_locit_parms(fname, cal_table, locit_name, ant_id, ddi, overwrite)
    attributes = extract_locit_parms.copy()

    _check_if_file_exists(extract_locit_parms['cal_table'])
    _check_if_file_will_be_overwritten(extract_locit_parms['locit_name'], extract_locit_parms['overwrite'])

    _extract_antenna_data(fname, extract_locit_parms)
    _extract_spectral_info(fname, extract_locit_parms)
    _extract_antenna_phase_gains(fname, extract_locit_parms)
    telescope_name, n_sources = _extract_source_and_telescope(fname, extract_locit_parms)

    attributes['telescope_name'] = telescope_name
    attributes['n_sources'] = n_sources
    attributes['reference_antenna'] = extract_locit_parms['reference_antenna']
    attributes['n_antennae'] = len(extract_locit_parms['ant_dict'])
    output_attr_file = "{name}/{ext}".format(name=extract_locit_parms['locit_name'], ext=".locit_attr")
    _write_meta_data(output_attr_file, attributes)

    logger.info(f"[{fname}]: Finished processing")
    locit_mds = AstrohackLocitFile(extract_locit_parms['locit_name'])
    locit_mds._open()
    return locit_mds


def _check_extract_locit_parms(fname, cal_table, locit_name, ant_id, ddi, overwrite):

    extract_locit_parms = {"cal_table": cal_table, "locit_name": locit_name, "ant": ant_id,
                           "ddi": ddi, "overwrite": overwrite}

    #### Parameter Checking ####
    logger = _get_astrohack_logger()
    parms_passed = True

    parms_passed = parms_passed and _check_parms(fname, extract_locit_parms, 'cal_table', [str], default=None)

    base_name = _remove_suffix(cal_table, '.cal')
    parms_passed = parms_passed and _check_parms(fname, extract_locit_parms, 'locit_name', [str],
                                                 default=base_name+'.locit.zarr')
    parms_passed = parms_passed and _check_parms(fname, extract_locit_parms, 'ant', [list, str],
                                                 list_acceptable_data_types=[str], default='all')
    parms_passed = parms_passed and _check_parms(fname, extract_locit_parms, 'ddi', [list, int],
                                                 list_acceptable_data_types=[int], default='all')
    parms_passed = parms_passed and _check_parms(fname, extract_locit_parms, 'overwrite', [bool], default=False)

    _parm_check_passed(fname, parms_passed)

    return extract_locit_parms
