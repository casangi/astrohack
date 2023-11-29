import inspect

import auror.parameter
import skriba.logger
from astrohack._utils._dio import _check_if_file_will_be_overwritten, _check_if_file_exists
from astrohack._utils._dio import _write_meta_data
from astrohack._utils._extract_locit import _extract_antenna_data, _extract_spectral_info
from astrohack._utils._extract_locit import _extract_source_and_telescope, _extract_antenna_phase_gains
from astrohack.mds import AstrohackLocitFile

CURRENT_FUNCTION = 0


@auror.parameter.validate(
    logger=skriba.logger.get_logger(logger_name="astrohack")
)
def extract_locit(
        cal_table,
        locit_name=None,
        ant="all",
        ddi="all",
        overwrite=False
):
    """
    Extract Antenna position determination data from an MS and stores it in a locit output file.

    :param cal_table: Name of input measurement file name.
    :type cal_table: str
    :param locit_name: Name of *<locit_name>.locit.zarr* file to create. Defaults to measurement set name with *locit.zarr* extension.
    :type locit_name: str, optional
    :param ant: List of antennas/antenna to be extracted, defaults to "all" when None, ex. ea25
    :type ant: list or str, optional
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
    extract_locit_params = locals()
    logger = skriba.logger.get_logger(logger_name="astrohack")

    function_name = inspect.stack()[CURRENT_FUNCTION].function

    if locit_name is None:
        logger.info('File not specified or doesn\'t exist. Creating ...')

        locit_name = cal_table + '.locit.zarr'
        extract_locit_params['locit_name'] = locit_name

        logger.info('Extracting locit name to {output}'.format(output=locit_name))

    input_params = extract_locit_params.copy()
    attributes = extract_locit_params.copy()

    _check_if_file_exists(extract_locit_params['cal_table'])
    _check_if_file_will_be_overwritten(extract_locit_params['locit_name'], extract_locit_params['overwrite'])

    _extract_antenna_data(function_name, extract_locit_params)
    _extract_spectral_info(function_name, extract_locit_params)
    _extract_antenna_phase_gains(function_name, extract_locit_params)
    telescope_name, n_sources = _extract_source_and_telescope(function_name, extract_locit_params)

    attributes['telescope_name'] = telescope_name
    attributes['n_sources'] = n_sources
    attributes['reference_antenna'] = extract_locit_params['reference_antenna']
    attributes['n_antennas'] = len(extract_locit_params['ant_dict'])

    output_attr_file = "{name}/{ext}".format(name=extract_locit_params['locit_name'], ext=".locit_input")
    _write_meta_data(output_attr_file, input_params)

    output_attr_file = "{name}/{ext}".format(name=extract_locit_params['locit_name'], ext=".locit_attr")
    _write_meta_data(output_attr_file, attributes)

    logger.info(f"[{function_name}]: Finished processing")
    locit_mds = AstrohackLocitFile(extract_locit_params['locit_name'])
    locit_mds._open()

    return locit_mds
