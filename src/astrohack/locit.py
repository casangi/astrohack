import inspect

import auror.parameter
import skriba.logger

from astrohack._utils._dask_graph_tools import _dask_general_compute
from astrohack._utils._dio import _check_if_file_will_be_overwritten, _check_if_file_exists, _write_meta_data
from astrohack._utils._locit import _locit_separated_chunk, _locit_combined_chunk, _locit_difference_chunk
from astrohack._utils._tools import _remove_suffix
from astrohack.mds import AstrohackLocitFile, AstrohackPositionFile

CURRENT_FUNCTION = 0


@auror.parameter.validate(
    logger=skriba.logger.get_logger(logger_name="astrohack")
)
def locit(
        locit_name,
        position_name=None,
        elevation_limit=10.0,
        polarization='both',
        fit_engine='linear algebra',
        fit_kterm=False,
        fit_delay_rate=True,
        ant="all",
        ddi="all",
        combine_ddis='simple',
        parallel=False,
        overwrite=False
):
    """
    Extract Antenna position determination data from an MS and stores it in a locit output file.

    :param locit_name: Name of input *.locit.zarr file.
    :type locit_name: str
    :param position_name: Name of *<position_name>.position.zarr* file to create. Defaults to measurement set name with *position.zarr* extension.
    :type position_name: str, optional
    :param elevation_limit: Lower elevation limit for excluding sources in degrees
    :type elevation_limit: float, optional
    :param polarization: Which polarization to use R, L or both for circular systems, X, Y, or both for linear systems.
    :type polarization: str, optional
    :param fit_kterm: Fit antenna elevation axis offset term, defaults to False
    :type fit_kterm: bool, optional
    :param fit_delay_rate: Fit delay rate with time, defaults to True
    :type fit_delay_rate: bool, optional
    :param fit_engine: What engine to use on fitting, default is linear algebra
    :type fit_engine: str, optional
    :param ant: List of antennas/antenna to be processed, defaults to "all" when None, ex. ea25
    :type ant: list or str, optional
    :param ddi: List of ddis/ddi to be processed, defaults to "all" when None, ex. 0
    :type ddi: list or int, optional
    :param combine_ddis: Type of DDI combination, if desired, defaults to simple
    :type combine_ddis: str, optional
    :param parallel: Run in parallel. Defaults to False.
    :type parallel: bool, optional
    :param overwrite: Boolean for whether to overwrite current position.zarr file, defaults to False.
    :type overwrite: bool, optional

    :return: Antenna position object.
    :rtype: AstrohackPositionFile

    .. _Description:

    **AstrohackPositionFile**
    Position object allows the user to access position data via compound dictionary keys with values, in order of depth,
    `ant` -> `ddi`. The position object also provides a `summary()` helper function to list available keys for each file.
    An outline of the position object structure is show below:

    .. rubric:: combine_ddis = no:
    .. parsed-literal::
        position_mds =
        {
            ant_0:{
                ddi_0: position_ds,
                 ⋮
                ddi_m: position_ds
            },
            ⋮
            ant_n: …
        }

    .. rubric:: combine_ddis = ["simple", "difference"]:
    .. parsed-literal::
        position_mds =
        {
            ant_0: position_ds
            ant_n: position_ds
        }

    **Additional Information**

    .. rubric:: Available fitting engines:

    For locit two fitting engines have been implemented, one the classic method used in AIPS is called here
    'linear algebra' and a newer more pythonic engine using scipy curve fitting capabilities, which we call
    scipy, more details below.

    * linear algebra: This fitting engine is based on the least square methods for solving linear systems,
                      this engine is fast, about one order of magnitude faster than scipy,  but may fail to
                      converge, also its uncertainties may be underestimated.

    * scipy: This fitting engine uses the well established scipy.optimize.curve_fit routine. This engine is
             slower than the linear algebra engine, but it is more robust with better estimated uncertainties.

    .. rubric:: Choosing a polarization

    The position fit may be done on either polarization (R or L for the VLA, X or Y for ALMA) or for both polarizations
    at once. When choosing both polarizations we increase the robustness of the solution by doubling the amount of data
    fitted.

    .. rubric:: Combining DDIs

    By default, (combine_ddis='simple') locit combines different DDIs so that there is a single position solution per
    antenna. The other options are, a solution for each of the DDIs for each antenna (combine_ddis='no') or combining
    two DDIs by computing the delays from the difference in phases between the two DDIs of different frequencies
    (combine_ddis='difference').

    combine_ddis='simple'     : Generates higher antenna position correction solutions of higher SNR as more data is used
                                each delay fit.
    combine_ddis='no'         : Useful for detecting systematic differences between different DDIs.
    combine_ddis='difference' : This method is useful for cases where phase wrapping may have occurred due to large
                                delays.
    """
    locit_params = locals()
    logger = skriba.logger.get_logger(logger_name="astrohack")

    function_name = inspect.stack()[CURRENT_FUNCTION].function

    if position_name is None:
        logger.info('File not specified or doesn\'t exist. Creating ...')

        position_name = _remove_suffix(locit_name, '.locit.zarr') + '.position.zarr'
        locit_params['position_name'] = position_name

        logger.info('Extracting position name to {output}'.format(output=position_name))

    locit_params["fit_rate"] = fit_delay_rate

    input_params = locit_params.copy()
    attributes = locit_params.copy()

    _check_if_file_exists(locit_params['locit_name'])
    _check_if_file_will_be_overwritten(locit_params['position_name'], locit_params['overwrite'])

    locit_mds = AstrohackLocitFile(locit_params['locit_name'])
    locit_mds._open()

    locit_params['ant_info'] = locit_mds['ant_info']
    locit_params['obs_info'] = locit_mds['obs_info']

    attributes['telescope_name'] = locit_mds._meta_data['telescope_name']
    attributes['reference_antenna'] = locit_mds._meta_data['reference_antenna']

    if combine_ddis == 'simple':
        function = _locit_combined_chunk
        key_order = ['ant']
    elif combine_ddis == 'difference':
        function = _locit_difference_chunk
        key_order = ['ant']
    else:
        function = _locit_separated_chunk
        key_order = ['ant', 'ddi']

    if _dask_general_compute(function_name, locit_mds, function, locit_params, key_order, parallel=parallel):
        logger.info(f"[{function_name}]: Finished processing")

        output_attr_file = "{name}/{ext}".format(name=locit_params['position_name'], ext=".position_attr")
        _write_meta_data(output_attr_file, attributes)

        output_attr_file = "{name}/{ext}".format(name=locit_params['position_name'], ext=".position_input")
        _write_meta_data(output_attr_file, input_params)

        position_mds = AstrohackPositionFile(locit_params['position_name'])
        position_mds._open()

        return position_mds

    else:
        logger.warning(f"[{function_name}]: No data to process")
        return None
