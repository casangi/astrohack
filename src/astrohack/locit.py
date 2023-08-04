from astrohack.mds import AstrohackLocitFile, AstrohackPositionFile
from astrohack._utils._locit import _locit_chunk
from astrohack._utils._dio import _check_if_file_will_be_overwritten, _check_if_file_exists, _write_meta_data
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._param_utils._check_parms import _check_parms, _parm_check_passed
from astrohack._utils._tools import _remove_suffix
from astrohack._utils._dask_graph_tools import _dask_general_compute


def locit(locit_name, position_name=None, elevation_limit=10.0, polarization='both', fit_engine='linear algebra',
          fit_kterm=False, fit_slope=True, ant_id=None, ddi=None, parallel=False, overwrite=False):
    """
    Extract Antenna position determination data from an MS and stores it in a locit output file.

    :param locit_name: Name of input measurement file name.
    :type locit_name: str
    :param position_name: Name of *<position_name>.position.zarr* file to create. Defaults to measurement set name with *position.zarr* extension.
    :type position_name: str, optional
    :param elevation_limit: Lower elevation limit for excluding sources in degrees
    :type elevation_limit: float, optional
    :param polarization: Which polarization to use R, L or both for circular systems, X, Y, or both for linear systems
    :type polarization: str, optional
    :param fit_kterm: Fit antenna focus offset term, defaults to False
    :type fit_kterm: bool, optional
    :param fit_slope: Fit phase slope with time, defaults to True
    :type fit_slope: bool, optional
    :param fit_engine: What engine to use on fitting, default is linear algebra
    :type fit_engine: str, optional
    :param ant_id: List of antennae/antenna to be processed, defaults to "all" when None, ex. ea25
    :type ant_id: list or str, optional
    :param ddi: List of ddis/ddi to be processed, defaults to "all" when None, ex. 0
    :type ddi: list or int, optional
    :param parallel: Run in parallel. Defaults to False.
    :type parallel: bool, optional
    :param overwrite: Boolean for whether to overwrite current position.zarr file, defaults to False.
    :type overwrite: bool, optional

    :return: Antenna position object.
    :rtype: AstrohackPositionFile

    .. _Description:

    **AstrohackPositionFile**


    **Additional Information**

    .. rubric:: Available fitting engines:

    For locit two fitting engines have been implemented, one the classic method used in AIPS is called here
    'linear algebra' and a newer more pythonic engine using scipy curve fitting capabilities, which we call
    scipy, more details below.

    * linear algebra: This fitting engine is based on the least square methods for solving linear systems,
                      this engine is fast, about one order of magnitude faster than scipy,  but may fail to
                      converge, also its uncertainties may be underestimated.

    * scipy: This fitting engine uses the well estabilished scipy.optimize.curve_fit routine. This engine is
             slower than the linear algebra engine, but it is more robust with better estimated uncertainties.


    """
    logger = _get_astrohack_logger()

    fname = 'locit'
    ######### Parameter Checking #########
    locit_parms = _check_locit_parms(fname, locit_name, position_name, elevation_limit, polarization, fit_engine,
                                     fit_kterm, fit_slope, ant_id, ddi, parallel, overwrite)
    attributes = locit_parms.copy()

    _check_if_file_exists(locit_parms['locit_name'])
    _check_if_file_will_be_overwritten(locit_parms['position_name'], locit_parms['overwrite'])
    locit_mds = AstrohackLocitFile(locit_parms['locit_name'])
    locit_mds._open()
    locit_parms['ant_info'] = locit_mds['ant_info']
    locit_parms['obs_info'] = locit_mds['obs_info']

    if _dask_general_compute(fname, locit_mds, _locit_chunk, locit_parms, ['ant', 'ddi'], parallel=parallel):
        logger.info(f"[{fname}]: Finished processing")
        output_attr_file = "{name}/{ext}".format(name=locit_parms['position_name'], ext=".position_attr")
        _write_meta_data(output_attr_file, attributes)
        position_mds = AstrohackPositionFile(locit_parms['position_name'])
        position_mds._open()
        return position_mds
    else:
        logger.warning(f"[{fname}]: No data to process")
        return None


def _check_locit_parms(fname, locit_name, position_name, elevation_limit, polarization, fit_engine, fit_kterm,
                       fit_slope, ant_id, ddi, parallel, overwrite):

    locit_parms = {"locit_name": locit_name, "position_name": position_name, "elevation_limit": elevation_limit,
                   "fit_engine": fit_engine, "polarization": polarization, "fit_kterm": fit_kterm,
                   "fit_slope": fit_slope, "ant": ant_id, "ddi": ddi, "parallel": parallel, "overwrite": overwrite}

    #### Parameter Checking ####
    logger = _get_astrohack_logger()
    parms_passed = True

    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'locit_name', [str], default=None)

    base_name = _remove_suffix(locit_name, '.locit.zarr')
    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'position_name', [str],
                                                 default=base_name+'.position.zarr')
    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'elevation_limit', [float],
                                                 acceptable_range=[0, 90], default=10)
    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'polarization', [str],
                                                 acceptable_data=['X', 'Y', 'R', 'L', 'both'], default='I')
    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'fit_engine', [str],
                                                 acceptable_data=['linear algebra', 'scipy'], default='linear algebra')
    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'fit_kterm', [bool], default=False)
    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'fit_slope', [bool], default=True)
    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'ant', [list, str],
                                                 list_acceptable_data_types=[str], default='all')
    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'ddi', [list, int],
                                                 list_acceptable_data_types=[int], default='all')
    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'parallel', [bool], default=False)
    parms_passed = parms_passed and _check_parms(fname, locit_parms, 'overwrite', [bool], default=False)

    _parm_check_passed(fname, parms_passed)

    return locit_parms
