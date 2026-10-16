import toolviper.utils.parameter

from astrohack.core.extract_locit import (
    extract_antenna_data,
    extract_source_and_telescope,
    extract_spectral_info,
)
from astrohack.core.fringefit_locit import (
    fringefit_locit_chunk,
    fringefit_locit_looping_dict,
)
from astrohack.io.position_mds import AstrohackPositionFile
from astrohack.utils.file import overwrite_file
from astrohack.utils.graph import create_and_execute_graph_from_dict
from astrohack.utils.text import get_default_file_name
from astrohack.utils.tools import build_bad_scan_list


@toolviper.utils.parameter.validate()
def fringefit_locit(
    fringefit_caltable: str,
    position_name: str | None = None,
    elevation_limit: float | int = 10.0,
    polarization: str = "both",
    fit_engine: str = "scipy",
    fit_kterm: bool = False,
    fit_delay_rate: bool = False,
    ant: str | list[str] = "all",
    ddi: str | int | list[int] = "all",
    exclude_scans: int | list[int] | tuple[int] | None = None,
    parallel: bool = True,
    overwrite: bool = False,
):
    """
    Extract delays from a fringefit cal table and fit them with a delay model to obtain rough antenna position corrections

    :param fringefit_caltable: fringefit cal table containing delays for all or most sources in a pointing observation.
    :type fringefit_caltable: str

    :param position_name: Name of *<position_name>.position.zarr* file to create. Defaults to fringefit cal table name \
    with *position.zarr* extension.
    :type position_name: str, optional

    :param elevation_limit: Lower elevation limit for excluding sources in degrees.
    :type elevation_limit: float, optional

    :param polarization: Which polarization to use R, L or both for circular systems, X, Y, or both for linear systems.
    :type polarization: str, optional

    :param fit_kterm: Fit antenna elevation axis offset term, defaults to False
    :type fit_kterm: bool, optional

    :param fit_delay_rate: Fit delay rate with time, defaults to False
    :type fit_delay_rate: bool, optional

    :param fit_engine: What engine to use on fitting, default is scipy
    :type fit_engine: str, optional

    :param ant: List of antennas/antenna to be processed, defaults to "all" when None, ex. ea25
    :type ant: list or str, optional

    :param ddi: List of ddis/ddi to be processed, defaults to "all" when None, ex. 0
    :type ddi: list or int, optional

    :param exclude_scans: List of scans in the fringefit cal table to be excluded from fitting, defaults to None
    :type exclude_scans: list, or int, optional

    :param parallel: Run in parallel. Defaults to False.
    :type parallel: bool, optional

    :param overwrite: Boolean for whether to overwrite current position.zarr file, defaults to False.
    :type overwrite: bool, optional

    :return: Antenna position object.
    :rtype: AstrohackPositionFile

    .. _Description:

    **AstrohackPositionFile**
    Position object allows the user to access position data via compound dictionary keys with values, in order of depth,
    `ant`. The position object also provides a `summary()` helper function to list available keys for each file.
    An outline of the position object structure is show below:

    .. parsed-literal::
        position_mds =
        {
            ant_0: position_ds,
            ⋮
            ant_n: position_ds
        }


    **Additional Information**

    .. rubric:: Available fitting engines:

    For fringefit_locit two fitting engines have been implemented, one the classic method used in AIPS is called here
    'linear algebra' and a newer more pythonic engine using scipy curve fitting capabilities, which we call
    scipy, more details below.

    * linear algebra: This fitting engine is based on the least square methods for solving linear systems, \
                      this engine is fast, about one order of magnitude faster than scipy,  but may fail to \
                      converge, also its uncertainties may be underestimated.

    * scipy: This fitting engine uses the well established scipy.optimize.curve_fit routine. This engine is \
             slower than the linear algebra engine, but it is more robust with better estimated uncertainties.

    .. rubric:: Choosing a polarization

    The position fit may be done on either polarization (R or L for the VLA, X or Y for ALMA) or for both polarizations
    at once. When choosing both polarizations we increase the robustness of the solution by doubling the amount of data
    fitted.
    """
    position_name = get_default_file_name(
        fringefit_caltable, ".position.zarr", position_name
    )
    exclude_scans = build_bad_scan_list(exclude_scans)
    locit_params = locals()

    input_params = locit_params.copy()

    overwrite_file(locit_params["position_name"], locit_params["overwrite"])

    position_mds = AstrohackPositionFile.create_from_input_parameters(
        locit_params["position_name"], input_params
    )

    extract_antenna_data(locit_params, position_mds)
    extract_source_and_telescope(locit_params, position_mds)

    ddi_dict = extract_spectral_info(locit_params)
    looping_dict, refant_name = fringefit_locit_looping_dict(
        locit_params, position_mds.root.attrs["full_antenna_list"]
    )
    locit_params["ddi_dict"] = ddi_dict

    position_mds.root.attrs.update(
        {
            "combined": True,
            "reference_antenna": refant_name,
            "combine_specifier": "fringefit",
        }
    )

    executed_graph = create_and_execute_graph_from_dict(
        looping_dict=looping_dict,
        chunk_function=fringefit_locit_chunk,
        param_dict=locit_params,
        key_order=["ant"],
        output_mds=position_mds,
    )
    if executed_graph:
        return position_mds
    else:
        return None
