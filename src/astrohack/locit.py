import toolviper.utils.parameter

from astrohack.core.locit import (
    locit_combined_chunk,
    locit_difference_chunk,
    locit_separated_chunk,
)
from astrohack.io.locit_mds import AstrohackLocitFile
from astrohack.io.position_mds import AstrohackPositionFile
from astrohack.utils.file import check_if_file_can_be_opened, overwrite_file
from astrohack.utils.graph import create_and_execute_graph_from_dict
from astrohack.utils.text import get_default_file_name
from astrohack.utils.tools import build_bad_scan_list


@toolviper.utils.parameter.validate()
def locit(
    locit_name: str,
    position_name: str | None = None,
    elevation_limit: float | int = 10.0,
    polarization: str = "both",
    fit_engine: str = "scipy",
    fit_kterm: bool = False,
    fit_delay_rate: bool = True,
    ant: str | list[str] = "all",
    ddi: str | int | list[int] = "all",
    combine_ddis: str = "simple",
    exclude_scans: int | list[int] | tuple[int] | None = None,
    parallel: bool = False,
    overwrite: bool = False,
):
    """
    Process phases from a locit file to obtain antenna position corrections which are stores in a position file.

    :param locit_name: Name of input *.locit.zarr file.
    :type locit_name: str

    :param position_name: Name of *<position_name>.position.zarr* file to create. Defaults to locit name \
    with *position.zarr* extension.
    :type position_name: str, optional

    :param elevation_limit: Lower elevation limit for excluding sources in degrees.
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
            ⋮
            ant_n: position_ds
        }

    **Additional Information**

    .. rubric:: Available fitting engines:

    For locit two fitting engines have been implemented, one the classic method used in AIPS is called here
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

    .. rubric:: Combining DDIs

    By default, (combine_ddis='simple') locit combines different DDIs so that there is a single position solution per
    antenna. The other options are, a solution for each of the DDIs for each antenna (combine_ddis='no') or combining
    two DDIs by computing the delays from the difference in phases between the two DDIs of different frequencies
    (combine_ddis='difference').

    combine_ddis='simple'     : Generates higher antenna position correction solutions of higher SNR as more data is \
                                used in each delay fit.
    combine_ddis='no'         : Useful for detecting systematic differences between different DDIs.
    combine_ddis='difference' : This method is useful for cases where phase wrapping may have occurred due to large \
                                delays.

    **Examples**

    - `position_mds = locit("myphase.locit.zarr", polarization='R', combine_ddis='simple')` -> Fit the phase delays in \
       "myphase.locit.zarr" for all antennas by combining the delays from all DDIs but using only the 'R' polarization.

    - `position_mds = locit("myphase.locit.zarr", combine_ddis='difference', elevation_limit=30.0)` -> Fit the phase \
       difference delays in "myphase.locit.zarr" for all antennas but only using sources above 30 degrees elevation.
    """

    # Doing this here allows it to get captured by locals()
    position_name = get_default_file_name(locit_name, ".position.zarr", position_name)
    exclude_scans = build_bad_scan_list(exclude_scans)
    locit_params = locals()

    input_params = locit_params.copy()
    attributes = locit_params.copy()

    check_if_file_can_be_opened(locit_params["locit_name"], "extract_locit", "0.10.1")

    overwrite_file(locit_params["position_name"], locit_params["overwrite"])

    locit_mds = AstrohackLocitFile(locit_params["locit_name"])
    locit_mds.open()

    if combine_ddis == "simple":
        function = locit_combined_chunk
        key_order = ["ant"]
        combined = True

    elif combine_ddis == "difference":
        function = locit_difference_chunk
        key_order = ["ant"]
        combined = True

    elif combine_ddis == "no":
        function = locit_separated_chunk
        key_order = ["ant", "ddi"]
        combined = False

    else:
        raise RuntimeError(
            "This part of the code should be unreacheable when parameter validation is online."
        )

    position_mds = AstrohackPositionFile.create_from_input_parameters(
        locit_params["position_name"], locit_params
    )
    position_mds.root.attrs.update(
        {
            "combined": combined,
            "telescope_name": locit_mds.root.attrs["telescope_name"],
            "reference_antenna": locit_mds.root.attrs["reference_antenna"],
            "source_dict": locit_mds.root.attrs["source_dict"],
            "combine_specifier": combine_ddis,
        }
    )

    executed_graph = create_and_execute_graph_from_dict(
        looping_dict=locit_mds,
        chunk_function=function,
        param_dict=locit_params,
        key_order=key_order,
        output_mds=position_mds,
    )
    if executed_graph:
        return position_mds
    else:
        return None
