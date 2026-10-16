import shutil

import toolviper.utils.parameter
from toolviper.utils import logger

from astrohack.core.extract_locit import (
    extract_antenna_data,
    extract_antenna_phase_gains,
    extract_source_and_telescope,
    extract_spectral_info,
)
from astrohack.io.locit_mds import AstrohackLocitFile
from astrohack.utils.file import check_ms_exists, overwrite_file
from astrohack.utils.text import get_default_file_name


@toolviper.utils.parameter.validate()
def extract_locit(
    cal_table: str,
    locit_name: str | None = None,
    ant: str | list[str] = "all",
    ddi: str | int | list[int] = "all",
    overwrite: bool = False,
):
    """
    Extract phases from a gaincal cal table and stores it in a locit output file.

    :param cal_table: Name of input measurement file name.
    :type cal_table: str

    :param locit_name: Name of *<locit_name>.locit.zarr* file to create. Defaults to cal table name \
    with *locit.zarr* extension.
    :type locit_name: str, optional

    :param ant: List of antennas/antenna to be extracted, defaults to "all" when None, ex. ea25
    :type ant: list or str, optional

    :param ddi: List of ddi to be extracted, defaults to "all" when None, ex. 0
    :type ddi: list or int, optional

    :param overwrite: Boolean for whether to overwrite current locit.zarr file, defaults to False.
    :type overwrite: bool, optional

    :return: Antenna position locit object.
    :rtype: AstrohackLocitFile

    .. _Description:

    extract_locit is a tool to extract the phase gains from a calibration table in an order that is suited for
    parallelized processing by locit. Along with the calibration data antenna position and source position information
    is extracted from the calibration table for use in the fitting process in locit.

    **AstrohackLocitFile**
    The locit object allows the user to access calibration data via compound dictionary keys with values, in order of
    depth, `ant` -> `ddi`. The locit object also provides a `summary()` helper function to list available keys for each
    file. An outline of the locit object structure is show below:

    .. parsed-literal::
        locit_mds =
        {
            ant_0:{
                ddi_0: locit_ds,
                 ⋮
                ddi_m: locit_ds
            },
            ⋮
            ant_n: …
        }

    **Examples**

    - `locit_mds = extract_locit("myphase.cal", locit_name="myphase.locit.zarr")` -> Extract phase calibration data for \
       all antennas and all DDIs in "myphase.cal" into a locit file called "myphase.locit.zarr"
    - `locit_mds = extract_locit("myphase.cal", ant=["ea06", "ea03", "ea25"], DDI=0, overwrite=True)` -> Extract phase \
       calibration data for DDI 0 of antennas ea06, ea03 and ea25 in "myphase.cal" into a locit file called \
       "myphase.locit.zarr" that will be overwritten if already present.
    """
    # Doing this here allows it to get captured by locals()
    locit_name = get_default_file_name(cal_table, ".locit.zarr", locit_name)
    extract_locit_params = locals()

    input_params = extract_locit_params.copy()
    attributes = extract_locit_params.copy()

    check_ms_exists(cal_table)

    overwrite_file(
        extract_locit_params["locit_name"], extract_locit_params["overwrite"]
    )

    ddi_dict = extract_spectral_info(extract_locit_params)

    locit_mds = AstrohackLocitFile.create_from_input_parameters(
        extract_locit_params["locit_name"], extract_locit_params
    )
    extract_antenna_data(extract_locit_params, locit_mds)
    extract_source_and_telescope(extract_locit_params, locit_mds)

    locit_mds.write(mode="a")
    locit_mds.consolidate(["ant"])

    extract_antenna_phase_gains(extract_locit_params, ddi_dict, locit_mds)

    if len(locit_mds.keys()) == 0:
        shutil.rmtree(extract_locit_params["locit_name"])
        logger.warning("No data produced")
        return None
    else:
        locit_mds.write(mode="a")
        return locit_mds
