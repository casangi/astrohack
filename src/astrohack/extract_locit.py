import pathlib
import toolviper.utils.parameter
import toolviper.utils.logger as logger

from typing import Union, List

from astrohack.utils.file import overwrite_file
from astrohack.utils.data import write_meta_data
from astrohack.core.extract_locit import extract_antenna_data, extract_spectral_info
from astrohack.core.extract_locit import extract_source_and_telescope, extract_antenna_phase_gains
from astrohack.utils.text import get_default_file_name
from astrohack.mds import AstrohackLocitFile


@toolviper.utils.parameter.validate()
def extract_locit(
        cal_table: str,
        locit_name: str = None,
        ant: Union[str, List[str]] = "all",
        ddi: Union[int, List[int]] = "all",
        overwrite: bool = False
):
    """
    Extract Antenna position determination data from an MS and stores it in a locit output file.

    :param cal_table: Name of input measurement file name.
    :type cal_table: str

    :param locit_name: Name of *<locit_name>.locit.zarr* file to create. Defaults to measurement set name \
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

    - `locit_mds = extract_locit("myphase.cal", locit_name="myphase.locit.zarr")` -> Extract phase calibration data for
       all antennas and all DDIs in "myphase.cal" into a locit file called "myphase.locit.zarr"
    - `locit_mds = extract_locit("myphase.cal", ant=["ea06", "ea03", "ea25"], DDI=0, overwrite=True) -> Extract phase
       calibration data for DDI 0 of antennas ea06, ea03 and ea25 in "myphase.cal" into a locit file called
       "myphase.locit.zarr" that will be overwritten if already present.
    """
    # Doing this here allows it to get captured by locals()
    if locit_name is None:
        locit_name = get_default_file_name(input_file=cal_table, output_type=".locit.zarr")

    extract_locit_params = locals()

    input_params = extract_locit_params.copy()
    attributes = extract_locit_params.copy()

    assert pathlib.Path(extract_locit_params['cal_table']).exists() is True, (
        logger.error(f'File {extract_locit_params["cal_table"]} does not exists.')
    )

    overwrite_file(extract_locit_params['locit_name'], extract_locit_params['overwrite'])

    extract_antenna_data(extract_locit_params)
    extract_spectral_info(extract_locit_params)
    extract_antenna_phase_gains(extract_locit_params)
    telescope_name, n_sources = extract_source_and_telescope(extract_locit_params)

    attributes['telescope_name'] = telescope_name
    attributes['n_sources'] = n_sources
    attributes['reference_antenna'] = extract_locit_params['reference_antenna']
    attributes['n_antennas'] = len(extract_locit_params['ant_dict'])

    output_attr_file = "{name}/{ext}".format(name=extract_locit_params['locit_name'], ext=".locit_input")
    write_meta_data(output_attr_file, input_params)

    output_attr_file = "{name}/{ext}".format(name=extract_locit_params['locit_name'], ext=".locit_attr")
    write_meta_data(output_attr_file, attributes)

    logger.info(f"Finished processing")
    locit_mds = AstrohackLocitFile(extract_locit_params['locit_name'])
    locit_mds.open()

    return locit_mds
