import graphviper.utils.parameter
import graphviper.utils.logger as logger

from astrohack._utils._dio import _load_point_file, _check_if_file_will_be_overwritten, _check_if_file_exists
from astrohack._utils._dio import _write_meta_data
from astrohack._utils._extract_point import _extract_pointing
from astrohack._utils._tools import get_default_file_name
from astrohack.mds import AstrohackPointFile

from typing import List, Union


@graphviper.utils.parameter.validate(
    external_logger=logger.get_logger(logger_name="astrohack")
)
def extract_pointing(
        ms_name: str,
        point_name: str = None,
        exclude: Union[str, List[str]] = None,
        parallel: bool = False,
        overwrite: bool = False,
) -> AstrohackPointFile:
    """ Extract pointing data from measurement set.  Creates holography output file.

    :param ms_name: Name of input measurement file name.
    :type ms_name: str

    :param point_name: Name of *<point_name>.point.zarr* file to create. Defaults to measurement set name with \
    *point.zarr* extension.
    :type point_name: str, optional

    :param exclude: Name of antenna to exclude from extraction.
    :type exclude: list, optional

    :param parallel: Boolean for whether to process in parallel. Defaults to False
    :type parallel: bool, optional

    :param overwrite: Overwrite pointing file on disk, defaults to False
    :type overwrite: bool, optional

    :return: Holography point object.
    :rtype: AstrohackPointFile

    .. _Description:

    **Example Usage**
    In this case, the pointing_name is the file name to be created after extraction.

    .. parsed-literal::
        from astrohack.extract_pointing import extract_pointing

        extract_pointing(
            ms_name="astrohack_observation.ms",
            point_name="astrohack_observation.point.zarr"
        )

    **AstrohackPointFile**

    Point object allows the user to access point data via dictionary keys with values `ant`. The point object also
    provides a `summary()` helper function to list available keys for each file.


    """
    # Doing this here allows it to get captured by locals()
    if point_name is None:
        point_name = get_default_file_name(input_file=ms_name, output_type=".point.zarr")

    # Returns the current local variables in dictionary form
    extract_pointing_params = locals()

    input_params = extract_pointing_params.copy()
    _check_if_file_exists(extract_pointing_params['ms_name'])
    _check_if_file_will_be_overwritten(extract_pointing_params['point_name'], extract_pointing_params['overwrite'])

    pnt_dict = _extract_pointing(
        ms_name=extract_pointing_params['ms_name'],
        pnt_name=extract_pointing_params['point_name'],
        exclude=extract_pointing_params['exclude'],
        parallel=extract_pointing_params['parallel']
    )

    # Calling this directly since it is so simple it doesn't need a "_create_{}" function.
    _write_meta_data(
        file_name="{name}/{ext}".format(name=extract_pointing_params['point_name'], ext=".point_input"),
        input_dict=input_params
    )

    logger.info(f"Finished processing")
    point_dict = _load_point_file(file=extract_pointing_params["point_name"], dask_load=True)

    pointing_mds = AstrohackPointFile(extract_pointing_params['point_name'])
    pointing_mds.open()

    return pointing_mds
