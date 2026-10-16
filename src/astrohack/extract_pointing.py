import toolviper.utils.parameter

from astrohack.core.extract_pointing import (
    extract_pointing_chunk,
    extract_pointing_preprocessing,
    post_process_evaluation,
)
from astrohack.io.point_mds import AstrohackPointFile
from astrohack.utils.file import check_ms_exists, overwrite_file
from astrohack.utils.graph import create_and_execute_graph_from_dict
from astrohack.utils.text import get_default_file_name


@toolviper.utils.parameter.validate()
def extract_pointing(
    ms_name: str,
    point_name: str | None = None,
    parallel: bool = False,
    overwrite: bool = False,
) -> AstrohackPointFile | None:
    """ Extract pointing data from measurement set.  Creates holography output file.

    :param ms_name: Name of input measurement file name.
    :type ms_name: str

    :param point_name: Name of *<point_name>.point.zarr* file to create. Defaults to measurement set name with \
    *point.zarr* extension.
    :type point_name: str, optional

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
    point_name = get_default_file_name(ms_name, ".point.zarr", point_name)

    # Returns the current local variables in dictionary form
    extract_pointing_params = locals()

    input_params = extract_pointing_params.copy()

    check_ms_exists(ms_name)

    overwrite_file(
        extract_pointing_params["point_name"], extract_pointing_params["overwrite"]
    )
    ant_dist_matrix, looping_dict, pnt_params, mapping_state_ids = (
        extract_pointing_preprocessing(extract_pointing_params)
    )
    # Create mds file here
    point_mds = AstrohackPointFile.create_from_input_parameters(
        point_name, input_params
    )
    point_mds.root.attrs["mapping_state_ids"] = mapping_state_ids
    point_mds.root.attrs["baseline_dist_matrix"] = ant_dist_matrix
    point_mds.root.attrs["antenna_names"] = pnt_params.pop("antenna_names")
    point_mds.root.attrs["antenna_ids"] = pnt_params.pop("antenna_ids")
    point_mds.root.attrs["antenna_stations"] = pnt_params.pop("antenna_stations")
    point_mds.root.attrs["telescope_name"] = pnt_params.pop("telescope_name")

    executed_graph = create_and_execute_graph_from_dict(
        looping_dict=looping_dict,
        chunk_function=extract_pointing_chunk,
        param_dict=pnt_params,
        key_order=["ant"],
        output_mds=point_mds,
    )

    post_process_evaluation(extract_pointing_params["point_name"])

    if executed_graph:
        return point_mds
    else:
        return None
