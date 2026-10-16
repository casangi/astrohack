import pathlib
from copy import deepcopy

import toolviper.utils.logger as logger
import toolviper.utils.parameter

from astrohack.core.extract_holog import (
    extract_holog_preprocessing,
    process_extract_holog_chunk,
)
from astrohack.core.holog_obs_dict import HologObsDict
from astrohack.io.dio import open_holog, open_pointing
from astrohack.io.holog_mds import AstrohackHologFile
from astrohack.utils.file import check_ms_exists, overwrite_file
from astrohack.utils.graph import create_and_execute_graph_from_dict
from astrohack.utils.text import get_default_file_name


@toolviper.utils.parameter.validate(add_data_type=HologObsDict)
def extract_holog(
    ms_name: str,
    point_name: str,
    holog_name: str | None = None,
    holog_obs_dict: HologObsDict | None = None,
    ant: str | list[str] = "all",
    ddi: int | list[int] | str = "all",
    baseline_average_distance: float | int | str = "all",
    baseline_average_nearest: int | str = 1,
    exclude_antennas: list[str] | str | None = None,
    data_column: str = "CORRECTED_DATA",
    time_smoothing_interval: float | int | None = None,
    pointing_interpolation_method: str = "linear",
    parallel: bool = False,
    overwrite: bool = False,
    append: bool = False,
) -> AstrohackHologFile | None:
    """
    Extract holography and optionally pointing data, from measurement set. Creates holography output file.

    :param ms_name: Name of input measurement file name.
    :type ms_name: str

    :param point_name: Name of *<point_name>.point.zarr* file to use. This is must be provided.
    :type holog_name: str

    :param holog_name: Name of *<holog_name>.holog.zarr* file to create. Defaults to measurement set name with \
    *holog.zarr* extension.
    :type holog_name: str, optional

    :param holog_obs_dict: The *holog_obs_dict* describes which scan and antenna data to extract from the measurement \
    set. As detailed below, this compound dictionary also includes important metadata needed for preprocessing and \
    extraction of the holography data from the measurement set. If not specified holog_obs_dict will be generated. \
    For auto generation of the holog_obs_dict the assumption is made that the same antenna beam is not mapped twice in \
    a row (alternating sets of antennas is fine). If the holog_obs_dict is specified, the ddi input is ignored. The \
    user can self generate this dictionary using `generate_holog_obs_dict`.
    :type holog_obs_dict: dict, optional

    :param ant: Antennas that should be extracted from the measurement set. Defaults to all mapping antennas in the ms.
    :type ant: str | list[str], optional

    :param ddi:  DDI(s) that should be extracted from the measurement set. Defaults to all DDI's in the ms.
    :type ddi: int numpy.ndarray | int list, optional

    :param baseline_average_distance: To increase the signal-to-noise for a mapping antenna multiple reference \
    antennas can be used. The baseline_average_distance is the acceptable distance (in meters) between a mapping \
    antenna and a reference antenna. The baseline_average_distance is only used if the holog_obs_dict is not \
    specified. If no distance is specified all reference antennas will be used. baseline_average_distance and \
    baseline_average_nearest can not be used together.
    :type baseline_average_distance: float, optional

    :param baseline_average_nearest: To increase the signal-to-noise for a mapping antenna multiple reference antennas \
    can be used. The baseline_average_nearest is the number of nearest reference antennas to use. The \
    baseline_average_nearest is only used if the holog_obs_dict is not specified.  baseline_average_distance and \
    baseline_average_nearest can not be used together.
    :type baseline_average_nearest: int, optional

    :param exclude_antennas: If an antenna is given for exclusion it will not be processed as a reference or a \
    mapping antenna. This can be used to exclude antennas that have bad data for whatever reason. Default is None, \
    meaning no antenna is excluded.
    :type exclude_antennas: str | list, optional

    :param data_column: Determines the data column to pull from the measurement set. Defaults to "CORRECTED_DATA".
    :type data_column: str, optional, ex. DATA, CORRECTED_DATA

    :param time_smoothing_interval: Determines the time smoothing interval, set to the integration time when None.
    :type time_smoothing_interval: float, optional

    :param pointing_interpolation_method: Determines which algorithm to use to interpolate pointing in time, two \
    options are available: 'linear' (Faster but brittle) and 'gaussian' (slower but robust), default is 'linear'.
    :type pointing_interpolation_method: str, optional

    :param parallel: Boolean for whether to process in parallel, defaults to False.
    :type parallel: bool, optional

    :param overwrite: Boolean for whether to overwrite current holog.zarr and point.zarr files, defaults to False.
    :type overwrite: bool, optional

    :param append: Should data be appended to an existing holog file on disk, append and overwrite cannot be both true\
    defaults to False.
    :type append: bool, optional

    :return: Holography holog object.
    :rtype: AstrohackHologFile

    .. _Description:

    **AstrohackHologFile**

    Holog object allows the user to access holog data via compound dictionary keys with values, in order of depth,
    `ddi` -> `map` -> `ant`. The holog object also provides a `summary()` helper function to list available keys for
    each file. An outline of the holog object structure is show below:

    .. parsed-literal::
        holog_mds = 
        {
            ddi_0:{
                map_0:{
                 ant_0: holog_ds,
                          ⋮
                 ant_n: holog_ds
                },
                ⋮
                map_p: …
            },
            ⋮
            ddi_m: …
        }

    **Example Usage** In this case the pointing file has already been created. In addition, the appropriate
    data_column value nees to be set for the type of measurement set data you are extracting.

    .. parsed-literal::
        from astrohack.extract_holog import extract_holog

        holog_mds = extract_holog(
            ms_name="astrohack_observation.ms",
            point_name="astrohack_observation.point.ms",
            holog_name="astrohack_observation.holog.ms",
            data_column='CORRECTED_DATA',
            parallel=True,
            overwrite=True
        )

    **Additional Information**

        This function extracts the holography related information from the given measurement file. The data is
        restructured into an astrohack file format and saved into a file in the form of *<holog_name>.holog.zarr*.
        The extension *.holog.zarr* is used for all holography files. In addition, the pointing information is
        recorded into a holography file of format *<pointing_name>.point.zarr*. The extension *.point.zarr* is used
        for all holography pointing files.

        **holog_obs_dict[holog_mapping_id] (dict):** *holog_mapping_id* is a unique, arbitrary, user-defined integer
        assigned to the data that describes a single complete mapping of the beam.
        
        .. rubric:: This is needed for two reasons:
        * A complete mapping of the beam can be done over more than one scan (for example the VLA data). 
        * A measurement set can contain more than one mapping of the beam (for example the ALMA data).
    
        **holog_obs_dict[holog_mapping_id][scans] (int | numpy.ndarray | list):**
        All the scans in the measurement set the *holog_mapping_id*.
    
        **holog_obs_dict[holog_mapping_id][ant] (dict):** The dictionary keys are the mapping antenna names and the
        values a list of the reference antennas. See example below.

        The below example shows how the *holog_obs_description* dictionary should be laid out. For each
        *holog_mapping_id* the relevant scans and antennas must be provided. For the `ant` key, an entry is required
        for each mapping antenna and the accompanying reference antenna(s).
    
        .. parsed-literal::
            holog_obs_description = {
                'map_0' :{
                    'scans':[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                    'ant':{
                        'DA44':[
                            'DV02', 'DV03', 'DV04', 
                            'DV11', 'DV12', 'DV13', 
                            'DV14', 'DV15', 'DV16', 
                            'DV17', 'DV18', 'DV19', 
                            'DV20', 'DV21', 'DV22', 
                            'DV23', 'DV24', 'DV25'
                        ]
                    }
                }
            }

    """
    # This copy here ensures that the user space holog_obs_dict given to extract_holog is not modified during execution
    if holog_obs_dict is not None:
        holog_obs_dict = deepcopy(holog_obs_dict)

    # Doing this here allows it to get captured by locals()
    holog_name = get_default_file_name(ms_name, ".holog.zarr", holog_name)
    extract_holog_params = locals()

    # VVV This is a temporary fix waiting for the implementation of a mapping parameter
    extract_holog_params["map"] = "all"

    check_ms_exists(ms_name)

    if append and overwrite:
        raise RuntimeError("Append and overwrite cannot be both set to True.")

    pnt_mds = open_pointing(point_name)

    looping_dict, used_holog_obs_dict = extract_holog_preprocessing(
        extract_holog_params, pnt_mds
    )
    # print_dict_types(looping_dict, show_values=True)

    if append:
        holog_mds = open_holog(holog_name)
        holog_mds.root.attrs["input_parameters"] = extract_holog_params.copy()
    else:
        overwrite_file(
            extract_holog_params["holog_name"], extract_holog_params["overwrite"]
        )
        holog_mds = AstrohackHologFile.create_from_input_parameters(
            holog_name, extract_holog_params.copy()
        )

    extract_holog_params["pnt_mds"] = pnt_mds
    holog_mds.root.attrs["holog_obs_dict"] = used_holog_obs_dict

    executed_graph = create_and_execute_graph_from_dict(
        looping_dict=looping_dict,
        chunk_function=process_extract_holog_chunk,
        param_dict=extract_holog_params,
        key_order=["ddi", "map"],
        output_mds=holog_mds,
    )

    if executed_graph:
        return holog_mds
    else:
        return None


def generate_holog_obs_dict(
    point_name: str,
    baseline_average_distance: str = "all",
    baseline_average_nearest: str = "all",
    exclude_antennas: list[str] | str | None = None,
    parallel: bool = False,
) -> HologObsDict:
    """
    Generate holography observation dictionary, from measurement set..

    :param point_name: Name of *<point_name>.point.zarr* file to use.
    :type point_name: str, optional

    :param baseline_average_distance: To increase the signal-to-noise for a mapping antenna multiple reference
    antennas can be used. The baseline_average_distance is the acceptable distance between a mapping antenna and a
    reference antenna. The baseline_average_distance is only used if the holog_obs_dict is not specified. If no
    distance is specified all reference antennas will be used. baseline_average_distance and baseline_average_nearest
    can not be used together.
    :type baseline_average_distance: float, optional

    :param baseline_average_nearest: To increase the signal-to-noise for a mapping antenna multiple reference antennas
    can be used. The baseline_average_nearest is the number of nearest reference antennas to use. The
    baseline_average_nearest is only used if the holog_obs_dict is not specified.  baseline_average_distance and
    baseline_average_nearest can not be used together.
    :type baseline_average_nearest: int, optional

    :param exclude_antennas: If an antenna is given for exclusion it will not be processed as a reference or a \
    mapping antenna. This can be used to exclude antennas that have bad data for whatever reason. Default is None, \
    meaning no antenna is excluded.
    :type exclude_antennas: str | list, optional

    :param parallel: Boolean for whether to process in parallel. Defaults to False
    :type parallel: bool, optional

    :return: holog observation dictionary
    :rtype: HologObsDict

    .. _Description:

    **AstrohackHologFile**

    Holog object allows the user to access holog data via compound dictionary keys with values, in order of depth,
    `ddi` -> `map` -> `ant`. The holog object also provides a `summary()` helper function to list available keys for
    each file. An outline of the holog object structure is show below:

    .. parsed-literal::
        holog_mds =
        {
            ddi_0:{
                map_0:{
                 ant_0: holog_ds,
                          ⋮
                 ant_n: holog_ds
                },
                ⋮
                map_p: …
            },
            ⋮
            ddi_m: …
        }

    **Example Usage**
    In this case the pointing file has already been created.

    .. parsed-literal::
        from astrohack.extract_holog import generate_holog_obs_dict

        holog_obs_obj = generate_holog_obs_dict(
            ms_name="astrohack_observation.ms",
            point_name="astrohack_observation.point.zarr"
        )

    **Additional Information**

        **holog_obs_dict[holog_mapping_id] (dict):** *holog_mapping_id* is a unique, arbitrary, user-defined integer
        assigned to the data that describes a single complete mapping of the beam.

        .. rubric:: This is needed for two reasons:
        * A complete mapping of the beam can be done over more than one scan (for example the VLA data).
        * A measurement set can contain more than one mapping of the beam (for example the ALMA data).

        **holog_obs_dict[holog_mapping_id][scans] (int | numpy.ndarray | list):**
        All the scans in the measurement set the *holog_mapping_id*.

        **holog_obs_dict[holog_mapping_id][ant] (dict):** The dictionary keys are the mapping antenna names and the
        values a list of the reference antennas. See example below.

        The below example shows how the *holog_obs_description* dictionary should be laid out. For each
        *holog_mapping_id* the relevant scans and antennas must be provided. For the `ant` key, an entry is required
        for each mapping antenna and the accompanying reference antenna(s).

        .. parsed-literal::
            holog_obs_description = {
                'map_0' :{
                    'scans':[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                    'ant':{
                        'DA44':[
                            'DV02', 'DV03', 'DV04',
                            'DV11', 'DV12', 'DV13',
                            'DV14', 'DV15', 'DV16',
                            'DV17', 'DV18', 'DV19',
                            'DV20', 'DV21', 'DV22',
                            'DV23', 'DV24', 'DV25'
                        ]
                    }
                }
            }

    """
    holog_dict_params = locals()

    assert pathlib.Path(point_name).exists() is True, logger.error(
        f"File {point_name} does not exists."
    )

    pnt_mds = open_pointing(holog_dict_params["point_name"])

    holog_obs_dict = HologObsDict.create_from_pointing_file(
        pnt_mds=pnt_mds,
        exclude_antennas=holog_dict_params["exclude_antennas"],
        baseline_average_distance=holog_dict_params["baseline_average_distance"],
        baseline_average_nearest=holog_dict_params["baseline_average_nearest"],
    )

    return holog_obs_dict
