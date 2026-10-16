import toolviper.utils.parameter

from astrohack.core.combine import process_combine_chunk
from astrohack.io.dio import open_image
from astrohack.io.image_mds import AstrohackImageFile
from astrohack.utils.file import overwrite_file
from astrohack.utils.graph import create_and_execute_graph_from_dict
from astrohack.utils.text import get_default_file_name


@toolviper.utils.parameter.validate()
def combine(
    image_name: str,
    combine_name: str | None = None,
    ant: str | list[str] = "all",
    ddi: int | list[int] | str = "all",
    weighted: bool = False,
    parallel: bool = False,
    overwrite: bool = False,
) -> AstrohackImageFile | None:
    """Combine DDIs in a Holography image to increase SNR

    :param image_name: Input holography data file name. Accepted data format is the output from ``astrohack.holog.holog``
    :type image_name: str

    :param combine_name: Name of output file; File name will be appended with suffix *.combine.zarr*. Defaults to \
    *basename* of input file plus holography panel file suffix.
    :type combine_name: str, optional

    :param ant: List of antennas to be processed. None will use all antennas. Defaults to None, ex. ea25.
    :type ant: list or str, optional

    :param ddi: List of DDIs to be combined. None will use all DDIs. Defaults to None, ex. [0, ..., 8].
    :type ddi: list of int, optional

    :param weighted: Weight phases by the corresponding amplitudes.
    :type weighted: bool, optional

    :param parallel: Run in parallel. Defaults to False.
    :type parallel: bool, optional

    :param overwrite: Overwrite files on disk. Defaults to False.
    :type overwrite: bool, optional

    :return: Holography image object.
    :rtype: AstrohackImageFile

    .. _Description:
    **combine**

    Combine combines the amplitude and corrected_phase members of the selected DDIs in the input image file. Currently, \
    combine only supports the combination of these two quantities to avoid long regridding times. Hence, the output \
    image file (.combine.zarr file name) contains the combined amplitude and corrected_phase images, but the aperture \
    and beam images present in this file will be those present in the DDI with the lowest frequency.

    **AstrohackImageFile**

    Image object allows the user to access image data via compound dictionary keys with values, in order of depth, \
    `ant` -> `ddi`. The image object produced by combine is special because it will always contain a single DDI.\
    The image object also provides a `summary()` helper function to list available keys for each file.\
     An outline of the image object structure when produced by combine is show below:

    .. parsed-literal::
        image_mds =
            {
            ant_0:{
                ddi_n: image_ds,
            },
            ⋮
            ant_n: …
        }

    **Example Usage**

    .. parsed-literal::
        from astrohack.combine import combine

        combine(
            "astrohack_obs.image.zarr",
            ant = "ea25"
            weight = False
        )
    """

    combine_name = get_default_file_name(image_name, ".combine.zarr", combine_name)

    combine_params = locals()

    overwrite_file(combine_params["combine_name"], combine_params["overwrite"])

    image_mds = open_image(image_name)

    combine_mds = AstrohackImageFile.create_from_input_parameters(
        combine_name, combine_params
    )

    executed_graph = create_and_execute_graph_from_dict(
        looping_dict=image_mds,
        chunk_function=process_combine_chunk,
        param_dict=combine_params,
        key_order=["ant"],
        output_mds=combine_mds,
    )

    if executed_graph:
        combine_mds.write(mode="a")
        return combine_mds
    else:
        return None
