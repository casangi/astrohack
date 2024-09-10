import pathlib
import toolviper.utils.parameter
import toolviper.utils.logger as logger

from typing import Union, List

from astrohack.core.combine import process_combine_chunk

from astrohack.utils.graph import compute_graph
from astrohack.utils.file import overwrite_file
from astrohack.utils.data import write_meta_data
from astrohack.utils.text import get_default_file_name

from astrohack.mds import AstrohackImageFile


@toolviper.utils.parameter.validate()
def combine(
        image_name: str,
        combine_name: str = None,
        ant: Union[str, List[str]] = "all",
        ddi: Union[int, List[int], str] = "all",
        weighted: bool = False,
        parallel: bool = False,
        overwrite: bool = False
) -> Union[AstrohackImageFile, None]:
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

    if combine_name is None:
        combine_name = get_default_file_name(input_file=image_name, output_type=".image.zarr")

    combine_params = locals()

    input_params = combine_params.copy()

    assert pathlib.Path(combine_params['image_name']).exists() is True, (
        logger.error(f'File {combine_params["image_name"]} does not exists.')
    )

    overwrite_file(combine_params['combine_name'], combine_params['overwrite'])

    image_mds = AstrohackImageFile(combine_params['image_name'])
    image_mds.open()

    combine_params['image_mds'] = image_mds
    image_attr = image_mds._meta_data

    if compute_graph(image_mds, process_combine_chunk, combine_params, ['ant'], parallel=parallel):
        logger.info("Finished processing")

        output_attr_file = "{name}/{ext}".format(name=combine_params['combine_name'], ext=".image_attr")
        write_meta_data(output_attr_file, image_attr)

        output_attr_file = "{name}/{ext}".format(name=combine_params['combine_name'], ext=".image_input")
        write_meta_data(output_attr_file, input_params)

        combine_mds = AstrohackImageFile(combine_params['combine_name'])
        combine_mds.open()
        return combine_mds
    else:
        logger.warning("No data to process")
        return None
