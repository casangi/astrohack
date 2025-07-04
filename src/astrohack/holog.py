import json
import pathlib
import numpy as np

import toolviper.utils.logger as logger
import toolviper

from numbers import Number
from typing import List, Union, NewType, Tuple

from astrohack.utils.graph import compute_graph
from astrohack.utils.file import overwrite_file, check_if_file_can_be_opened
from astrohack.utils.data import write_meta_data
from astrohack.core.holog import process_holog_chunk
from astrohack.utils.text import get_default_file_name
from astrohack.mds import AstrohackImageFile


Array = NewType("Array", Union[np.array, List[int], List[float]])


@toolviper.utils.parameter.validate()
def holog(
    holog_name: str,
    grid_size: Union[int, Array, List] = None,
    cell_size: Union[float, Array, List] = None,
    image_name: str = None,
    padding_factor: int = 10,
    grid_interpolation_mode: str = "gaussian",
    chan_average: bool = True,
    chan_tolerance_factor: float = 0.005,
    scan_average: bool = True,
    alma_osf_pad: str = None,
    ant: Union[str, List[str]] = "all",
    ddi: Union[int, List[int]] = "all",
    zernike_n_order: int = 4,
    phase_fit_engine: str = "perturbations",
    phase_fit_control: Union[List[bool], Tuple[bool]] = (True, True, True, True, True),
    to_stokes: bool = True,
    overwrite: bool = False,
    parallel: bool = False,
) -> Union[AstrohackImageFile, None]:
    """ Process holography data and derive aperture illumination pattern.

    :param holog_name: Name of holography .holog.zarr file to process.
    :type holog_name: str

    :param grid_size: Numpy array specifying the dimensions of the grid used in data gridding. If not specified \
    grid_size is calculated using POINTING_OFFSET in pointing table.
    :type grid_size: numpy.ndarray, dtype int, list optional

    :param cell_size: Numpy array defining the cell size of each grid bin. If not specified cell_size is calculated \
    using POINTING_OFFSET in pointing table.
    :type cell_size: numpy.ndarray, dtype float, list optional

    :param image_name: Defines the name of the output image name. If value is None, the name will be set to \
    <base_name>.image.zarr, defaults to None
    :type image_name: str, optional

    :param padding_factor: Padding factor applied to beam grid before computing the fast-fourier transform. The default\
     has been set for operation on most systems. The user should be aware of memory constraints before increasing this\
      parameter significantly., defaults to 10
    :type padding_factor: int, optional

    :param parallel: Run in parallel with Dask or in serial., defaults to False
    :type parallel: bool, optional

    :param grid_interpolation_mode: Method of interpolation used when gridding data. For modes 'linear', 'nearest' and \
    'cubic' this is done using the `scipy.interpolate.griddata` method. For more information see `scipy.interpolate \
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata>`_.\
    The remaining mode 'gaussian' convolves the visibilities with a gaussian kernel with a FWHM equal HPBW for the \
    primary beam main lobe at the given frequency, this is slower than `scipy.interpolate.griddata` but better at\
    preserving the small scales variations in the beam. Defaults to "gaussian".
    :type grid_interpolation_mode: str, optional. Available options: {"gaussian", "linear", "nearest", "cubic"}

    :param chan_average: Boolean dictating whether the channel average is computed and written to the output holog \
    file., defaults to True
    :type chan_average: bool, optional

    :param chan_tolerance_factor: Tolerance used in channel averaging to determine the number of primary beam \
    channels., defaults to 0.005
    :type chan_tolerance_factor: float, optional

    :param scan_average: Boolean dictating whether averaging is done over scan., defaults to True
    :type scan_average: bool, optional

    :param alma_osf_pad: Pad on which the antenna was poitioned at the ALMA OSF (only relevant for ALMA near field \
    holographies).
    :type alma_osf_pad: str, optional

    :param ant: List of antennas/antenna to be processed, defaults to "all" when None, ex. ea25
    :type ant: list or str, optional

    :param ddi: List of ddi to be processed, defaults to "all" when None, ex. 0
    :type ddi: list or int, optional

    :param zernike_n_order: Maximal N order for the Zernike Polynomials to be fitted to the aperture data.
    :type zernike_n_order: int, optional

    :param phase_fit_engine: Choose between the two available phase fitting engines, "perturbations" which assumes \
    cassegrain optics and "zernike" which makes no assumption about the optical system but may overfit the aperture \
    phase, default is "perturbations".
    :type phase_fit_engine: str, optional.

    :param phase_fit_control: Controls which type of optical perturbations are to be fitted when phase_fit_engine is \
    set to "perturbations".

        Available phase fit controls:

        - [0]: pointing offset; 
        - [1]: focus xy offsets; 
        - [2]: focus z offset; 
        - [3]: subreflector tilt (off by default except for VLA and VLBA)
        - [4]: cassegrain offset

    :type phase_fit_control: bool array, optional

    :param to_stokes: Dictates whether polarization is computed according to stokes values., defaults to True
    :type to_stokes: bool, optional

    :param overwrite: Overwrite existing files on disk, defaults to False
    :type overwrite: bool, optional

    :return: Holography image object.
    :rtype: AstrohackImageFile
    
    .. _Description:
    **AstrohackImageFile**

    Image object allows the user to access image data via compound dictionary keys with values, in order of depth,\
     `ant` -> `ddi`. The image object also provides a `summary()` helper function to list available keys for each file.\
      An outline of the image object structure is show below:

    .. parsed-literal::
        image_mds = 
            {
            ant_0:{
                ddi_0: image_ds,
                 ⋮               
                ddi_m: image_ds
            },
            ⋮
            ant_n: …
        }

    **Example Usage**
    
    .. parsed-literal::
        from astrohack.holog import holog

        holog(
            holog_name="astrohack_observation.holog.zarr", 
            padding_factor=50, 
            grid_interpolation_mode='linear',
            chan_average = True,
            scan_average = True,
            ant='ea25',
            overwrite=True,
            parallel=True
        )

    """

    check_if_file_can_be_opened(holog_name, "0.7.2")

    # Doing this here allows it to get captured by locals()
    if image_name is None:
        image_name = get_default_file_name(
            input_file=holog_name, output_type=".image.zarr"
        )

    holog_params = locals()

    input_params = holog_params.copy()
    assert pathlib.Path(holog_params["holog_name"]).exists() is True, logger.error(
        f'File {holog_params["holog_name"]} does not exists.'
    )

    overwrite_file(holog_params["image_name"], holog_params["overwrite"])

    json_data = "/".join((holog_params["holog_name"], ".holog_json"))

    with open(json_data, "r") as json_file:
        holog_json = json.load(json_file)

    if compute_graph(
        holog_json, process_holog_chunk, holog_params, ["ant", "ddi"], parallel=parallel
    ):

        output_attr_file = "{name}/{ext}".format(
            name=holog_params["image_name"], ext=".image_attr"
        )
        write_meta_data(output_attr_file, holog_params)

        output_attr_file = "{name}/{ext}".format(
            name=holog_params["image_name"], ext=".image_input"
        )
        write_meta_data(output_attr_file, input_params)

        image_mds = AstrohackImageFile(holog_params["image_name"])
        image_mds.open()

        logger.info("Finished processing")

        return image_mds

    else:
        logger.warning("No data to process")
        return None


def _convert_gridding_parameter(
    gridding_parameter: Union[List, Array], reflect_on_axis=False
) -> np.ndarray:
    if isinstance(gridding_parameter, Number):
        gridding_parameter = np.array(
            [np.power(-1, reflect_on_axis) * gridding_parameter, gridding_parameter]
        )

    elif isinstance(gridding_parameter, list):
        gridding_parameter = np.array(gridding_parameter)

    elif isinstance(gridding_parameter, np.ndarray):
        pass

    else:
        logger.error(
            "Unknown dtype for gridding parameter: {}".format(gridding_parameter)
        )

    return gridding_parameter
