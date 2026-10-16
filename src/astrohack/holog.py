import numpy as np
import toolviper.utils.parameter

from astrohack.core.holog import process_holog_chunk
from astrohack.io.dio import open_holog
from astrohack.io.image_mds import AstrohackImageFile
from astrohack.utils.file import overwrite_file
from astrohack.utils.graph import create_and_execute_graph_from_dict
from astrohack.utils.text import get_default_file_name


@toolviper.utils.parameter.validate()
def holog(
    holog_name: str,
    image_name: str | None = None,
    grid_size: int | np.ndarray | list[int] | None = None,
    cell_size: float | int | np.ndarray | list[int] | None = None,
    padding_factor: int = 10,
    grid_interpolation_mode: str = "gaussian",
    chan_average: bool = True,
    chan_tolerance_factor: float = 0.005,
    scan_average: bool = True,
    alma_osf_pad: str | None = None,
    ant: str | list[str] = "all",
    ddi: str | int | list[int] = "all",
    zernike_n_order: int = 4,
    phase_fit_engine: str = "perturbations",
    phase_fit_control: list[bool] | tuple[bool] = (True, True, True, True, True),
    to_stokes: bool = True,
    overwrite: bool = False,
    parallel: bool = False,
) -> AstrohackImageFile | None:
    """ Process holography data and derive aperture illumination pattern.

    :param holog_name: Name of holography .holog.zarr file to process.
    :type holog_name: str

    :param image_name: Defines the name of the output image name. If value is None, the name will be set to \
    <base_name>.image.zarr, defaults to None
    :type image_name: str, optional

    :param grid_size: Numpy array specifying the dimensions of the grid used in data gridding. If not specified \
    grid_size is calculated using POINTING_OFFSET in pointing table.
    :type grid_size: numpy.ndarray, dtype int, list optional

    :param cell_size: Size 2 array defining the cell size of each beam grid bin in radians. If not specified, the used \
    cell_size is the one given in the observation_summary of the input holog file.
    :type cell_size: numpy.ndarray, dtype float, list optional

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

    # Doing this here allows it to get captured by locals()
    image_name = get_default_file_name(holog_name, ".image.zarr", image_name)

    holog_params = locals()

    holog_mds = open_holog(holog_name)

    overwrite_file(image_name, overwrite)
    image_mds = AstrohackImageFile.create_from_input_parameters(
        image_name, holog_params
    )

    executed_graph = create_and_execute_graph_from_dict(
        looping_dict=holog_mds,
        chunk_function=process_holog_chunk,
        param_dict=holog_params,
        key_order=["ant", "ddi"],
        output_mds=image_mds,
    )

    if executed_graph:
        return image_mds
    else:
        return None
