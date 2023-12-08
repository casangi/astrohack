import json
import numpy as np

import skriba.logger
import auror.parameter

from typing import List, Union, NewType

from astrohack._utils._dask_graph_tools import _dask_general_compute
from astrohack._utils._dio import _check_if_file_exists
from astrohack._utils._dio import _check_if_file_will_be_overwritten
from astrohack._utils._dio import _read_meta_data
from astrohack._utils._dio import _write_meta_data
from astrohack._utils._holog import _holog_chunk
from astrohack._utils._tools import get_default_file_name
from astrohack.mds import AstrohackImageFile

Array = NewType("Array", Union[np.array, List[int], List[float]])

@auror.parameter.validate(
    logger=skriba.logger.get_logger(logger_name="astrohack")
)
def holog(
        holog_name: str,
        grid_size: Union[int, Array] = None,
        cell_size: Union[int, Array] = None,
        image_name: str = None,
        padding_factor: int = 50,
        grid_interpolation_mode: str = "linear",
        chan_average: bool = True,
        chan_tolerance_factor: float = 0.005,
        scan_average: bool = True,
        ant: Union[str, List[str]] = "all",
        ddi: Union[int, List[int]] = "all",
        to_stokes: bool = True,
        apply_mask: bool = True,
        phase_fit: bool = True,
        overwrite: bool = False,
        parallel: bool = False
) -> AstrohackImageFile:
    """ Process holography data and derive aperture illumination pattern.

    :param holog_name: Name of holography .holog.zarr file to process.
    :type holog_name: str

    :param grid_size: Numpy array specifying the dimensions of the grid used in data gridding. If not specified \
    grid_size is calculated using POINTING_OFFSET in pointing table.
    :type grid_size: numpy.ndarray, dtype int, optional

    :param cell_size: Numpy array defining the cell size of each grid bin. If not specified cell_size is calculated \
    using POINTING_OFFSET in pointing table.
    :type cell_size: numpy.ndarray, dtype float, optional

    :param image_name: Defines the name of the output image name. If value is None, the name will be set to \
    <base_name>.image.zarr, defaults to None
    :type image_name: str, optional

    :param padding_factor: Padding factor applied to beam grid before computing the fast-fourier transform. The default\
     has been set for operation on most systems. The user should be aware of memory constraints before increasing this\
      parameter significantly., defaults to 50
    :type padding_factor: int, optional

    :param parallel: Run in parallel with Dask or in serial., defaults to False
    :type parallel: bool, optional
    :param grid_interpolation_mode: Method of interpolation used when gridding data. This is done using the \
    `scipy.interpolate.griddata` method. For more information on the interpolation see `scipy.interpolate \
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata>`_,\
     defaults to "linear"
    :type grid_interpolation_mode: str, optional. Available options: {"linear", "nearest", "cubic"}
    :param chan_average: Boolean dictating whether the channel average is computed and written to the output holog \
    file., defaults to True
    :type chan_average: bool, optional
    :param chan_tolerance_factor: Tolerance used in channel averaging to determine the number of primary beam \
    channels., defaults to 0.005
    :type chan_tolerance_factor: float, optional
    :param scan_average: Boolean dicating whether averagin is done over scan., defaults to True
    :type scan_average: bool, optional
    :param ant: List of antennas/antenna to be processed, defaults to "all" when None, ex. ea25
    :type ant: list or str, optional
    :param ddi: List of ddis/ddi to be processed, defaults to "all" when None, ex. 0
    :type ddi: list or int, optional
    :param to_stokes: Dictates whether polarization is computed according to stokes values., defaults to True
    :type to_stokes: bool, optional
    :param apply_mask: If True applies a mask to the aperture setting values outside of the aperture to zero., defaults\
     to True
    :type apply_mask: bool, optional
    :param phase_fit: If a boolean array is given each element controls one aspect of phase fitting. defaults to True.
        
        Phase fitting:
        
        - [0]: pointing offset; 
        - [1]: focus xy offsets; 
        - [2]: focus z offset; 
        - [3]: subreflector tilt (off by default except for VLA and VLBA)
        - [4]: cassegrain offset

    :type phase_fit: bool, optional

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
            ant_id=['ea25'],
            overwrite=True,
            parallel=True
        )

    """
    # Doing this here allows it to get captured by locals()
    if image_name is None:
        image_name = get_default_file_name(input_file=holog_name, output_type=".image.zarr")

    holog_params = locals()

    logger = skriba.logger.get_logger(logger_name="astrohack")

    input_params = holog_params.copy()
    _check_if_file_exists(holog_params['holog_name'])

    _check_if_file_will_be_overwritten(holog_params['image_name'], holog_params['overwrite'])

    json_data = "/".join((holog_params['holog_name'], ".holog_json"))

    with open(json_data, "r") as json_file:
        holog_json = json.load(json_file)

    meta_data = _read_meta_data(holog_params['holog_name'] + '/.holog_attr')

    if holog_params["cell_size"] is None:
        if meta_data['cell_size'] is None:
            logger.error(
                "Cell size meta data not found. There was likely an issue with the holography data extraction. Fix\
                 extract data or provide cell_size as argument.")
            logger.error("There was an error, see log above for more info.")

            return None

        else:
            cell_size = np.array([-meta_data["cell_size"], meta_data["cell_size"]])
            holog_params["cell_size"] = cell_size

    if holog_params["grid_size"] is None:
        if meta_data['n_pix'] is None:
            logger.error(
                "Grid size meta data not found. There was likely an issue with the holography data extraction. Fix \
                extract data or provide grid_size as argument.")
            logger.error("There was an error, see log above for more info.")

            return None

        else:
            n_pix = int(np.sqrt(meta_data["n_pix"]))
            grid_size = np.array([n_pix, n_pix])
            holog_params["grid_size"] = grid_size

    logger.info('Cell size: {cell_size}, Grid size {grid_size}'.format(
        cell_size=holog_params["cell_size"],
        grid_size=holog_params["grid_size"]
    ))

    json_data = {
        "cell_size": holog_params["cell_size"].tolist(),
        "grid_size": holog_params["grid_size"].tolist()
    }

    with open(".holog_diagnostic.json", "w") as out_file:
        json.dump(json_data, out_file)

    try:
        if _dask_general_compute(
                holog_json,
                _holog_chunk,
                holog_params,
                ['ant', 'ddi'],
                parallel=parallel
        ):

            output_attr_file = "{name}/{ext}".format(name=holog_params['image_name'], ext=".image_attr")
            _write_meta_data(output_attr_file, holog_params)

            output_attr_file = "{name}/{ext}".format(name=holog_params['image_name'], ext=".image_input")
            _write_meta_data(output_attr_file, input_params)

            image_mds = AstrohackImageFile(holog_params['image_name'])
            image_mds.open()

            logger.info('Finished processing')

            return image_mds

        else:
            logger.warning("No data to process")
            return None

    except Exception as error:
        logger.error("There was an error, see log above for more info :: {error}".format(error=error))
