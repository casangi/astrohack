import json
import inspect
import numbers

import numpy as np

from astrohack._utils._holog import _holog_chunk
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._param_utils._check_parms import _check_parms, _parm_check_passed
from astrohack._utils._tools import _remove_suffix
from astrohack._utils._dio import _check_if_file_will_be_overwritten, _check_if_file_exists, _read_meta_data, _write_meta_data
from astrohack.mds import AstrohackImageFile
from astrohack._utils._dask_graph_tools import _dask_general_compute

CURRENT_FUNCTION=0

def holog(
    holog_name, 
    grid_size=None, 
    cell_size=None, 
    image_name=None, 
    padding_factor=50,
    grid_interpolation_mode="linear", 
    chan_average=True, 
    chan_tolerance_factor=0.005, 
    scan_average=True,
    ant_id=None, 
    ddi=None, 
    to_stokes=True, 
    apply_mask=True, 
    phase_fit=True, 
    overwrite=False, 
    parallel=False
):
    """ Process holography data and derive aperture illumination pattern.

    :param holog_name: Name of holography .holog.zarr file to process.
    :type holog_name: str
    :param grid_size: Numpy array specifying the dimensions of the grid used in data gridding. If not specified grid_size is calculated using POINTING_OFFSET in pointing table.
    :type grid_size: numpy.ndarray, dtype int, optional
    :param cell_size: Numpy array defining the cell size of each grid bin. If not specified cell_size is calculated using POINTING_OFFSET in pointing table.
    :type cell_size: numpy.ndarray, dtype float, optional
    :param image_name: Defines the name of the output image name. If value is None, the name will be set to <base_name>.image.zarr, defaults to None
    :type image_name: str, optional
    :param padding_factor: Padding factor applied to beam grid before computing the fast-fourier transform. The default has been set for operation on most systems. The user should be aware of memory constraints before increasing this parameter significatly., defaults to 50
    :type padding_factor: int, optional
    :param parallel: Run in parallel with Dask or in serial., defaults to False
    :type parallel: bool, optional
    :param grid_interpolation_mode: Method of interpolation used when gridding data. This is done using the `scipy.interpolate.griddata` method. For more information on the interpolation see `scipy.interploate <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata>`_, defaults to "linear"
    :type grid_interpolation_mode: str, optional. Available options: {"linear", "nearest", "cubic"}
    :param chan_average: Boolean dictating whether the channel average is computed and written to the output holog file., defaults to True
    :type chan_average: bool, optional
    :param chan_tolerance_factor: Tolerance used in channel averaging to determine the number of primary beam channels., defaults to 0.005
    :type chan_tolerance_factor: float, optional
    :param scan_average: Boolean dicating whether averagin is done over scan., defaults to True
    :type scan_average: bool, optional
    :param ant_id: List of antennae/antenna to be processed, defaults to "all" when None, ex. ea25
    :type ant_id: list or str, optional
    :param ddi: List of ddis/ddi to be processed, defaults to "all" when None, ex. 0
    :type ddi: list or int, optional
    :param to_stokes: Dictates whether polarization is computed according to stokes values., defaults to True
    :type to_stokes: bool, optional
    :param apply_mask: If True applies a mask to the aperture setting values outside of the aperture to zero., defaults to True
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

    Image object allows the user to access image data via compound dictionary keys with values, in order of depth, `ant` -> `ddi`. The image object also provides a `summary()` helper function to list available keys for each file. An outline of the image object structure is show below:

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

    """
    
    holog_params = locals()

    logger = _get_astrohack_logger()

    function_name = inspect.stack()[CURRENT_FUNCTION].function

    ######### Parameter Checking #########
    holog_params = _check_holog_params(function_name=function_name, holog_params=holog_params)
    
    _check_if_file_exists(holog_params['holog_name'])
    _check_if_file_will_be_overwritten(holog_params['image_name'], holog_params['overwrite'])

    json_data = "/".join((holog_params['holog_name'], ".holog_json"))

    with open(json_data, "r") as json_file:
        holog_json = json.load(json_file)
        
    meta_data = _read_meta_data(holog_params['holog_name'], 'holog', 'extract_holog')

    if holog_params["cell_size"] is None:
        cell_size = np.array([-meta_data["cell_size"], meta_data["cell_size"]])
        holog_params["cell_size"] = cell_size
    
    if holog_params["grid_size"] is None:
        n_pix = int(np.sqrt(meta_data["n_pix"]))
        grid_size = np.array([n_pix, n_pix])
        holog_params["grid_size"] = grid_size

    logger.info(f'[{function_name}]: Cell size: {str(cell_size)}, Grid size {str(grid_size)}')
    
    json_data = {
            "cell_size": holog_params["cell_size"].tolist(),
            "grid_size": holog_params["grid_size"].tolist()
    }
    
    with open(".holog_diagnostic.json", "w") as out_file:
        json.dump(json_data, out_file)

    if _dask_general_compute(function_name, holog_json, _holog_chunk, holog_params, ['ant', 'ddi'], parallel=parallel):
        
        output_attr_file = "{name}/{ext}".format(name=holog_params['image_name'], ext=".image_attr")
        _write_meta_data(output_attr_file, holog_params.copy())
        
        image_mds = AstrohackImageFile(holog_params['image_name'])
        image_mds._open()
        
        logger.info(f'[{function_name}]: Finished processing')

        return image_mds
    else:
        logger.warning(f"[{function_name}]: No data to process")
        return None


def _check_holog_params(function_name, holog_params):

    #### Parameter Checking ####
    parms_passed = True
    
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'holog_name', [str], default=None)

    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'grid_size', [list, np.ndarray],
                                                 list_acceptable_data_types=[np.int64, int], list_len=2, default='None',
                                                 log_default_setting=False)
    if (isinstance(holog_params['grid_size'], str)) and (holog_params['grid_size'] == 'None'):
        holog_params['grid_size'] = None
    else:
        holog_params['grid_size'] = np.array(holog_params['grid_size'])

    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'cell_size', [list, np.ndarray],
                                                 list_acceptable_data_types=[numbers.Number], list_len=2,
                                                 default='None', log_default_setting=False)
    if (isinstance(holog_params['cell_size'], str)) and (holog_params['cell_size'] == 'None'):
        holog_params['cell_size'] = None
    else:
        holog_params['cell_size'] = np.array(holog_params['cell_size'])

    base_name = _remove_suffix(holog_params['holog_name'], '.holog.zarr')
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'image_name', [str],
                                                 default=base_name+'.image.zarr')
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'padding_factor', [int], default=50)
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'parallel', [bool], default=False)
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'grid_interpolation_mode', [str],
                                                 acceptable_data=["nearest", "linear", "cubic"], default="nearest")
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'chan_average', [bool], default=True)
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'chan_tolerance_factor', [float],
                                                 acceptable_range=[0, 1], default=0.005)
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'scan_average', [bool], default=True)
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'ant', [list, str],
                                                 list_acceptable_data_types=[str], default='all')
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'ddi', [list, int],
                                                 list_acceptable_data_types=[int], default='all')
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'to_stokes', [bool], default=True)

    if isinstance(holog_params['phase_fit'], list) or isinstance(holog_params['phase_fit'], type(np.ndarray)):
        parms_passed = parms_passed and _check_parms(function_name, holog_params, 'phase_fit', [list, type(np.ndarray)],
                                                     list_acceptable_data_types=[bool], list_len=5)
    else:
        parms_passed = parms_passed and _check_parms(function_name, holog_params, 'phase_fit', [bool], default=True)

    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'apply_mask', [bool], default=True)
    parms_passed = parms_passed and _check_parms(function_name, holog_params, 'overwrite', [bool], default=False)

    _parm_check_passed(function_name, parms_passed)

    return holog_params

