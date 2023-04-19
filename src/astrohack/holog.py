import json
import os

import dask
import dask.distributed
import numpy as np
import numbers

from astrohack._utils._holog import _holog_chunk

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._parm_utils._check_parms import _check_parms
from astrohack._utils._utils import _remove_suffix
   
from astrohack._utils._io import check_if_file_will_be_overwritten, check_if_file_exists
from astrohack._utils._dio import AstrohackImageFile

#fp=open('holog.log','w+')
#@profile(stream=fp)
def holog(
    holog_name,
    grid_size,
    cell_size,
    image_name=None,
    padding_factor=50,
    grid_interpolation_mode="nearest",
    chan_average=True,
    chan_tolerance_factor=0.005,
    reference_scaling_frequency=None,
    scan_average=True,
    ant_list=None,
    to_stokes=True,
    apply_mask=True,
    phase_fit=True,
    overwrite=False,
    parallel=True):
    """ Process holography data and derive aperture illumination pattern.

    :param holog_name: Name of holography .holog.zarr file to process.
    :type holog_name: str

    :param grid_size: Numpy array specifying the dimensions of the grid used in data gridding.
    :type grid_size: numpy.ndarray, dtype int

    :param cell_size: Numpy array defining the cell size of each grid bin.
    :type cell_size: numpy.ndarray, dtype float

    :param image_name: Defines the name of the output image name. If value is None, the name will be set to <base_name>.image.zarr, defaults to None
    :type image_name: str, optional

    :param padding_factor: Padding factor applied to beam grid before computing the fast-fourier transform. The default has been set for operation on most systems. The user should be aware of memory constraints before increasing this parameter significatly., defaults to 50
    :type padding_factor: int, optional

    :param parallel: Run in parallel with Dask or in serial., defaults to True
    :type parallel: bool, optional

    :param grid_interpolation_mode: Method of interpolation used when gridding data. This is done using the `scipy.interpolate.griddata` method. For more information on the interpolation see `scipy.interploate <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata>`_, defaults to "nearest"
    :type grid_interpolation_mode: str, optional

    :param chan_average: Boolean dictating whether the channel average is computed and written to the output holog file., defaults to True
    :type chan_average: bool, optional

    :param chan_tolerance_factor: Tolerance used in channel averaging to determine the number of primary beam channels., defaults to 0.005
    :type chan_tolerance_factor: float, optional

    :param reference_scaling_frequency: When computing the channel average the lm frequency values are scaled by frequency. If the default None is used, the scaling is simply unity, however if `reference_scaling_frequency` is set then the scaling is done according to (average_frequency/reference_scaling_frequency)., defaults to None
    :type reference_scaling_frequency: _type_, optional

    :param scan_average: Boolean dicating whether averagin is done over scan., defaults to True
    :type scan_average: bool, optional

    :param ant_list: Optional list of sub-antennas to use when the user doesn't to do holography for all antennas, defaults to None
    :type ant_list: list, optional

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

    """
    
    logger = _get_astrohack_logger()
    
    holog_params = _check_holog_parms(
        holog_name, 
        grid_size,
        cell_size, 
        image_name, 
        padding_factor, 
        parallel, 
        grid_interpolation_mode, 
        chan_average, 
        chan_tolerance_factor, 
        reference_scaling_frequency, 
        scan_average,
        ant_list, 
        to_stokes, 
        apply_mask, 
        phase_fit,
        overwrite
    )
    
    check_if_file_exists(holog_params['holog_file'])
    check_if_file_will_be_overwritten(holog_params['image_file'],holog_params['overwrite'])

    json_data = "/".join((holog_params['holog_file'], ".holog_json"))
    meta_data = "/".join((holog_params['holog_file'], ".holog_attr"))
    
    with open(json_data, "r") as json_file:
        holog_json = json.load(json_file)
    
    with open(meta_data, "r") as meta_file:
        meta_data = json.load(meta_file)
        image_name=None,

    if  holog_params['ant_list'] == 'all':
        holog_params['ant_list'] = list(holog_json.keys())
        
    logger.info('Mapping antennas ' + str(holog_params['ant_list']))

    ''' VLA data sampling can be uneven so averging step in extracty_holog does not work consequntly int(np.sqrt(meta_data["n_time"])) is not correct.
    if (cell_size is None) and (grid_size is None):
        ###To Do: Calculate one gridsize and cell_size for all ddi's, antennas, ect. Fix meta data ant_holog_dict gets overwritten for more than one ddi.
        
        n_points = int(np.sqrt(meta_data["n_time"]))
        grid_size = np.array([n_points, n_points])

        l_min_extent = meta_data["extent"]["l"]["min"]
        l_max_extent = meta_data["extent"]["l"]["max"]

        m_min_extent = meta_data["extent"]["m"]["min"]
        m_max_extent = meta_data["extent"]["m"]["max"]

        step_l = (l_max_extent - l_min_extent) / grid_size[0]
        step_m = (m_max_extent - m_min_extent) / grid_size[1]
        step_l = (step_l+step_m)/2
        step_m = step_l

        cell_size = np.array([step_l, step_m])

        holog_chunk_params["cell_size"] = cell_size
        holog_chunk_params["grid_size"] = grid_size

        logger.info("Cell size: " + str(cell_size) + " Grid size " + str(grid_size))
    else:
        holog_chunk_params["cell_size"] = cell_size
        holog_chunk_params["grid_size"] = grid_size
    '''

    
    holog_chunk_params =  holog_params
    delayed_list = []
    
    
    for ant_id in holog_chunk_params['ant_list']:
        for ddi in list(holog_json[ant_id].keys()):
            logger.info("Processing ant_id: " + str(ant_id)  + " and " + ddi)
            holog_chunk_params["ant_id"] = ant_id
            holog_chunk_params["ddi_id"] = ddi
            
            if parallel:
                delayed_list.append(
                    dask.delayed(_holog_chunk)(dask.delayed(holog_chunk_params))
                )

            else:
                _holog_chunk(holog_chunk_params)
            

    if holog_chunk_params['parallel']:
        dask.compute(delayed_list)
        
    image_mds = AstrohackImageFile(holog_chunk_params['image_file'])
    image_mds.open()
    
    return image_mds



def _check_holog_parms(holog_name,grid_size,cell_size,image_name,
                      padding_factor,parallel,grid_interpolation_mode,
                      chan_average,chan_tolerance_factor,
                      reference_scaling_frequency,scan_average,
                      ant_list,to_stokes,apply_mask,phase_fit,overwrite):

    holog_params = {}
    holog_params["holog_file"] = holog_name
    holog_params["grid_size"] = grid_size
    holog_params["cell_size"] = cell_size
    holog_params["image_file"] = image_name
    holog_params["padding_factor"] = padding_factor
    holog_params["parallel"] = parallel
    holog_params["grid_interpolation_mode"] = grid_interpolation_mode
    holog_params["chan_average"] = chan_average
    holog_params["chan_tolerance_factor"] = chan_tolerance_factor
    holog_params["reference_scaling_frequency"] = reference_scaling_frequency
    holog_params["scan_average"] = scan_average
    holog_params["ant_list"] = ant_list
    holog_params["to_stokes"] = to_stokes
    holog_params["apply_mask"] = apply_mask
    holog_params["phase_fit"] = phase_fit
    holog_params["overwrite"] = overwrite
    
    #### Parameter Checking ####
    logger = _get_astrohack_logger()
    parms_passed = True
    
    parms_passed = parms_passed and _check_parms(holog_params, 'holog_file', [str],default=None)

    parms_passed = parms_passed and _check_parms(holog_params, 'grid_size', [list,np.ndarray], list_acceptable_data_types=[np.int,np.int64], list_len=2, default=None)
    holog_params['grid_size'] = np.array(holog_params['grid_size'])

    parms_passed = parms_passed and _check_parms(holog_params, 'cell_size', [list,np.ndarray], list_acceptable_data_types=[numbers.Number], list_len=2, default=None)
    holog_params['cell_size'] = np.array(holog_params['cell_size'])

    
    base_name = _remove_suffix(holog_params['holog_file'],'.holog.zarr')
    parms_passed = parms_passed and _check_parms(holog_params,'image_file', [str],default=base_name+'.image.zarr')
    
    parms_passed = parms_passed and _check_parms(holog_params, 'padding_factor', [int], default=50)
  
    
    parms_passed = parms_passed and _check_parms(holog_params, 'parallel', [bool],default=False)

    
    parms_passed = parms_passed and _check_parms(holog_params,'grid_interpolation_mode', [str],acceptable_data=["nearest","linear","cubic"],default="nearest")
   
    
    parms_passed = parms_passed and _check_parms(holog_params, 'chan_average', [bool],default=True)

    
    parms_passed = parms_passed and _check_parms(holog_params, 'chan_tolerance_factor', [float], acceptable_range=[0,1], default=0.005)
   
    
    parms_passed = parms_passed and _check_parms(holog_params, 'reference_scaling_frequency', [float,np.float],default='None')
    if holog_params['reference_scaling_frequency'] == 'None':
        holog_params['reference_scaling_frequency'] =  None
    
    parms_passed = parms_passed and _check_parms(holog_params, 'scan_average', [bool],default=True)

    parms_passed = parms_passed and _check_parms(holog_params, 'ant_list', [list,np.ndarray], list_acceptable_data_types=[str], default='all')
 
    parms_passed = parms_passed and _check_parms(holog_params, 'to_stokes', [bool],default=True)
   
 
    if isinstance(holog_params['phase_fit'],list) or isinstance(holog_params['phase_fit'],type(np.ndarray)):
        parms_passed = parms_passed and _check_parms(holog_params, 'phase_fit', [list,type(np.ndarray)], list_acceptable_data_types=[bool], list_len=5)
    else:
        parms_passed = parms_passed and _check_parms(holog_params, 'phase_fit', [bool], default=True)
  
   
    parms_passed = parms_passed and _check_parms(holog_params, 'apply_mask', [bool],default=True)
    
    parms_passed = parms_passed and _check_parms(holog_params, 'overwrite', [bool],default=False)

    if not parms_passed:
        logger.error("extract_holog parameter checking failed.")
        raise Exception("extract_holog parameter checking failed.")
    #### End Parameter Checking ####
    
    
    return holog_params

