from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._constants import *

def _convert_ant_name_to_id(ant_list, ant_names):
  """_summary_

  Args:
      ant_list (_type_): _description_
      ant_names (_type_): _description_

  Returns:
      _type_: _description_
  """
  
  return np.nonzero(np.in1d(ant_list, ant_names))[0]

# Global conversion functions
def _convert_to_db(val: float):
    """
    Converts a float value to decibels
    Args:
        val (float): Value to be converted to decibels
    Returns:
        Value in decibels
    """
    return 10.0 * np.log10(val)


def _convert_unit(unitin, unitout, kind):
    """
    Convert between unit of the same kind
    Args:
        unitin: Origin unit
        unitout: Destiny unit
        kind: 'trigonometric' or 'length'

    Returns:
        Convertion factor to go from unitin to unitout
    """
    try:
        unitlist = unit_dict[kind]
        factorlist = fact_dict[kind]
        
    except KeyError:
        logger = _get_astrohack_logger()
        logger.error("Unrecognized unit kind: " + kind)
        raise KeyError('Unrecogized unit kind')

    inidx = _test_unit(unitin, unitlist)
    ouidx = _test_unit(unitout, unitlist)
    factor = factorlist[inidx]/factorlist[ouidx]

    return factor


def _test_unit(unit, unitlist):
    """
    Test if a unit is known
    Args:
        unit: unit name
        unitlist: List containing unit names

    Returns:
        Unit index in unitlist
    """
    try:
        idx = unitlist.index(unit)
    except ValueError:
        logger = _get_astrohack_logger()
        logger.error("Unrecognized unit: " + unit)
        raise ValueError('Unit not in list')

    return idx


def _to_stokes(grid,pol):
    grid_stokes = np.zeros_like(grid)
    
    if 'RR' in pol:
        grid_stokes[:,:,0,:,:] = (grid[:,:,0,:,:] + grid[:,:,3,:,:])/2
        grid_stokes[:,:,1,:,:] = (grid[:,:,1,:,:] + grid[:,:,2,:,:])/2
        grid_stokes[:,:,2,:,:] = 1j*(grid[:,:,1,:,:] - grid[:,:,2,:,:])/2
        grid_stokes[:,:,3,:,:] = (grid[:,:,0,:,:] - grid[:,:,3,:,:])/2
    elif 'XX' in pol:
        grid_stokes[:,:,0,:,:] = (grid[:,:,0,:,:] + grid[:,:,3,:,:])/2
        grid_stokes[:,:,1,:,:] = (grid[:,:,0,:,:] - grid[:,:,3,:,:])/2
        grid_stokes[:,:,2,:,:] = (grid[:,:,1,:,:] + grid[:,:,2,:,:])/2
        grid_stokes[:,:,3,:,:] = 1j*(grid[:,:,1,:,:] - grid[:,:,2,:,:])/2
    else:
        raise Exception("Pol not supported " + str(pol))
    
    return grid_stokes

def convert_dict_from_numba(func):
    def wrapper(*args, **kwargs):
        numba_dict = func(*args, **kwargs)

        converted_dict = dict(numba_dict)
    
        for key, _ in numba_dict.items():
            converted_dict[key] = dict(converted_dict[key])

        return converted_dict
    return wrapper