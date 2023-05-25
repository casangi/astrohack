import dask
import xarray

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._tools import _parm_to_list


def _construct_general_graph_recursively(caller, looping_dict, function, param_dict, delayed_list, key_order,
                                         parallel=False, oneup=None):
    logger = _get_astrohack_logger()
    if len(key_order) == 0:
        if isinstance(looping_dict, xarray.Dataset):
            param_dict['xds_data'] = looping_dict
        if parallel:
            delayed_list.append(dask.delayed(function)(dask.delayed(param_dict)))
        else:
            delayed_list.append(0)
            function(param_dict)
    else:
        key = key_order[0]
        exec_list = _parm_to_list(caller, param_dict[key], looping_dict, key)
        for item in exec_list:
            param_dict[f'this_{key}'] = item
            try:
                _construct_general_graph_recursively(caller, looping_dict[item], function, param_dict, delayed_list,
                                                     key_order[1:], parallel=parallel, oneup=item)
            except KeyError:
                if oneup is None:
                    logger.warning(f'[{caller}]: {item} is not present in this mds')
                else:
                    logger.warning(f'[{caller}]: {item} is not present for {oneup}')


def _dask_general_compute(caller, looping_dict, function, param_dict, key_order, parallel=False):
    logger = _get_astrohack_logger()
    delayed_list = []
    _construct_general_graph_recursively(caller, looping_dict, function, param_dict, delayed_list=delayed_list,
                                         key_order=key_order, parallel=parallel)
    if len(delayed_list) == 0:
        logger.warning(f"[{caller}]: No data to process")
        return False
    else:
        if parallel:
            dask.compute(delayed_list)
        return True
