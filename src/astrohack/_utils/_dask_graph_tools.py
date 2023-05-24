import os
import dask
import xarray

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._tools import _parm_to_list


def _generate_antenna_ddi_graph_and_compute(function_name, chunk_function, parm_dict, parallel):
    """
    Simple routine to factorize the possibily parallel loop of processing
    Args:
        function_name: The caller function
        chunk_function: The chunk function to be passed to dask
        parm_dict: The parameter dictionary to be used by the chunk function
        parallel: Should the loop be executed in parallel?

    Returns: How many times the loop has been executed

    """
    logger = _get_astrohack_logger()
    delayed_list = []
    antennae = _parm_to_list(parm_dict['ant_name'], parm_dict['filename'], 'ant')
    if len(antennae) == 0:
        logger.warning(f'[{function_name}]: Antenna list is empty for antenna selection: {parm_dict["ant_name"]}')
    count = 0
    for antenna in antennae:
        parm_dict['this_antenna'] = antenna
        path = f'{parm_dict["filename"]}/{antenna}'
        if os.path.isdir(path):
            ddis = _parm_to_list(parm_dict['ddi'], f'{parm_dict["filename"]}/{antenna}', 'ddi')
            if len(ddis) == 0:
                logger.warning(
                    f'[{function_name}]: DDI list for antenna {antenna} is empty for ddi selection: {parm_dict["ddi"]}')
            for ddi in ddis:
                parm_dict['this_ddi'] = ddi
                path = f'{parm_dict["filename"]}/{antenna}/{ddi}'
                if os.path.isdir(path):
                    if parallel:
                        delayed_list.append(dask.delayed(chunk_function)(dask.delayed(parm_dict)))
                    else:
                        chunk_function(parm_dict)
                    count += 1
                else:
                    logger.warning(f'[{function_name}]: DDI {ddi} is not present for {antenna}')
        else:
            logger.warning(f'[{function_name}]: Antenna {antenna} is not present in {parm_dict["filename"]}')

    if parallel:
        dask.compute(delayed_list)

    if count == 0:
        logger.warning(f"[{function_name}]: No data to process")
    return count


def _construct_graph(data_dict, function, param_dict, delayed_list, key_list, parallel=False):

    if isinstance(data_dict, xarray.Dataset):
        param_dict['data'] = data_dict

        if parallel:
            delayed_list.append(dask.delayed(function)(dask.delayed(param_dict)))

        else:
            function(param_dict)

    else:
        for key, value in data_dict.items():
            if key_list:
                for element in key_list:
                    if key.find(element) == 0:
                        _construct_graph(value, function, param_dict, delayed_list, key_list, parallel)
            else:
                _construct_graph(value, function, param_dict, key_list, parallel)


def _dask_compute(data_dict, function, param_dict, key_list=[], parallel=False):

    delayed_list = []
    _construct_graph(data_dict, function, param_dict, delayed_list=delayed_list, key_list=key_list, parallel=parallel)

    if parallel:
        dask.compute(delayed_list)
