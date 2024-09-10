import dask
import xarray
import toolviper.utils.logger as logger

from astrohack.utils.text import approve_prefix
from astrohack.utils.text import param_to_list


def _construct_general_graph_recursively(
        looping_dict,
        chunk_function,
        param_dict,
        delayed_list,
        key_order,
        parallel=False,
        oneup=None
):
    if len(key_order) == 0:
        if isinstance(looping_dict, xarray.Dataset):
            param_dict['xds_data'] = looping_dict

        elif isinstance(looping_dict, dict):
            param_dict['data_dict'] = looping_dict

        if parallel:
            delayed_list.append(dask.delayed(chunk_function)(dask.delayed(param_dict)))

        else:
            delayed_list.append(0)
            chunk_function(param_dict)
    else:
        key = key_order[0]

        exec_list = param_to_list(param_dict[key], looping_dict, key)

        white_list = [key for key in exec_list if approve_prefix(key)]

        for item in white_list:
            param_dict[f'this_{key}'] = item

            if item in looping_dict:
                _construct_general_graph_recursively(
                    looping_dict=looping_dict[item],
                    chunk_function=chunk_function,
                    param_dict=param_dict,
                    delayed_list=delayed_list,
                    key_order=key_order[1:],
                    parallel=parallel,
                    oneup=item
                )

            else:
                if oneup is None:
                    logger.warning(f'{item} is not present in looping dict')
                else:
                    logger.warning(f'{item} is not present for {oneup}')


def compute_graph(looping_dict, chunk_function, param_dict, key_order, parallel=False):
    """
    General tool for looping over the data and constructing graphs for dask parallel processing
    Args:
        looping_dict: The dictionary containing the keys over which the loops are to be executed
        chunk_function: The chunk function to be executed
        param_dict: The parameter dictionary for the chunk function
        key_order: The order over which to loop over the keys inside the looping dictionary
        parallel: Are loops to be executed in parallel?

    Returns: True if processing has occurred, False if no data was processed

    """

    delayed_list = []
    _construct_general_graph_recursively(
        looping_dict=looping_dict,
        chunk_function=chunk_function,
        param_dict=param_dict,
        delayed_list=delayed_list,
        key_order=key_order,
        parallel=parallel
    )

    if len(delayed_list) == 0:
        logger.warning(f"List of delayed processing jobs is empty: No data to process")

        return False

    else:
        if parallel:
            dask.compute(delayed_list)
        return True
