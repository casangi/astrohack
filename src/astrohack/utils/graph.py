import copy
import pathlib
import shutil

import dask
import toolviper.utils.logger as logger
import xarray as xr

from astrohack.utils.text import approve_prefix, param_to_list


def _white_list_creation(key_prefix, looping_dict, param_dict):
    exec_list = param_to_list(param_dict[key_prefix], looping_dict, key_prefix)
    white_list = [key for key in exec_list if approve_prefix(key)]
    return white_list


def _add_exec_data_to_param_dict(data_for_exec, param_dict):
    # param_dict is modified in place!
    if isinstance(data_for_exec, xr.DataTree):
        param_dict["xdt_data"] = data_for_exec
    elif isinstance(data_for_exec, xr.Dataset):
        param_dict["xds_data"] = data_for_exec
    elif isinstance(data_for_exec, dict):
        param_dict["dic_data"] = data_for_exec
    else:
        param_dict["unk_data"] = data_for_exec


def _factorized_graph_execution_return(status, ret_list, fetch_ret):
    if fetch_ret:
        if status:
            return status, ret_list
        else:
            return status, None
    else:
        return status


def _construct_general_graph_recursively(
    looping_dict,
    chunk_function,
    param_dict,
    delayed_list,
    key_order,
    output_mds,
    parallel,
    oneup=None,
):
    if len(key_order) == 0:
        _add_exec_data_to_param_dict(looping_dict, param_dict)
        if output_mds is None:
            args = [param_dict]
        else:
            args = [param_dict, output_mds]
        if parallel:
            delayed_list.append(dask.delayed(chunk_function)(*args))
        else:
            delayed_list.append((chunk_function, args))
    else:
        first_key_prefix = key_order[0]
        white_list = _white_list_creation(first_key_prefix, looping_dict, param_dict)

        for item in white_list:
            if item in looping_dict:
                this_param_dict = copy.deepcopy(param_dict)
                this_param_dict[f"this_{first_key_prefix}"] = item

                _construct_general_graph_recursively(
                    looping_dict=looping_dict[item],
                    chunk_function=chunk_function,
                    param_dict=this_param_dict,
                    delayed_list=delayed_list,
                    key_order=key_order[1:],
                    output_mds=output_mds,
                    parallel=parallel,
                    oneup=item,
                )

            else:
                if oneup is None:
                    logger.warning(f"{item} is not present in looping dict")
                else:
                    logger.warning(f"{item} is not present for {oneup}")


def create_and_execute_graph_from_dict(
    looping_dict,
    chunk_function,
    param_dict,
    key_order,
    output_mds=None,
    fetch_returns=False,
):
    parallel = param_dict["parallel"]
    if hasattr(looping_dict, "root"):
        looping_dict = looping_dict.root

    if output_mds is not None:
        output_mds.write(mode="a")

    # List created here to avoid complicated returns due to recursion.
    delayed_list = []
    _construct_general_graph_recursively(
        looping_dict=looping_dict,
        chunk_function=chunk_function,
        param_dict=param_dict,
        delayed_list=delayed_list,
        key_order=key_order,
        output_mds=output_mds,
        parallel=parallel,
    )

    if len(delayed_list) == 0:
        logger.warning("List of delayed processing jobs is empty: No data to process")
        return _factorized_graph_execution_return(False, [], fetch_returns)

    if parallel:
        return_list = dask.compute(delayed_list)[0]
    else:
        return_list = []
        for function, args in delayed_list:
            return_list.append(function(*args))

    if output_mds is not None:
        output_mds.consolidate(key_order)

        if len(output_mds.keys()) == 0:
            logger.warning("Processing did not yield any data")
            shutil.rmtree(output_mds.filename)
            return _factorized_graph_execution_return(False, return_list, fetch_returns)
        else:
            return _factorized_graph_execution_return(True, return_list, fetch_returns)

    return _factorized_graph_execution_return(True, return_list, fetch_returns)


def _sub_graph_execution_for_plots(
    looping_dict,
    chunk_function,
    param_dict,
    key_order,
):
    lvl1_key_prefix = key_order[0]
    lvl1_white_list = _white_list_creation(lvl1_key_prefix, looping_dict, param_dict)
    result_list = []

    if len(key_order) == 1:
        for lvl1_item in lvl1_white_list:
            if lvl1_item in looping_dict:
                this_param_dict = copy.deepcopy(param_dict)
                _add_exec_data_to_param_dict(looping_dict[lvl1_item], this_param_dict)
                this_param_dict[f"this_{lvl1_key_prefix}"] = lvl1_item
                result_list.append(chunk_function(this_param_dict))
            else:
                logger.warning(f"{lvl1_item} is not present in looping dict")
    elif len(key_order) == 2:
        lvl2_key_prefix = key_order[1]
        for lvl1_item in lvl1_white_list:
            if lvl1_item in looping_dict:
                lvl2_white_list = _white_list_creation(
                    lvl2_key_prefix, looping_dict[lvl1_item], param_dict
                )
                for lvl2_item in lvl2_white_list:
                    if lvl2_item in looping_dict[lvl1_item]:
                        this_param_dict = copy.deepcopy(param_dict)
                        _add_exec_data_to_param_dict(
                            looping_dict[lvl1_item][lvl2_item], this_param_dict
                        )
                        this_param_dict[f"this_{lvl1_key_prefix}"] = lvl1_item
                        this_param_dict[f"this_{lvl2_key_prefix}"] = lvl2_item
                        result_list.append(chunk_function(this_param_dict))
                    else:
                        logger.warning(
                            f"{lvl2_item} is not present for {lvl1_item} in looping_dict"
                        )
            else:
                logger.warning(f"{lvl1_item} is not present in looping dict")

    return result_list


def create_and_execute_graphs_for_outputs(
    mds_object,
    chunk_function,
    param_dict,
    key_order,
    fetch_returns=False,
):
    """
    Dask parallelization exclusively for exports, parallelization is done at the antenna level to decrease graph size \
    and optimize plot creation.
    Args:
        mds_object: Astrohack MDS object from which to plot
        chunk_function: Plotting chunk function
        param_dict: The chunk function parameters
        key_order: Order in which to execute keys
        fetch_returns: Return value from chunk function

    Returns:
        None
    """
    parallel = param_dict["parallel"]
    try:
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)
    except KeyError:
        # Observation summary creation case, where a destination folder is not necessary
        pass

    n_lvls = len(key_order)
    if n_lvls == 1:
        # This can be the case for position_mds where the key depth is only known at runtime.
        # In this case of 1 level then the usual executioner is to be used.
        return create_and_execute_graph_from_dict(
            looping_dict=mds_object,
            chunk_function=chunk_function,
            param_dict=param_dict,
            key_order=key_order,
        )

    # here only the first level of the tree is parallelized
    looping_dict = mds_object.root

    if "display" in param_dict.keys():
        if param_dict["display"] and param_dict["parallel"]:
            logger.warning(
                "Display cannot be True in parallel mode, setting it to False"
            )
            param_dict["display"] = False

    first_key_prefix = key_order[0]
    white_list = _white_list_creation(first_key_prefix, looping_dict, param_dict)

    delayed_list = []
    for item in white_list:
        if item in looping_dict:
            this_param_dict = copy.deepcopy(param_dict)
            this_param_dict[f"this_{first_key_prefix}"] = item
            if parallel:
                delayed_list.append(
                    dask.delayed(_sub_graph_execution_for_plots)(
                        looping_dict[item],
                        chunk_function,
                        this_param_dict,
                        key_order[1:],
                    )
                )
            else:
                delayed_list.append(
                    (
                        _sub_graph_execution_for_plots,
                        (
                            looping_dict[item],
                            chunk_function,
                            this_param_dict,
                            key_order[1:],
                        ),
                    )
                )
        else:
            logger.warning(f"{item} is not present in looping dict")

    if parallel:
        result_list = dask.compute(delayed_list)[0]
    else:
        result_list = []
        for this_chuk_function, args in delayed_list:
            result_list.append(this_chuk_function(*args))

    return_list = [item for sublist in result_list for item in sublist]
    return _factorized_graph_execution_return(True, return_list, fetch_returns)


def compute_graph_from_lists(
    param_dict,
    chunk_function,
    looping_key_list,
):
    """
    Creates and executes a graph based on entries in a parameter dictionary that are lists
    Args:
        param_dict: The parameter dictionary
        chunk_function: The function for the operation chunk
        looping_key_list: The keys that are lists in the parameter dictionaries over which to loop over

    Returns:
        A list containing the returns of the calls to the chunk function.
    """
    parallel = param_dict["parallel"]
    niter = len(param_dict[looping_key_list[0]])

    delayed_list = []
    result_list = []
    for i_iter in range(niter):
        this_param = copy.deepcopy(param_dict)
        for key in looping_key_list:
            this_param[f"this_{key}"] = param_dict[key][i_iter]

        if parallel:
            delayed_list.append(dask.delayed(chunk_function)(dask.delayed(this_param)))
        else:
            delayed_list.append(0)
            result_list.append(chunk_function(this_param))

    if parallel:
        result_list = dask.compute(delayed_list)

    return result_list
