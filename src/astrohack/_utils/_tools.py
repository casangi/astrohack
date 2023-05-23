import os
import json
import dask
import xarray

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

def _well_positioned_colorbar(ax, fig, image, label, location='right', size='5%', pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    return fig.colorbar(image, label=label, cax=cax)


def _remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def _jsonify(holog_obj):
    """ Convert holog_obs_description dictionay to json format. This just means converting numpy.ndarry
        entries to string lists.

    :param holog_obj: holog_obs_description dictionary.
    :type holog_obj: dict
    :param holog_obj: holog_obs_description dictionary.
    :type holog_obj: dict
    """
    for ddi_key, ddi_value in holog_obj.items():
        for map_key, map_value in holog_obj[ddi_key].items():
            for attr_key, attr_value in holog_obj[ddi_key][map_key].items():
                if "scans" in attr_key:
                    holog_obj[ddi_key][map_key][attr_key] = list(map(str, attr_value))
                
                elif "ant" in attr_key:
                    for ant_key, ant_value in holog_obj[ddi_key][map_key][attr_key].items():
                        holog_obj[ddi_key][map_key][attr_key][ant_key] = list(map(str, ant_value))

                else:
                    pass

def _add_prefix(input_string, prefix):
    wrds = input_string.split('/')
    wrds[-1] = prefix+'_'+wrds[-1]
    return '/'.join(wrds)

def _print_holog_obs_dict(holog_obj):
    logger = _get_astrohack_logger()
    
    OPEN_DICT  = ":{"
    CLOSE_DICT = "}"
    
    OPEN_LIST  = ":["
    CLOSE_LIST = "]"

    logger.info("| ********************************************************** |")
    logger.info("|                 HOLOG OBSERVATION DICTIONARY               |")
    logger.info("| ********************************************************** |\n\n")
    
    for ddi_key, ddi_value in holog_obj.items():
        logger.info("{ddi_key} {open_bracket}".format(ddi_key=ddi_key, open_bracket=OPEN_DICT))
        for map_key, map_value in holog_obj[ddi_key].items():
            logger.info("{map_key: >10} {open_bracket}".format(map_key=map_key, open_bracket=OPEN_DICT))
            for attr_key, attr_value in holog_obj[ddi_key][map_key].items():
                if "scans" in attr_key:
                    logger.info("{attr_key: >12} {open_list}".format(attr_key=attr_key, open_list=OPEN_LIST))
    
                    scan_list = ", ".join(list(map(str, holog_obj[ddi_key][map_key][attr_key])))
                    logger.info("{scan: >18}".format(scan=scan_list))                                   # The print just ification in notebook is weird on this and seems to move according to list length ...
                    logger.info("{close_bracket: >10}".format(close_bracket=CLOSE_LIST))
                
                elif "ant" in attr_key:
                    logger.info("{attr_key: >12} {open_bracket}".format(attr_key=attr_key, open_bracket=OPEN_DICT))
                    for ant_key, ant_value in holog_obj[ddi_key][map_key][attr_key].items():
                        logger.info("{ant_key: >18} {open_list}".format(ant_key=ant_key, open_list=OPEN_LIST))
                        logger.info("{antenna: >25}".format( antenna=", ".join(ant_value) ))
                        logger.info("{close_list: >15}".format(close_list=CLOSE_LIST))
                    
                    logger.info("{close_bracket: >10}".format(close_bracket=CLOSE_DICT))

                else:
                    pass
        logger.info("{close_bracket: >5}".format(close_bracket=CLOSE_DICT))
        
    logger.info("{close_bracket}".format(close_bracket=CLOSE_DICT))


def _parm_to_list(parm, path):
    if parm == 'all':
        oulist = os.listdir(path)
    elif isinstance(parm, str):
        oulist = [parm]
    else:
        oulist = parm
    return oulist

def _numpy_to_json(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)

def _split_pointing_table(ms_name, antennas):
    """ Split pointing table to contain only specified antennas

    :param ms_name: Measurement file
    :type ms_name: str
    :param antennas: List of antennas to sub-select on.
    :type antennas: list (str)
    """

    # Need to get thea antenna-id values for teh input antenna names. This is not available in the POINTING table
    # so we build the values from the ANTENNA table.
    
    table = "/".join((ms_name, 'ANTENNA'))
    query = 'select NAME from {table}'.format(table=table)
    
    ant_names = np.array(tables.taql(query).getcol('NAME'))
    ant_id = np.arange(len(ant_names))
    
    query_ant = np.searchsorted(ant_names, antennas)
    
    ant_list = " or ".join(["ANTENNA_ID=={ant}".format(ant=ant) for ant in query_ant])
    
    # Build new POINTING table from the sub-selection of antenna values.
    table = "/".join((ms_name, "POINTING"))    
    
    selection = "select * from {table} where {antennas}".format(table=table, antennas=ant_list)
    
    reduced = tables.taql(selection)
    
    # Copy the new table to the source measurement set.
    table = "/".join((ms_name, 'REDUCED'))
    
    reduced.copy(newtablename='{table}'.format(table=table), deep=True)
    reduced.done()
    
    # Remove old POINTING table.
    shutil.rmtree("/".join((ms_name, 'POINTING')))
    
    # Rename REDUCED table to POINTING
    tables.tablerename(
        tablename="/".join((ms_name, 'REDUCED')), 
        newtablename="/".join((ms_name, 'POINTING'))
    )

def _dask_compute(data_dict, function, param_dict, key_list=["ant_", "ddi_", "map_"], parallel=False):
    
    delayed_list = []
    _construct_graph(data_dict, function, param_dict, delayed_list=delayed_list, key_list=key_list, parallel=parallel)
    
    if parallel:
        dask.compute(delayed_list)

def _construct_graph(data_dict, function, param_dict, delayed_list, key_list=["ant_", "ddi_", "map_"], parallel=False):

    
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