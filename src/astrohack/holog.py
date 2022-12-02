import dask
import time
import os
import dask

import dask.array as da
import numpy as np
import xarray as xr

from astrohack.dio import extract_holog

@dask.delayed
def _holog_chunk(ant, hack_file):
        """_summary_

        Args:
            ant (_type_): _description_
            hack_file (_type_): _description_
        """
        
        dask.distributed.get_worker().log_event("runtimes", {
                "ant": ant, 
                "hackfile": hack_file, 
                "message": "h.a.c.k. ...."}
        )
        

def holog(ms_name, hack_name, holog_obs_description, data_col, subscan_intent, parallel):
        """_summary_

        Args:
            ms_name (_type_): _description_
            hack_name (_type_): _description_
            holog_obs_description (_type_): _description_
            data_col (_type_): _description_
            subscan_intent (_type_): _description_
            parallel (_type_): _description_
        """
       
        extract_holog(
                ms_name=ms_name,
                hack_name=hack_name,
                holog_obs_dict=holog_obs_description,
                data_col=data_col,
                subscan_intent=subscan_intent,
                parallel=parallel
        )

        ant_list = os.listdir('/'.join((hack_name, 'pnt.dict/')))
        ant_list = list(map(int, ant_list))

        delayed_list = []

        for ant in ant_list:
                delayed_list.append(_holog_chunk(ant, hack_name))

        dask.compute(delayed_list)



