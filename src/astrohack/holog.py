import dask
import time
import os
import dask

import dask.array as da
import numpy as np
import xarray as xr

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
        

def holog(hack_name, parallel=True):
        """_summary_

        Args:
            hack_name (str): Hack file name
            parallel (bool, optional): Run in parallel with Dask or in serial. Defaults to True.
        """
       
        try:
                if os.path.exists(hack_name):
                        ant_list = os.listdir('/'.join((hack_name, 'pnt.dict/')))
                        ant_list = list(map(int, ant_list))

                        delayed_list = []

                        for ant in ant_list:
                                if parallel:
                                        delayed_list.append(
                                                dask.delayed(_holog_chunk(ant, hack_name))
                                        )
                                else:
                                        _holog_chunk(ant, hack_name)

                        if parallel: 
                                dask.compute(delayed_list)

                else:
                        raise FileNotFoundError("File: {} - not found error.".format(hack_name))
        except Exception as e:
                print(e)
