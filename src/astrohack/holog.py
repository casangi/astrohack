import dask
import time
import os
import dask
import dask.distributed

import dask.array as da
import numpy as np
import xarray as xr

from astrohack.dio import load_hack_file

def _holog_chunk(holog_chunk_params):
        """_summary_

        Args:
            holog_chunk_params (dict): _description_
        """

        hack, ant_meta_data = load_hack_file(holog_chunk_params['hack_name'], dask_load=False, load_pnt_dict=False, ant_id=holog_chunk_params['ant_id'])
        
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

                        holog_chunk_params = {}
                        holog_chunk_params['hack_name'] = hack_name

                        delayed_list = []

                        for ant in ant_list:
                                holog_chunk_params['ant_id'] = ant
                                
                                if parallel:
                                        delayed_list.append(
                                                dask.delayed(_holog_chunk)(holog_chunk_params)
                                        )

                                else:
                                        _holog_chunk(holog_chunk_params)

                        if parallel: 
                                dask.compute(delayed_list)

                else:
                        raise FileNotFoundError("File: {} - not found error.".format(hack_name))
        except Exception as e:
                print(e)
