import dask
import time
import json
import os
import dask
import dask.distributed

import dask.array as da
import numpy as np
import xarray as xr

from scipy.interpolate import griddata

from astrohack.dio import load_hack_file
from astrohack._utils._io import _read_dimensions_meta_data

def _holog_chunk(holog_chunk_params):
        """_summary_

        Args:
            holog_chunk_params (dict): _description_
        """
        hack, ant_data_dict = load_hack_file('hack.dict', dask_load=False, load_pnt_dict=False, ant_id=27)

        dims = _read_dimensions_meta_data(hack_name='hack.dict', ant_id=holog_chunk_params['ant_id'])

        n_ddi = len(ant_data_dict.keys())
        n_scan = len(ant_data_dict[0].keys())
        n_pol = dims['pol']
        n_points = int(np.sqrt(dims['time'])+1)

        ant_data_array = np.empty((n_scan, n_ddi, n_pol, n_points, n_points), dtype=np.cdouble)

        time_centroid = []

        for ddi_index, ddi in enumerate(ant_data_dict.keys()):
                for scan_index, scan in enumerate(ant_data_dict[ddi].keys()):

                        # Grid (l, m) points
                        lm = ant_data_dict[ddi][scan].DIRECTIONAL_COSINES.values[:, np.newaxis, :]
                        lm = np.tile(lm, (1, n_pol, 1))
        
                        # VIS values    
                        vis = ant_data_dict[ddi][scan].VIS.mean(dim='chan').values
                        time_centroid_index = int(ant_data_dict[ddi][scan].dims['time']*0.5)+1
        
                        time_centroid.append(ant_data_dict[ddi][scan].coords['time'][time_centroid_index].values)
                        extent = dims['extent']
        
                        grid_x, grid_y = np.mgrid[-extent:extent:n_points*1j, -extent:extent:n_points*1j]
        
                        grid = griddata(lm[:, ddi, :], vis[:, ddi], (grid_x, grid_y), method='nearest')
                        for pol in range(n_pol):
                                ant_data_array[scan_index, ddi_index, pol, :, :] = grid
        

        xds = xr.Dataset()
        xds.assign_coords({
                'time_centroid': np.array(time_centroid), 
                'ddi':list(map(int, ant_data_dict.keys())), 
                'pol':[i for i in range(n_pol)]
        })

        xds['GRID'] = xr.DataArray(ant_data_array, dims=['time-centroid','ddi', 'pol', 'l', 'm'])

        xds.attrs['ant_id'] = holog_chunk_params['ant_id']
        xds.attrs['time_centroid'] = np.array(time_centroid)

        xds.to_zarr(os.path.join(".".join((holog_chunk_params['hack_name'], "holog")) , str(holog_chunk_params['ant_id'])), mode='w', compute=True, consolidated=True)
        


def holog(hack_name, parallel=True):
        """_summary_

        Args:
            hack_name (str): Hack file name
            parallel (bool, optional): Run in parallel with Dask or in serial. Defaults to True.
        """
       
        try:
                if os.path.exists(hack_name):
                        hack_meta_data = "/".join((hack_name, ".hack_json"))


                        with open(hack_meta_data, "r") as json_file: 
                                hack_json = json.load(json_file)
    
                        ant_list = hack_json.keys()


                        holog_chunk_params = {}
                        holog_chunk_params['hack_name'] = hack_name

                        delayed_list = []

                        for ant_id in ant_list:
                                holog_chunk_params['ant_id'] = ant_id
                                
                                if parallel:
                                        delayed_list.append(
                                                dask.delayed(_holog_chunk)(
                                                        dask.delayed(holog_chunk_params)
                                                )
                                        )

                                else:
                                        _holog_chunk(holog_chunk_params)

                        if parallel: 
                                dask.compute(delayed_list)

                else:
                        raise FileNotFoundError("File: {} - not found error.".format(hack_name))
        except Exception as e:
                print('Exception thrown for antenna: ', e)
