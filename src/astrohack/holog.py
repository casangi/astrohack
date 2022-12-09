import dask
import time
import os
import dask
import dask.distributed

import dask.array as da
import numpy as np
import xarray as xr

from astrohack.dio import load_hack_file
from astrohack._utils._io import _read_dimensions_meta_data

def _holog_chunk(holog_chunk_params):
        """_summary_

        Args:
            holog_chunk_params (dict): _description_
        """

        hack, ant_data_dict = load_hack_file(holog_chunk_params['hack_name'], dask_load=False, load_pnt_dict=False, ant_id=holog_chunk_params['ant_id'])

        init = False

        for ddi in ant_data_dict.keys():
                for scan in ant_data_dict[ddi].keys():
                        if init is False:
                                n_pol = ant_data_dict[ddi][scan].dims['pol']
                                lm = ant_data_dict[ddi][scan].DIRECTIONAL_COSINES.values[:, np.newaxis, np.newaxis, :]
                                lm = np.tile(lm, (1, 1, n_pol, 1))
            
                                vis = ant_data_dict[ddi][scan].VIS.mean(dim='chan').values[:, np.newaxis, :]
            
                                init = True
                        else:
                                temp = ant_data_dict[ddi][scan].DIRECTIONAL_COSINES.values[:, np.newaxis, np.newaxis, :]
                                temp = np.tile(vis, (1, 1, n_pol, 1))
                                lm = np.concatenate( (lm, temp), axis=1)
            
                                temp = ant_data_dict[ddi][scan].VIS.mean(dim='chan').values[:, np.newaxis, :]
                                vis = np.concatenate( (vis, temp), axis=1)

        coords = _read_dimensions_meta_data(hack_name=holog_chunk_params['hack_name'], ant_id=holog_chunk_params['ant_id'])
        xds = xr.Dataset()
        xds.assign_coords(coords)
        xds['VIS']= xr.DataArray(vis, dims=['time','ddi','pol'])
        xds['DIRECTIONAL_COSINES'] = xr.DataArray(lm, dims=['time','ddi', 'pol', 'lm'])
        xds.attrs['ant_id'] = holog_chunk_params['ant_id']
        #xds.attrs['parallactic_samples'] = parallactic_samples
        xds.to_zarr(os.path.join(".".join((holog_chunk_params['hack_name'], "holog")) , str(holog_chunk_params['ant_id'])), mode='w', compute=True, consolidated=True)


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
