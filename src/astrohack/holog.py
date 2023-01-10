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

from astrohack._utils import _system_message as console
from astrohack.dio import load_hack_file
from astrohack._utils._io import _read_dimensions_meta_data

def _holog_chunk(holog_chunk_params):
        """_summary_

        Args:
            holog_chunk_params (dict): Dictionary containing holography parameters.
        """
        _, ant_data_dict = load_hack_file(holog_chunk_params['hack_file'], dask_load=False, load_pnt_dict=False, ant_id=27)
        average = holog_chunk_params['average']
        lmscale = holog_chunk_params['lmscale']
        simple_avg = average and not lmscale
        for ddi_index, ddi in enumerate(ant_data_dict.keys()):
                meta_data = _read_dimensions_meta_data(hack_file=holog_chunk_params['hack_file'], ddi=ddi_index, ant_id=holog_chunk_params['ant_id'])

                n_scan = len(ant_data_dict[ddi_index].keys())
                n_pol = meta_data['pol']
                n_points = int(np.sqrt(meta_data['time']))
                n_chan = meta_data['chan']
                if not simple_avg:
                        beams = np.empty((n_scan, n_chan, n_pol, n_points, n_points), dtype=np.cdouble)
                if average:
                        ant_data_array = np.empty((n_scan, 1, n_pol, n_points, n_points), dtype=np.cdouble)
                else:
                        ant_data_array = np.empty((n_scan, n_chan, n_pol, n_points, n_points), dtype=np.cdouble)

                time_centroid = []
                for scan_index, scan in enumerate(ant_data_dict[ddi].keys()):
                        # Grid (l, m) points
                        lm = ant_data_dict[ddi][scan].DIRECTIONAL_COSINES.values[:, np.newaxis, np.newaxis, :]
                        lm = np.tile(lm, (1, n_chan, n_pol, 1))
                        # VIS values
                        if simple_avg:
                                vis = ant_data_dict[ddi][scan].VIS.mean(dim='chan').values
                        else:
                                vis = ant_data_dict[ddi][scan].VIS.values
                                if lmscale:
                                        frequencies = ant_data_dict[ddi][scan].chan_freq.values
                                        # Reference frequency not available yet, using the middle channel frequency
                                        reffreq = frequencies[n_chan//2]
                                        # This can be vectorized for a speedup
                                        for chan in range(n_chan):
                                                lm[:, chan, :, :] *= frequencies[chan]/reffreq

                        time_centroid_index = int(ant_data_dict[ddi][scan].dims['time']*0.5)+1
        
                        time_centroid.append(ant_data_dict[ddi][scan].coords['time'][time_centroid_index].values)

                        for chan in range(n_chan):
                                for pol in range(n_pol):
                                        l_min_extent = meta_data['extent']['l']['min']
                                        l_max_extent = meta_data['extent']['l']['max']
        
                                        m_min_extent = meta_data['extent']['m']['min']
                                        m_max_extent = meta_data['extent']['m']['max']
        
                                        grid_x, grid_y = np.mgrid[l_min_extent:l_max_extent:n_points*1j, m_min_extent:m_max_extent:n_points*1j]

                                        grid = griddata(lm[:, chan, pol, :], vis[:, chan, pol], (grid_x, grid_y), method='nearest')
                                        if simple_avg:
                                                ant_data_array[scan_index, chan, pol, :, :] = grid
                                        else:
                                                beams[scan_index, chan, pol, :, :] = grid
                if simple_avg:
                        pass
                else:
                        if average:
                                ant_data_array[:, 0, :, :, :] = np.mean(beams, axis=1)
                        else:
                                ant_data_array = beams

                xds = xr.Dataset()
                xds.assign_coords({
                        'time_centroid': np.array(time_centroid), 
                        'ddi':list(map(int, ant_data_dict.keys())), 
                        'pol':[i for i in range(n_pol)]
                })

                xds['GRID'] = xr.DataArray(ant_data_array, dims=['time-centroid', 'chan', 'pol', 'l', 'm'])

                xds.attrs['ant_id'] = holog_chunk_params['ant_id']
                xds.attrs['time_centroid'] = np.array(time_centroid)

                hack_base_name = holog_chunk_params['hack_file'].split('.holog.zarr')[0]

                xds.to_zarr("{name}.image.zarr/{ant}/{ddi}".format(name=hack_base_name, ant=holog_chunk_params['ant_id'], ddi=ddi_index), mode='w', compute=True, consolidated=True)

def holog(hack_file, parallel=True, average=True, lmscale=False):
        """ Process holography data

        Args:
            hack_name (str): Hack file name
            parallel (bool, optional): Run in parallel with Dask or in serial. Defaults to True.
        """
        console.info("Loading holography file {hack_file} ...".format(hack_file=hack_file))

        try:
                if os.path.exists(hack_file):
                        hack_meta_data = "/".join((hack_file, ".hack_json"))


                        with open(hack_meta_data, "r") as json_file: 
                                hack_json = json.load(json_file)
    
                        ant_list = hack_json.keys()


                        holog_chunk_params = {}
                        holog_chunk_params['hack_file'] = hack_file
                        holog_chunk_params['average'] = average
                        holog_chunk_params['lmscale'] = lmscale

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
                        console.error("[holog] Holography file {hack_file} not found.".format(hack_file=hack_file))
                        raise FileNotFoundError()
        except Exception as error:
                console.error('[holog] {error}'.format(error=error))
