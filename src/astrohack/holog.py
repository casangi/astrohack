import math
import json
import os

import dask
import dask.distributed
import scipy.fftpack
import scipy.constants
import scipy.signal
import matplotlib.pyplot as plt

import dask.array as da
import numpy as np
import xarray as xr
import scipy
import scipy.constants

from numba import njit

from skimage.draw import disk
from scipy.interpolate import griddata

from astrohack._utils import _system_message as console
from astrohack._utils._io import _read_meta_data

from astrohack._utils._holog import _load_holog_file
from astrohack._utils._holog import _holog_chunk

from astrohack._utils._panel import _phase_fitting

from memory_profiler import profile
   

#fp=open('holog.log','w+')
#@profile(stream=fp)
def holog(
    holog_file,
    grid_size=None,
    cell_size=None,
    padding_factor=50,
    parallel=True,
    grid_interpolation_mode="nearest",
    chan_average=True,
    chan_tolerance_factor=0.005,
    reference_scaling_frequency=None,
    scan_average = True,
    ant_list = None,
    to_stokes = False,
    phase_fit=True,
    apply_mask=False
):
    """Process holography data

    Args:
        holog_name (str): holog file name
        parallel (bool, optional): Run in parallel with Dask or in serial. Defaults to True.

        cell_size: float np.array 2x1
        grid_size: int np.array 2X1
    """
    console.info("Loading holography file {holog_file} ...".format(holog_file=holog_file))

    #try:
    if True:
        if os.path.exists(holog_file):
            json_data = "/".join((holog_file, ".holog_json"))
            meta_data = "/".join((holog_file, ".holog_attr"))
            
            with open(json_data, "r") as json_file:
                holog_json = json.load(json_file)
            
            with open(meta_data, "r") as meta_file:
                meta_data = json.load(meta_file)
            
            if  ant_list is None:
                ant_list = list(holog_json.keys())

            holog_chunk_params = {}
            holog_chunk_params["holog_file"] = holog_file
            holog_chunk_params["padding_factor"] = padding_factor
            holog_chunk_params["grid_interpolation_mode"] = grid_interpolation_mode
            holog_chunk_params["chan_average"] = chan_average
            holog_chunk_params["chan_tolerance_factor"] = chan_tolerance_factor
            holog_chunk_params["reference_scaling_frequency"] = reference_scaling_frequency
            holog_chunk_params["scan_average"] = scan_average
            holog_chunk_params["to_stokes"] = to_stokes
            holog_chunk_params["apply_mask"] = apply_mask
            holog_chunk_params["phase_fit"] = phase_fit

            
            if (cell_size is None) and (grid_size is None):
                ###To Do: Calculate one gridsize and cell_size for all ddi's, antennas, ect. Fix meta data ant_holog_dict gets overwritten for more than one ddi.
                
                n_points = int(np.sqrt(meta_data["n_time"]))
                grid_size = np.array([n_points, n_points])

                l_min_extent = meta_data["extent"]["l"]["min"]
                l_max_extent = meta_data["extent"]["l"]["max"]

                m_min_extent = meta_data["extent"]["m"]["min"]
                m_max_extent = meta_data["extent"]["m"]["max"]

                step_l = (l_max_extent - l_min_extent) / grid_size[0]
                step_m = (m_max_extent - m_min_extent) / grid_size[1]
                step_l = (step_l+step_m)/2
                step_m = step_l

                cell_size = np.array([step_l, step_m])

                holog_chunk_params["cell_size"] = cell_size
                holog_chunk_params["grid_size"] = grid_size

                console.info("Cell size: " + str(cell_size) + " Grid size " + str(grid_size))
            else:
                holog_chunk_params["cell_size"] = cell_size
                holog_chunk_params["grid_size"] = grid_size

            delayed_list = []
            
            

            for ant_id in ant_list:
                console.info("Processing ant_id: " + str(ant_id))
                holog_chunk_params["ant_id"] = ant_id

                if parallel:
                    delayed_list.append(
                        dask.delayed(_holog_chunk)(dask.delayed(holog_chunk_params))
                    )

                else:
                    _holog_chunk(holog_chunk_params)

            if parallel:
                dask.compute(delayed_list)

        else:
            console.error(
                "[holog] Holography file {holog_file} not found.".format(
                    holog_file=holog_file
                )
            )
            raise FileNotFoundError()