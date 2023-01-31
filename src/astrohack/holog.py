import math
import time
import json
import os

import dask
import dask.distributed
import scipy.fftpack
import scipy.constants
import scipy.signal

import dask.array as da
import numpy as np
import xarray as xr

from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator

from astrohack._utils import _system_message as console
from astrohack.dio import load_holog_file
from astrohack._utils._io import _read_dimensions_meta_data


def _calculate_euclidean_distance(x, y, center):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        center (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.sqrt(np.power(x - center[0], 2) + np.power(y - center[1], 2))


def _apply_mask(data, scaling=0.5):
    x, y = data.shape
    assert scaling > 0, console.error("Scaling must be > 0")

    mask = int(x // (1 // scaling))

    assert mask > 0, console.error(
        "Scaling values too small. Minimum values is:{}, though search may still fail due to lack of poitns.".format(
            1 / x
        )
    )

    start = int(x // 2 - mask // 2)
    return data[start : (start + mask), start : (start + mask)]


def _find_peak_beam_value(data, height=0.5, scaling=0.5):
    masked_data = _apply_mask(data, scaling=scaling)

    array = masked_data.flatten()
    cutoff = np.abs(array).max() * height

    index, _ = scipy.signal.find_peaks(np.abs(array), height=cutoff)
    x, y = np.unravel_index(index, masked_data.shape)

    center = (masked_data.shape[0] // 2, masked_data.shape[1] // 2)

    distances = _calculate_euclidean_distance(x, y, center)
    index = distances.argmin()

    return masked_data[x[index], y[index]]


def _calculate_aperture_pattern(grid, delta, padding_factor=20):
    console.info("Calculating aperture illumination pattern ...")

    assert grid.shape[-1] == grid.shape[-2] ###To do: why is this expected that l.shape == m.shape
    initial_dimension = grid.shape[-1]

    # Calculate padding as the nearest power of 2
    # k log (2) = log(N) => k = log(N)/log(2)
    # New shape => K = math.ceil(k) => shape = (K, K)

    k = np.log(initial_dimension * padding_factor) / np.log(2)
    K = math.ceil(k)

    padding = (np.power(2, K) - padding_factor * initial_dimension) // 2

    padded_grid = np.pad(
        array=grid,
        pad_width=[(0, 0), (0, 0), (0, 0), (padding, padding), (padding, padding)],
        mode="constant",
    )

    shifted = scipy.fftpack.fftshift(padded_grid)

    grid_fft = scipy.fftpack.fft2(shifted)

    aperture_grid = scipy.fftpack.fftshift(grid_fft)

    u_size = aperture_grid.shape[-2]
    v_size = aperture_grid.shape[-1]

    image_size = np.array([u_size, v_size])

    cell_size = 1 / (image_size * delta)

    image_center = image_size // 2

    u = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell_size[0]
    v = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell_size[1]

    return aperture_grid, u, v


def _calc_coords(image_size, cell_size):
    image_center = image_size // 2
    x = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell_size[0]
    y = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell_size[1]
    return x, y


def _chunked_average(data, weight, avg_map, avg_freq):

    avg_chan_indx = np.arange(avg_freq.shape[0])

    data_avg_shape = list(data.shape)
    n_time, n_chan, n_pol = data_avg_shape

    n_avg_chan = avg_freq.shape[0]
    data_avg_shape[1] = n_avg_chan  # Update new chan dim.

    data_avg = np.zeros(data_avg_shape, dtype=np.complex)
    weight_sum = np.zeros(data_avg_shape, dtype=np.float)

    indx = 0
    for avg_indx in avg_chan_indx:

        while (indx < n_chan) and (avg_map[indx] == avg_indx):
            # Most probably will have to unravel assigment
            data_avg[:, avg_indx, :] = (
                data_avg[:, avg_indx, :] + weight[:, indx, :] * data[:, indx, :]
            )
            weight_sum[:, avg_indx, :] = weight_sum[:, avg_indx, :] + weight[:, indx, :]
            indx = indx + 1

        for i_t in range(n_time):
            for i_p in range(n_pol):
                if weight_sum[i_t, avg_indx, i_p] == 0:
                    data_avg[i_t, avg_indx, i_p] = 0.0
                else:
                    data_avg[i_t, avg_indx, i_p] = (
                        data_avg[i_t, avg_indx, i_p] / weight_sum[i_t, avg_indx, i_p]
                    )

    return data_avg, weight_sum


def _holog_chunk(holog_chunk_params):
    """_summary_

    Args:
        holog_chunk_params (dict): Dictionary containing holography parameters.
    """
    c = scipy.constants.speed_of_light

    _, ant_data_dict = load_holog_file(
        holog_chunk_params["holog_file"],
        dask_load=False,
        load_pnt_dict=False,
        ant_id=holog_chunk_params["ant_id"],
    )

    # Calculate lm coordinates
    l, m = _calc_coords(
        holog_chunk_params["grid_size"], holog_chunk_params["cell_size"]
    )
    grid_l, grid_m = list(map(np.transpose, np.meshgrid(l, m)))

    for ddi_index, ddi in enumerate(ant_data_dict.keys()):
        meta_data = _read_dimensions_meta_data(
            holog_file=holog_chunk_params["holog_file"],
            ddi=ddi_index,
            ant_id=holog_chunk_params["ant_id"],
        )

        n_scan = len(ant_data_dict[ddi_index].keys())
        scan0 = list(ant_data_dict[ddi].keys())[
            0
        ]  # For a fixed ddi the frequency axis should not change over scans, consequently we only have to consider the first scan.
        freq_chan = ant_data_dict[ddi][scan0].chan.values
        n_chan = ant_data_dict[ddi][scan0].dims["chan"]
        n_pol = ant_data_dict[ddi][scan0].dims["pol"]

        if holog_chunk_params["chan_average"]:
            reference_scaling_frequency = holog_chunk_params[
                "reference_scaling_frequency"
            ]

            if reference_scaling_frequency is None:
                reference_scaling_frequency = np.mean(freq_chan)

            avg_chan_map, avg_freq = _create_average_chan_map(
                freq_chan, holog_chunk_params["chan_tolerance_factor"]
            )
            beam_grid = np.zeros(
                (n_scan,) + grid_l.shape + (1, n_pol), dtype=np.complex
            )  # Only a single channel left after averaging.
        else:
            beam_grid = np.zeros(
                (n_scan,) + grid_l.shape + (n_chan, n_pol), dtype=np.complex
            )


        time_centroid = []

        import matplotlib.pyplot as plt

        for scan_index, scan in enumerate(ant_data_dict[ddi].keys()):
            ant_xds = ant_data_dict[ddi][scan]


            ###To Do: Add flagging code

            # Grid the data
            vis = ant_xds.VIS.values
            lm = ant_xds.DIRECTIONAL_COSINES.values
            weight = ant_xds.WEIGHT

            if holog_chunk_params["chan_average"]:
                vis_avg, weight_sum = _chunked_average(
                    vis, weight, avg_chan_map, avg_freq
                )
                lm_freq_scaled = lm[:, :, None] * (
                    avg_freq / reference_scaling_frequency
                )

                n_chan = avg_freq.shape[0]
                
                for i_c in range(
                    n_chan
                ):  # Unavoidable for loop because lm chnage over frequency.
                    # Average scaled beams.
                    beam_grid[scan_index, :, :, 0, :] = (
                        beam_grid[scan_index, :, :, 0, :]
                        + np.sum(weight_sum[:, i_c, :], axis=0)[None, None, :]
                        * griddata(
                            lm_freq_scaled[:, :, i_c],
                            vis_avg[:, i_c, :],
                            (grid_l, grid_m),
                            method=holog_chunk_params["grid_interpolation_mode"],
                        )
                    )

                beam_grid[scan_index, ...] = beam_grid[scan_index, ...] / np.sum(
                    weight_sum, axis=(0, 1)
                )
                n_chan =  1 #avergaing now complete
                
                freq_chan = [np.mean(avg_freq)]
            else:
                beam_grid[scan_index, ...] = griddata(
                    lm,
                    vis,
                    (grid_l, grid_m),
                    method=holog_chunk_params["grid_interpolation_mode"],
                )

            beam_grid = np.moveaxis(beam_grid,(1,2),(3,4)) #["time-centroid", "chan", "pol", "l", "m"]


            time_centroid_index = ant_data_dict[ddi][scan].dims["time"] // 2

            time_centroid.append(
                ant_data_dict[ddi][scan].coords["time"][time_centroid_index].values
            )

        '''
            #Normalization
            for chan in range(n_chan): ###To Do: Vectorize channel axis
                xx_peak = _find_peak_beam_value(
                    beam_grid[scan_index, chan, 0, ...], scaling=0.25
                )
                yy_peak = _find_peak_beam_value(
                    beam_grid[scan_index, chan, 3, ...], scaling=0.25
                )

                normalization = np.abs(0.5 * (xx_peak + yy_peak))
                beam_grid[scan_index, chan, ...] /= normalization
        '''
        
        #Create aperture
        console.info(
            "[_holog_chunk] FFT padding factor {}".format(
                holog_chunk_params["padding_factor"]
            )
        )
        aperture_grid, u, v = _calculate_aperture_pattern(
            grid=beam_grid,
            delta=holog_chunk_params["cell_size"],
            padding_factor=holog_chunk_params["padding_factor"],
        )


        xds = xr.Dataset()
        xds["BEAM"] = xr.DataArray(
            beam_grid, dims=["time-centroid", "chan", "pol", "l", "m"]
        )
        xds["APERTURE"] = xr.DataArray(
            aperture_grid, dims=["time-centroid", "chan", "pol", "u", "v"]
        )

        xds.attrs["ant_id"] = holog_chunk_params["ant_id"]
        xds.attrs["time_centroid"] = np.array(time_centroid)

        coords = {}
        coords["time_centroid"] = np.array(time_centroid)
        coords["ddi"] = list(map(int, ant_data_dict.keys()))
        coords["pol"] = [i for i in range(n_pol)]
        coords["l"] = l
        coords["m"] = m
        coords["u"] = u
        coords["v"] = v
        coords["chan"] = freq_chan

        xds = xds.assign_coords(coords)

        holog_base_name = holog_chunk_params["holog_file"].split(".holog.zarr")[0]

        xds.to_zarr(
            "{name}.image.zarr/{ant}/{ddi}".format(
                name=holog_base_name, ant=holog_chunk_params["ant_id"], ddi=ddi_index
            ),
            mode="w",
            compute=True,
            consolidated=True,
        )

def holog(
    holog_file,
    padding_factor=20,
    frequency_scaling=False,
    parallel=True,
    cell_size=None,
    grid_size=None,
    grid_interpolation_mode="nearest",
    chan_average=False,
    chan_tolerance_factor=0.005,
    reference_scaling_frequency=None,
):
    """Process holography data

    Args:
        holog_name (str): holog file name
        parallel (bool, optional): Run in parallel with Dask or in serial. Defaults to True.

        cell_size: float np.array 2x1
        grid_size: int np.array 2X1
    """
    console.info(
        "Loading holography file {holog_file} ...".format(holog_file=holog_file)
    )

    # try:
    if True:
        if os.path.exists(holog_file):
            holog_meta_data = "/".join((holog_file, ".holog_json"))

            with open(holog_meta_data, "r") as json_file:
                holog_json = json.load(json_file)

            ant_list = list(holog_json.keys())

            holog_chunk_params = {}
            holog_chunk_params["holog_file"] = holog_file
            holog_chunk_params["padding_factor"] = padding_factor
            holog_chunk_params["frequency_scaling"] = frequency_scaling
            holog_chunk_params["grid_interpolation_mode"] = grid_interpolation_mode
            holog_chunk_params["chan_average"] = chan_average
            holog_chunk_params["chan_tolerance_factor"] = chan_tolerance_factor
            holog_chunk_params[
                "reference_scaling_frequency"
            ] = reference_scaling_frequency

            ### TO DO Caculate cell_size and grid_size

            if (cell_size is None) or (grid_size is None):
                ###To Do: Calculate one gridsize and cell_size for all ddi's, antennas, ect. Fix meta data ant_holog_dict gets overwritten for more than one ddi.
                a = 42

            else:
                holog_chunk_params["cell_size"] = cell_size
                holog_chunk_params["grid_size"] = grid_size

            delayed_list = []



            for ant_id in ant_list: #[ant_list[3]]:

                #print(ant_id)
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
    # except Exception as error:
    #    console.error("[holog] {error}".format(error=error))


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def _create_average_chan_map(freq_chan, chan_tolerance_factor):
    n_chan = len(freq_chan)
    cf_chan_map = np.zeros((n_chan,), dtype=int)

    orig_width = (np.max(freq_chan) - np.min(freq_chan)) / len(freq_chan)

    tol = np.max(freq_chan) * chan_tolerance_factor
    n_pb_chan = int(np.floor((np.max(freq_chan) - np.min(freq_chan)) / tol) + 0.5)

    # Create PB's for each channel
    if n_pb_chan == 0:
        n_pb_chan = 1

    if n_pb_chan >= n_chan:
        cf_chan_map = np.arange(n_chan)
        pb_freq = freq_chan
        return cf_chan_map, pb_freq

    pb_delta_bandwdith = (np.max(freq_chan) - np.min(freq_chan)) / n_pb_chan
    pb_freq = (
        np.arange(n_pb_chan) * pb_delta_bandwdith
        + np.min(freq_chan)
        + pb_delta_bandwdith / 2
    )

    cf_chan_map = np.zeros((n_chan,), dtype=int)
    for i in range(n_chan):
        cf_chan_map[i], _ = _find_nearest(pb_freq, freq_chan[i])

    return cf_chan_map, pb_freq


#            ant_xds = ant_xds.drop_vars(('chan','VIS','WEIGHT'))
#                ant_xds['VIS'] = (('time','chan','pol'),vis_avg)
#                ant_xds['WEIGHT'] = (('time','chan','pol'),weight_sum)
#                ant_xds.assign_coords({'chan':avg_freq})
#                #add coord avg_freq
