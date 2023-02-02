import math
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
import scipy

from scipy.interpolate import griddata

from astrohack._utils import _system_message as console
from astrohack.dio import load_hack_file
from astrohack._utils._io import _read_dimensions_meta_data


def _calculate_euclidean_distance(x, y, center):
    """ Calculates the euclidean distance between a pair of pair of input points.

    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        center (tuple (float)): float tuple containing the coordinates to the center pixel

    Returns:
        float: euclidean distance of points from center pixel
    """

    return np.sqrt(np.power(x - center[0], 2) + np.power(y - center[1], 2))


def _apply_mask(data, scaling=0.5):
    """ Applies a cropping mask to the input data according to the scale factor

    Args:
        data (numpy,ndarray): numpy array containing the aperture grid.
        scaling (float, optional): Scale factor which is used to determine the amount of the data to crop, ex. scale=0.5 
                                   means to crop the data by 50%. Defaults to 0.5.

    Returns:
        numpy.ndarray: cropped aperture grid data
    """

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
    """ Search algorithm to determine the maximal signal peak in the beam pattern.

    Args:
        data (numpy.ndarray): beam data grid
        height (float, optional): Peak threshold. Looks for the maixmimum peak in data and uses a percentage of this 
                                  peak to determine a threhold for other peaks. Defaults to 0.5.
        scaling (float, optional): scaling factor for beam data cropping. Defaults to 0.5.

    Returns:
        float: peak maximum value
    """
    masked_data = _apply_mask(data, scaling=scaling)

    array = masked_data.flatten()
    cutoff = np.abs(array).max() * height

    index, _ = scipy.signal.find_peaks(np.abs(array), height=cutoff)
    x, y = np.unravel_index(index, masked_data.shape)

    center = (masked_data.shape[0] // 2, masked_data.shape[1] // 2)

    distances = _calculate_euclidean_distance(x, y, center)
    index = distances.argmin()

    return masked_data[x[index], y[index]]


def _calculate_aperture_pattern(grid, frequency, delta, padding_factor=20):
    """ Calcualtes the aperture illumination pattern from the beam data.

    Args:
        grid (numpy.ndarray): gridded beam data
        frequency (float): channel frequency
        delta (float): incremental spacing between lm values, ie. delta_l = l_(n+1) - l_(n)
        padding_factor (int, optional): Padding to apply to beam data grid before FFT. Padding is applied on outer edged of 
                                        each beam data grid and not between layers. Defaults to 20.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: aperture grid, u-coordinate array, v-coordinate array
    """
    console.info("Calculating aperture illumination pattern ...")

    assert grid.shape[-1] == grid.shape[-2]
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

    wave_length = scipy.constants.speed_of_light / frequency

    cell_size = wave_length / (image_size * delta)

    image_center = image_size // 2

    u = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell_size[0]
    v = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell_size[1]

    return aperture_grid, u, v

def _parallactic_derotation(data, parallactic_angle_dict):
    """ Uses samples of parallactic angle (PA) values to correct differences in PA between scans. The reference PA is selected 
        to be the first scans median parallactic angle. All values are rotated to this PA value using scypi.ndimage.rotate(...)

    Args:
        data (numpy.ndarray): beam data grid (scan, chan, pol, l, m)
        parallactic_angle_dict (dict): dictionary containing antenna selected xds from which the aprallactic angle samples 
                                       are retrieved ==> [scan](xds), here the scan referres to the scan values not the scan index.

    Returns:
        numpy.ndarray: rotation adjusted beam data grid
    """
    # Find the middle index of the array. This is calcualted because there might be a desire to change 
    # the array length at some point and I don't want to hard code the middle value.
    #
    # It is assumed, and should be true, that the parallacitc angle array size is consistent over scan.
    scans = list(parallactic_angle_dict.keys())
    # Get the median index for the first scan (this should be the same for every scan).
    median_index = len(parallactic_angle_dict[scans[0]].parallactic_samples)//2
    
    # This is the angle we will rotated the scans to.
    median_angular_reference = parallactic_angle_dict[scans[0]].parallactic_samples[median_index]
    
    for scan, scan_value in enumerate(scans):
        median_angular_offset = median_angular_reference - parallactic_angle_dict[scan_value].parallactic_samples[median_index]
        median_angular_offset *= 180/np.pi
            
        data[scan] = scipy.ndimage.rotate(input=data[scan, ...], angle=median_angular_offset, axes=(3, 2), reshape=False)
        
    return data


def _holog_chunk(holog_chunk_params):
    """ Process chunk holography data along the antenna axis. Works with holography file to properly grid , normalize, average and correct data
        and returns the aperture pattern.

    Args:
        holog_chunk_params (dict): Dictionary containing holography parameters.
    """

    _, ant_data_dict = load_hack_file(
        holog_chunk_params["hack_file"],
        dask_load=False,
        load_pnt_dict=False,
        ant_id=holog_chunk_params["ant_id"],
    )

    for ddi_index, ddi in enumerate(ant_data_dict.keys()):
        meta_data = _read_dimensions_meta_data(
            hack_file=holog_chunk_params["hack_file"],
            ddi=ddi_index,
            ant_id=holog_chunk_params["ant_id"],
        )

        n_scan = len(ant_data_dict[ddi_index].keys())
        n_pol = meta_data["pol"]
        n_points = int(np.sqrt(meta_data["time"]))
        n_chan = 1

        if holog_chunk_params["frequency_scaling"]:
            # This assumes the number of channels don't change over a measurement.
            n_chan = ant_data_dict[0][0].chan.values.shape[0]

        ant_data_array = np.empty(
            (n_scan, n_chan, n_pol, n_points, n_points), dtype=np.cdouble
        )

        time_centroid = []

        l_min_extent = meta_data["extent"]["l"]["min"]
        l_max_extent = meta_data["extent"]["l"]["max"]

        m_min_extent = meta_data["extent"]["m"]["min"]
        m_max_extent = meta_data["extent"]["m"]["max"]

        delta_lm = np.array(
            [
                (l_max_extent - l_min_extent) / n_points,
                (m_max_extent - m_min_extent) / n_points,
            ]
        )

        image_size = np.array([n_points, n_points])

        image_center = image_size // 2

        c = scipy.constants.speed_of_light

        step_l = (l_max_extent - l_min_extent) / n_points
        step_m = (m_max_extent - m_min_extent) / n_points

        l = np.arange(-image_center[0], image_size[0] - image_center[0]) * step_l
        m = np.arange(-image_center[1], image_size[1] - image_center[1]) * step_m

        grid_x, grid_y = np.meshgrid(l, m)

        for scan_index, scan in enumerate(ant_data_dict[ddi].keys()):

            frequencies = ant_data_dict[ddi][scan].chan.values

            lm = ant_data_dict[ddi][scan].DIRECTIONAL_COSINES.values[
                :, np.newaxis, np.newaxis, :
            ]
            lm = np.tile(lm, (1, n_chan, n_pol, 1))

            # VIS values
            if holog_chunk_params["frequency_scaling"]:
                vis = ant_data_dict[ddi][scan].VIS.values

                # Reference frequency not available yet, using the middle channel frequency
                reffreq = frequencies[n_chan // 2]

                # This can be vectorized for a speedup
                for chan in range(n_chan):
                    lm[:, chan, :, :] *= frequencies[chan] / reffreq
            else:
                vis = ant_data_dict[ddi][scan].VIS.mean(dim="chan").values
                vis = vis[:, np.newaxis, ...]

            time_centroid_index = ant_data_dict[ddi][scan].dims["time"] // 2

            time_centroid.append(
                ant_data_dict[ddi][scan].coords["time"][time_centroid_index].values
            )

            for chan in range(n_chan):
                for pol in range(n_pol):
                    grid = griddata(
                        lm[:, chan, pol, :],
                        vis[:, chan, pol],
                        (grid_x.T, grid_y.T),
                        method="nearest",
                    )
                    ant_data_array[scan_index, chan, pol, :, :] = grid

                xx_peak = _find_peak_beam_value(
                    ant_data_array[scan_index, chan, 0, ...], scaling=0.25
                )
                yy_peak = _find_peak_beam_value(
                    ant_data_array[scan_index, chan, 3, ...], scaling=0.25
                )

                normalization = np.abs(0.5 * (xx_peak + yy_peak))
                ant_data_array[scan_index, chan, ...] /= normalization

            if holog_chunk_params["frequency_scaling"]:
                ant_data_array = ant_data_array.mean(axis=1, keep_dims=True)

        ant_data_array = _parallactic_derotation(data=ant_data_array, parallactic_angle_dict=ant_data_dict[ddi])

        console.info(
            "[_holog_chunk] FFT padding factor {} ...".format(
                holog_chunk_params["padding_factor"]
            )
        )
        aperture_grid, u, v = _calculate_aperture_pattern(
            grid=ant_data_array,
            frequency=frequencies.mean(),
            delta=delta_lm,
            padding_factor=holog_chunk_params["padding_factor"],
        )

        xds = xr.Dataset()

        xds["GRID"] = xr.DataArray(
            ant_data_array, dims=["time-centroid", "chan", "pol", "l", "m"]
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

        xds = xds.assign_coords(coords)

        hack_base_name = holog_chunk_params["hack_file"].split(".holog.zarr")[0]

        xds.to_zarr(
            "{name}.image.zarr/{ant}/{ddi}".format(
                name=hack_base_name, ant=holog_chunk_params["ant_id"], ddi=ddi_index
            ),
            mode="w",
            compute=True,
            consolidated=True,
        )


def holog(hack_file, padding_factor=20, frequency_scaling=False, parallel=True):
    """Process holography data

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
            holog_chunk_params["hack_file"] = hack_file
            holog_chunk_params["padding_factor"] = padding_factor
            holog_chunk_params["frequency_scaling"] = frequency_scaling

            delayed_list = []

            for ant_id in ant_list:
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
                "[holog] Holography file {hack_file} not found.".format(
                    hack_file=hack_file
                )
            )
            raise FileNotFoundError()
    except Exception as error:
        console.error("[holog] {error}".format(error=error))
