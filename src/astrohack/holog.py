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
import scipy.constants

from numba import njit

from scipy.interpolate import griddata

from astrohack._utils import _system_message as console
from astrohack.dio import _load_holog_file
from astrohack._utils._io import _read_meta_data

from astrohack._utils._phase_fitting import phase_fitting
from astrohack._classes.telescope import Telescope


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


def _calculate_aperture_pattern(grid, frequency, delta, padding_factor=50):
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
    console.info("[_calculate_aperture_pattern] Calculating aperture illumination pattern ...")

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

    shifted = scipy.fftpack.ifftshift(padded_grid)

    grid_fft = scipy.fftpack.fft2(shifted)

    aperture_grid = scipy.fftpack.fftshift(grid_fft)

    u_size = aperture_grid.shape[-2]
    v_size = aperture_grid.shape[-1]

    image_size = np.array([u_size, v_size])

    cell_size = 1 / (image_size * delta)

    image_center = image_size // 2

    u, v = _calc_coords(image_size, cell_size)

    return aperture_grid, u, v, cell_size

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


def _calc_coords(image_size, cell_size):
    """Calculate the center pixel of the image given a cell and image size

    Args:
        image_size (float): image size
        cell_size (float): cell size

    Returns:
        float, float: center pixel location in coordinates x, y
    """
    image_center = image_size // 2

    x = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell_size[0]
    y = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell_size[1]
    
    return x, y

#@njit(cache=False, nogil=True)
def _chunked_average(data, weight, avg_map, avg_freq):

    avg_chan_index = np.arange(avg_freq.shape[0])

    data_avg_shape = list(data.shape)
    n_time, n_chan, n_pol = data_avg_shape

    n_avg_chan = avg_freq.shape[0]
    
    # Update new chan dim.
    data_avg_shape[1] = n_avg_chan  

    data_avg = np.zeros(data_avg_shape, dtype=np.complex)
    weight_sum = np.zeros(data_avg_shape, dtype=np.float)

    index = 0
    for avg_index in avg_chan_index:

        while (index < n_chan) and (avg_map[index] == avg_index):

            # Most probably will have to unravel assigment
            data_avg[:, avg_index, :] = (data_avg[:, avg_index, :] + weight[:, index, :] * data[:, index, :])
            weight_sum[:, avg_index, :] = weight_sum[:, avg_index, :] + weight[:, index, :]
            
            index = index + 1

        for time_index in range(n_time):
            for pol_index in range(n_pol):
                if weight_sum[time_index, avg_index, pol_index] == 0:
                    data_avg[time_index, avg_index, pol_index] = 0.0

                else:
                    data_avg[time_index, avg_index, pol_index] = (data_avg[time_index, avg_index, pol_index] / weight_sum[time_index, avg_index, pol_index])

    return data_avg, weight_sum


def _holog_chunk(holog_chunk_params):
    """ Process chunk holography data along the antenna axis. Works with holography file to properly grid , normalize, average and correct data
        and returns the aperture pattern.

    Args:
        holog_chunk_params (dict): Dictionary containing holography parameters.
    """
    c = scipy.constants.speed_of_light

    holog_file, ant_data_dict = _load_holog_file(
        holog_chunk_params["holog_file"],
        dask_load=True,
        load_pnt_dict=False,
        ant_id=holog_chunk_params["ant_id"],
    )

    meta_data = _read_meta_data(holog_chunk_params["holog_file"])

    # Calculate lm coordinates
    l, m = _calc_coords(holog_chunk_params["grid_size"], holog_chunk_params["cell_size"])
    grid_l, grid_m = list(map(np.transpose, np.meshgrid(l, m)))
    
    #print(ant_data_dict)

    #for ddi in ant_data_dict.keys():
    for ddi in [1]:
        n_scan = len(ant_data_dict[ddi].keys())
        
        # For a fixed ddi the frequency axis should not change over scans, consequently we only have to consider the first scan.
        scan0 = list(ant_data_dict[ddi].keys())[0]  
        
        freq_chan = ant_data_dict[ddi][scan0].chan.values
        n_chan = ant_data_dict[ddi][scan0].dims["chan"]
        n_pol = ant_data_dict[ddi][scan0].dims["pol"]
        
        if holog_chunk_params["chan_average"]:
            reference_scaling_frequency = holog_chunk_params["reference_scaling_frequency"]

            if reference_scaling_frequency is None:
                reference_scaling_frequency = np.mean(freq_chan)

            avg_chan_map, avg_freq = _create_average_chan_map(freq_chan, holog_chunk_params["chan_tolerance_factor"])
            
            # Only a single channel left after averaging.
            beam_grid = np.zeros((n_scan,) + (1, n_pol) + grid_l.shape, dtype=np.complex)  
            
            
        else:
            beam_grid = np.zeros((n_scan,) + (n_chan, n_pol) + grid_l.shape, dtype=np.complex)

        time_centroid = []

        for scan_index, scan in enumerate(ant_data_dict[ddi].keys()):
            ant_xds = ant_data_dict[ddi][scan]
            
            ###To Do: Add flagging code

            # Grid the data
            vis = ant_xds.VIS.values
            lm = ant_xds.DIRECTIONAL_COSINES.values
            weight = ant_xds.WEIGHT

            if holog_chunk_params["chan_average"]:
                vis_avg, weight_sum = _chunked_average(vis, weight, avg_chan_map, avg_freq)
                lm_freq_scaled = lm[:, :, None] * (avg_freq / reference_scaling_frequency)

                n_chan = avg_freq.shape[0]

                # Unavoidable for loop because lm change over frequency.
                for chan_index in range(n_chan):
                    
                    # Average scaled beams.
                    beam_grid[scan_index, 0, :, :, :] = (beam_grid[scan_index, 0, :, :, :] + np.moveaxis(griddata(lm_freq_scaled[:, :, chan_index], vis_avg[:, chan_index, :], (grid_l, grid_m), method=holog_chunk_params["grid_interpolation_mode"],),(2),(0)))

                # Avergaing now complete
                n_chan =  1 
                
                freq_chan = [np.mean(avg_freq)]
            else:
                beam_grid[scan_index, ...] = np.moveaxis(griddata(lm, vis, (grid_l, grid_m), method=holog_chunk_params["grid_interpolation_mode"]), (0,1), (2,3))


            time_centroid_index = ant_data_dict[ddi][scan].dims["time"] // 2

            time_centroid.append(ant_data_dict[ddi][scan].coords["time"][time_centroid_index].values)

            for chan in range(n_chan): ### Todo: Vectorize scan and channel axis
                xx_peak = _find_peak_beam_value(beam_grid[scan_index, chan, 0, ...], scaling=0.25)
                
                yy_peak = _find_peak_beam_value(beam_grid[scan_index, chan, 3, ...], scaling=0.25)

                normalization = np.abs(0.5 * (xx_peak + yy_peak))
                beam_grid[scan_index, chan, ...] /= normalization
            
        beam_grid = _parallactic_derotation(data=beam_grid, parallactic_angle_dict=ant_data_dict[ddi])

        if holog_chunk_params["scan_average"]:          
            beam_grid = np.mean(beam_grid,axis=0)[None,...]
        
        # Current bottleneck
        aperture_grid, u, v, uv_cell_size = _calculate_aperture_pattern(
            grid=beam_grid,
            delta=holog_chunk_params["cell_size"],
            frequency = freq_chan,
            padding_factor=holog_chunk_params["padding_factor"],
        )

        console.info("[_holog_chunk] Applying phase correction ...")

        wavelength = scipy.constants.speed_of_light/freq_chan[0]
        
        print(meta_data["ant_map"])
        
        ant_name = 'EA24' #meta_data["ant_map"]['holog_grid_0']['ant']

        if  ant_name.__contains__('DV'):
            telescope_name = "_".join((meta_data['telescope_name'], 'DV'))

        elif  ant_name.__contains__('DA'):
            telescope_name = "_".join((meta_data['telescope_name'], 'DA'))
            
        elif  ant_name.__contains__('EA'):
            telescope_name = 'VLA'

        else:
            raise Exception("Antenna type not found: {}".format(meta_data['ant_name']))
        
        
        '''
        print(meta_data["ant_map"])

        if meta_data["ant_map"][str(holog_chunk_params["ant_id"])].__contains__('DV'):
            telescope_name = "_".join((meta_data['telescope_name'], 'DV'))

        elif meta_data["ant_map"][str(holog_chunk_params["ant_id"])].__contains__('DA'):
            telescope_name = "_".join((meta_data['telescope_name'], 'DA'))
            
        elif meta_data["ant_map"][str(holog_chunk_params["ant_id"])].__contains__('EA'):
            telescope_name = 'VLA'

        else:
            raise Exception("Antenna type not found: {}".format(meta_data['ant_name']))
        '''

        telescope = Telescope(telescope_name)

        phase_corrected_angle = np.empty_like(aperture_grid)

        aperture_radius = (0.75*telescope.diam)/wavelength 

        i = np.where(np.abs(u) < aperture_radius)[0]
        j = np.where(np.abs(v) < aperture_radius)[0]

        # Ensure the cut is square by using the smaller dimension. Phase correction fails otherwise.
        if i.shape[0] < j.shape[0]: 
            cut = i
        else:
            cut = j

        u_prime = u[cut.min():cut.max()]
        v_prime = v[cut.min():cut.max()]

        amplitude = np.absolute(aperture_grid[..., cut.min():cut.max(), cut.min():cut.max()])

        phase = np.angle(aperture_grid[..., cut.min():cut.max(), cut.min():cut.max()], deg=True)
        phase_corrected_angle = np.empty_like(phase)

        for time in range(amplitude.shape[0]):
            for chan in range(amplitude.shape[1]):
                for pol in range(amplitude.shape[2]):
                    
                    _, _, phase_corrected_angle[time, chan, pol, ...], _, _, _ = phase_fitting(
                        wavelength=wavelength, 
                        telescope=telescope, 
                        cellxy=uv_cell_size[0]*wavelength, # THIS HAS TO BE CHANGES, (X, Y) CELL SIZE ARE NOT THE SAME.
                        amplitude_image=amplitude[time, chan, pol, ...], 
                        phase_image=phase[time, chan, pol, ...], 
                        pointing_offset=False, 
                        focus_xy_offsets=False, 
                        focus_z_offset=False,
                        subreflector_tilt=False, 
                        cassegrain_offset=True
                    )
                    
        ###To Do: Add Paralactic angle as a non-dimension coordinate dependant on time.
        xds = xr.Dataset()

        xds["BEAM"] = xr.DataArray(beam_grid, dims=["time-centroid", "chan", "pol", "l", "m"])
        xds["APERTURE"] = xr.DataArray(aperture_grid, dims=["time-centroid", "chan", "pol", "u", "v"])
        xds["AMPLITUDE"] = xr.DataArray(amplitude, dims=["time-centroid", "chan", "pol", "u_prime", "v_prime"])

        xds["ANGLE"] = xr.DataArray(phase_corrected_angle, dims=["time-centroid", "chan", "pol", "u_prime", "v_prime"])

        xds.attrs["ant_id"] = holog_chunk_params["ant_id"]
        xds.attrs["ant_name"] = ant_name
        xds.attrs["telescope_name"] = meta_data['telescope_name']
        xds.attrs["time_centroid"] = np.array(time_centroid)

        coords = {}
        coords["time_centroid"] = np.array(time_centroid)
        coords["ddi"] = list(map(int, ant_data_dict.keys()))
        coords["pol"] = [i for i in range(n_pol)]
        coords["l"] = l
        coords["m"] = m
        coords["u"] = u
        coords["v"] = v
        coords["u_prime"] = u_prime
        coords["v_prime"] = v_prime
        coords["chan"] = freq_chan

        xds = xds.assign_coords(coords)

        holog_base_name = holog_chunk_params["holog_file"].split(".holog.zarr")[0]

        xds.to_zarr("{name}.image.zarr/{ant}/{ddi}".format(name=holog_base_name, ant=holog_chunk_params["ant_id"], ddi=ddi), mode="w", compute=True, consolidated=True)
   

def holog(
    holog_file,
    grid_size,
    cell_size=None,
    padding_factor=50,
    parallel=True,
    grid_interpolation_mode="nearest",
    chan_average=True,
    chan_tolerance_factor=0.005,
    reference_scaling_frequency=None,
    scan_average = True,
    ant_list = None
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

            
            if (cell_size is None):
                ###To Do: Calculate one gridsize and cell_size for all ddi's, antennas, ect. Fix meta data ant_holog_dict gets overwritten for more than one ddi.
                
                #n_points = int(np.sqrt(meta_data["n_time"])), Not always true

                l_min_extent = meta_data["extent"]["l"]["min"]
                l_max_extent = meta_data["extent"]["l"]["max"]

                m_min_extent = meta_data["extent"]["m"]["min"]
                m_max_extent = meta_data["extent"]["m"]["max"]

                #grid_size = np.array([n_points, n_points])

                step_l = (l_max_extent - l_min_extent) / grid_size[0]
                step_m = (m_max_extent - m_min_extent) / grid_size[1]

                cell_size = np.array([step_l, step_m])

                holog_chunk_params["cell_size"] = cell_size
                holog_chunk_params["grid_size"] = grid_size

            else:
                holog_chunk_params["cell_size"] = cell_size
                holog_chunk_params["grid_size"] = grid_size

            delayed_list = []
            
            

            for ant_id in ant_list:
                holog_chunk_params["ant_id"] = ant_id
                print(grid_size,cell_size,ant_id)

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
#    except Exception as error:
#        console.error("[holog] {error}".format(error=error))


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
