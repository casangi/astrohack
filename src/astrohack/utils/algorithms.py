import numpy
import numpy as np
import scipy.signal as scisig
import scipy.constants
import xarray as xr
from numba import njit

import toolviper.utils.logger as logger

from astrohack.utils.text import format_angular_distance, create_dataset_label
from astrohack.utils.conversion import convert_unit


def calculate_suggested_grid_parameter(parameter, quantile=0.005):
    import scipy

    logger.warning(parameter)
    # Determine skew properties and return median. Only do this if there are at least 5 values.
    if np.abs(scipy.stats.skew(parameter)) > 0.5:

        if scipy.stats.skew(parameter) > 0:
            cutoff = np.quantile(parameter, 1 - quantile)
            filtered_parameter = parameter[parameter <= cutoff]

        else:

            cutoff = np.quantile(parameter, quantile)
            filtered_parameter = parameter[parameter >= cutoff]

        # The culling of data that is extremely skewed can fail if the number of values is too small causing all the
        # data to be filtered. In this case just return the median which should be better with skewed data.
        if filtered_parameter.shape[0] == 0:
            logger.warning(
                "Filtering of outliers in skewed data has failed, returning median value for gridding parameter.")

            return np.median(parameter)

        return np.median(filtered_parameter)

    # Process as mean
    else:

        upper_cutoff = np.quantile(parameter, 1 - quantile)
        lower_cutoff = np.quantile(parameter, quantile)

        filtered_parameter = parameter[parameter >= lower_cutoff]
        filtered_parameter = filtered_parameter[filtered_parameter <= upper_cutoff]

        if filtered_parameter.shape[0] == 0:
            logger.warning("Filtering of outlier data has failed, returning mean value for gridding parameter.")
            return np.mean(parameter)

        return np.mean(filtered_parameter)


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

    assert scaling > 0, logger.error("Scaling must be > 0")

    mask = int(x // (1 // scaling))

    assert mask > 0, logger.error(
        "Scaling values too small. Minimum values is:{}, though search may still fail due to lack of points.".format(
            1 / x)
    )

    start = int(x // 2 - mask // 2)
    return data[start: (start + mask), start: (start + mask)]


def calc_coords(image_size, cell_size):
    """Calculate the center pixel of the image given a cell and image size

    Args:
        image_size (float): image size
        cell_size (float): cell size

    Returns:
        float, float: center pixel location in coordinates x, y
    """

    image_center = image_size // 2

    x = np.arange(-image_center[0], image_size[0] - image_center[0]) * cell_size[0] + cell_size[0]/2
    y = np.arange(-image_center[1], image_size[1] - image_center[1]) * cell_size[1] + cell_size[1]/2

    return x, y


def find_nearest(array, value):
    """ Find the nearest entry in array to that of value.

    Args:
        array (numpy.array): _description_
        value (float): _description_

    Returns:
        int, float: index, array value
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx, array[idx]


def chunked_average(data, weight, avg_map, avg_freq):
    """
    Average visibilities in chunks with close enough frequencies
    Args:
        data: Visibility data
        weight: Visibility weights
        avg_map: mapping of channels to average
        avg_freq:  new frequency ranges

    Returns:
    Chunked average of visibilities and weights
    """
    avg_chan_index = np.arange(avg_freq.shape[0])

    data_avg_shape = list(data.shape)
    n_time, n_chan, n_pol = data_avg_shape

    n_avg_chan = avg_freq.shape[0]

    # Update new chan dim.
    data_avg_shape[1] = n_avg_chan

    data_avg = np.zeros(data_avg_shape, dtype=np.complex128)
    weight_sum = np.zeros(data_avg_shape, dtype=np.float64)

    index = 0

    for avg_index in avg_chan_index:

        while (index < n_chan) and (avg_map[index] == avg_index):
            # Most probably will have to unravel assignment
            data_avg[:, avg_index, :] = (data_avg[:, avg_index, :] + weight[:, index, :] * data[:, index, :])
            weight_sum[:, avg_index, :] = weight_sum[:, avg_index, :] + weight[:, index, :]

            index = index + 1

        for time_index in range(n_time):
            for pol_index in range(n_pol):
                if weight_sum[time_index, avg_index, pol_index] == 0:
                    data_avg[time_index, avg_index, pol_index] = 0.0

                else:
                    data_avg[time_index, avg_index, pol_index] = (
                            data_avg[time_index, avg_index, pol_index] / weight_sum[time_index, avg_index, pol_index])

    return data_avg, weight_sum


def _calculate_euclidean_distance(x, y, center):
    """ Calculates the Euclidean distance between a pair of input points.

    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        center (tuple (float)): float tuple containing the coordinates to the center pixel

    Returns:
        float: euclidean distance of points from center pixel
    """

    return np.sqrt(np.power(x - center[0], 2) + np.power(y - center[1], 2))


def find_peak_beam_value(data, height=0.5, scaling=0.5):
    """ Search algorithm to determine the maximal signal peak in the beam pattern.

    Args:
        data (numpy.ndarray): beam data grid
        height (float, optional): Peak threshold. Looks for the maximum peak in data and uses a percentage of this
                                  peak to determine a threshold for other peaks. Defaults to 0.5.
        scaling (float, optional): scaling factor for beam data cropping. Defaults to 0.5.

    Returns:
        float: peak maximum value
    """
    masked_data = _apply_mask(data, scaling=scaling)

    array = masked_data.flatten()
    cutoff = np.abs(array).max() * height

    index, _ = scisig.find_peaks(np.abs(array), height=cutoff)
    x, y = np.unravel_index(index, masked_data.shape)

    center = (masked_data.shape[0] // 2, masked_data.shape[1] // 2)

    distances = _calculate_euclidean_distance(x, y, center)
    index = distances.argmin()

    return masked_data[x[index], y[index]]


def gauss_elimination(system, vector):
    """
    Gauss elimination solving of a system using numpy
    Args:
        system: System matrix to be solved
        vector: Vector that represents the right hand side of the system

    Returns:
    The solved system
    """
    inverse = np.linalg.inv(system)
    return np.dot(inverse, vector)


def least_squares(system, vector, return_sigma=False):
    """
    Least squares fitting of a system of linear equations
    The variances are simplified as the diagonal of the covariances
    Args:
        system: System matrix to be solved
        vector: Vector that represents the right hand side of the system
        return_sigma: Return sigma value

    Returns:
    The solved system, the variances of the system solution and the sum of the residuals
    """
    if len(system.shape) != 2:
        raise Exception('System must have 2 dimensions')
    if system.shape[0] < system.shape[1]:
        raise Exception('System must have at least the same number of rows as it has of columns')

    result, residuals, _, _ = np.linalg.lstsq(system, vector, rcond=None)
    dof = len(vector) - len(result)
    if dof > 0:
        errs = (vector - np.dot(system, result)) / dof
    else:
        errs = (vector - np.dot(system, result))
    sigma2 = np.sum(errs ** 2)
    covar = np.linalg.inv(np.dot(system.T, system))
    variance = np.diag(sigma2 * covar)

    if return_sigma:
        return result, variance, residuals, np.sqrt(sigma2)
    else:
        return result, variance, residuals


@njit(cache=False, nogil=True)
def least_squares_jit(system, vector):
    """
    Least squares fitting of a system of linear equations
    The variances are simplified as the diagonal of the covariances
    Args:
        system: System matrix to be solved
        vector: Vector that represents the right hand side of the system

    Returns:
    The solved system, the variances of the system solution and the sum of the residuals
    """
    if len(system.shape) != 2:
        raise Exception('System must have 2 dimensions')
    if system.shape[0] < system.shape[1]:
        raise Exception('System must have at least the same number of rows as it has of columns')

    result, residuals, _, _ = np.linalg.lstsq(system, vector)
    dof = len(vector) - len(result)
    if dof > 0:
        errs = (vector - np.dot(system, result)) / dof
    else:
        errs = (vector - np.dot(system, result))
    sigma2 = np.sum(errs ** 2)
    covar = np.linalg.inv(np.dot(system.T, system))
    variance = np.diag(sigma2 * covar)
    return result, variance, residuals, np.sqrt(sigma2)


def _least_squares_fit_block(system, vector):
    """
    Least squares fitting of a system of linear equations
    The variances are simplified as the diagonal of the covariances
    Args:
        system: System matrix to be solved
        vector: Vector that represents the right hand side of the system

    Returns:
    The solved system and the variances of the system solution
    """
    if len(system.shape) < 2:
        raise Exception('System block must have at least 2 dimensions')
    if system.shape[-2] < system.shape[-1]:
        raise Exception('Systems must have at least the same number of rows as they have of columns')
    shape = system.shape
    results = np.zeros_like(vector)
    variances = np.zeros_like(vector)
    if len(shape) > 2:
        for it0 in range(shape[0]):
            results[it0], variances[it0] = _least_squares_fit_block(system[it0], vector[it0])
    else:
        results, variances, _ = least_squares(system, vector)
    return results, variances


def calculate_optimal_grid_parameters(pnt_map_dict, antenna_name, telescope_diameter, chan_freq, ddi):
    reference_frequency = np.median(chan_freq)
    reference_lambda = scipy.constants.speed_of_light / reference_frequency

    # reference_lambda / D is the maximum cell size we should use so reduce is by 85% to get a safer answer.
    # Since this is just an estimate for the situation where the user doesn't specify a values, I am picking
    # a values according to the developer heuristic, i.e. it seems to be good.
    cell_size = 0.85 * reference_lambda / telescope_diameter

    # Get data range
    data_range = \
        (pnt_map_dict[antenna_name].POINTING_OFFSET.values[:, 1].max()
         - pnt_map_dict[antenna_name].POINTING_OFFSET.values[:, 1].min())

    logger.info(f'{create_dataset_label(antenna_name, ddi)}: Cell size {format_angular_distance(cell_size)}, '
                f'FOV: {format_angular_distance(data_range)}')
    # logger.info(f"cell_size: {cell_size}")
    # logger.info(f"data_range: {data_range}")

    try:
        n_pix = int(np.ceil(data_range / cell_size)) ** 2

    except ZeroDivisionError:
        logger.error(f"Zero division error, there was likely a problem calculating the data range.", verbose=True)
        raise ZeroDivisionError

    return n_pix, cell_size


def compute_average_stokes_visibilities(vis, stokes):
    n_chan = len(vis.chan)
    chan_ave_vis = vis.mean(dim='chan', skipna=True)
    amp, pha, sigma_amp, sigma_pha = compute_stokes(
        chan_ave_vis['VIS'].values,
        n_chan * chan_ave_vis['WEIGHT'].values,
        chan_ave_vis.pol
    )

    coords = {
        'time': chan_ave_vis.time,
        'pol': ['I', 'Q', 'U', 'V']
    }

    xds = xr.Dataset()
    xds = xds.assign_coords(coords)
    xds["AMPLITUDE"] = xr.DataArray(amp, dims=["time", 'pol'], coords=coords)
    xds["PHASE"] = xr.DataArray(pha, dims=["time", 'pol'], coords=coords)
    xds['SIGMA_AMP'] = xr.DataArray(sigma_amp, dims=["time", 'pol'], coords=coords)
    xds['SIGMA_PHA'] = xr.DataArray(sigma_amp, dims=["time", 'pol'], coords=coords)
    xds.attrs['frequency'] = np.mean(vis.chan) / 1e9  # in GHz
    return xds.sel(pol=stokes)


def compute_stokes(data, weight, pol_axis):
    stokes_data = np.zeros_like(data)
    weight[weight == 0] = np.nan
    sigma = np.sqrt(1 / weight)
    sigma_amp = np.zeros_like(weight)
    if 'RR' in pol_axis:
        stokes_data[:, 0] = (data[:, 0] + data[:, 3]) / 2
        sigma_amp[:, 0] = (sigma[:, 0] + sigma[:, 3]) / 2
        stokes_data[:, 1] = (data[:, 1] + data[:, 2]) / 2
        sigma_amp[:, 1] = (sigma[:, 1] + sigma[:, 2]) / 2
        stokes_data[:, 2] = 1j * (data[:, 1] - data[:, 2]) / 2
        sigma_amp[:, 2] = sigma_amp[:, 1]
        stokes_data[:, 3] = (data[:, 0] - data[:, 3]) / 2
        sigma_amp[:, 0] = (sigma[:, 0] + sigma[:, 3]) / 2
    elif 'XX' in pol_axis:
        stokes_data[:, 0] = (data[:, 0] + data[:, 3]) / 2
        sigma_amp[:, 0] = (sigma[:, 0] + sigma[:, 3]) / 2
        stokes_data[:, 1] = (data[:, 0] - data[:, 3]) / 2
        sigma_amp[:, 1] = sigma_amp[:, 0]
        stokes_data[:, 2] = (data[:, 1] + data[:, 2]) / 2
        sigma_amp[:, 2] = (sigma[:, 1] + sigma[:, 2]) / 2
        stokes_data[:, 3] = 1j * (data[:, 1] - data[:, 2]) / 2
        sigma_amp[:, 3] = sigma_amp[:, 2]
    else:
        raise Exception("Pol not supported " + str(pol_axis))
    stokes_amp = np.absolute(stokes_data)
    stokes_pha = np.angle(stokes_data, deg=True)
    sigma_amp[~np.isfinite(sigma_amp)] = np.nan
    sigma_amp[sigma_amp == 0] = np.nan
    snr = stokes_amp / sigma_amp
    cst = np.sqrt(9 / (2 * np.pi ** 3))
    # Both sigmas here are probably wrong because of the uncertainty of how weights are stored.
    sigma_pha = np.pi / np.sqrt(3) * (1 - cst * snr)
    sigma_pha = np.where(snr > 2.5, 1 / snr, sigma_pha)
    sigma_pha *= convert_unit('rad', 'deg', 'trigonometric')
    return stokes_amp, stokes_pha, sigma_amp, sigma_pha


def compute_antenna_relative_off(antenna, tel_lon, tel_lat, tel_rad, scaling=1.0):
    """
    Computes an antenna offset to the array center
    Args:
        antenna: Antenna information dictionary
        tel_lon: array center longitude
        tel_lat: array center latitude
        tel_rad: array center's distance to the center of the earth
        scaling: scale factor

    Returns:
    Offset to the east, Offset to the North, elevation offset and distance to array center
    """
    antenna_off_east = tel_rad * (antenna['longitude'] - tel_lon) * np.cos(tel_lat)
    antenna_off_north = tel_rad * (antenna['latitude'] - tel_lat)
    antenna_off_ele = antenna['radius'] - tel_rad
    antenna_dist = np.sqrt(antenna_off_east ** 2 + antenna_off_north ** 2 + antenna_off_ele ** 2)
    return antenna_off_east * scaling, antenna_off_north * scaling, antenna_off_ele * scaling, antenna_dist * scaling


def rotate_to_gmt(positions, errors, longitude):
    """
    Rotate geometrical delays from antenna reference frame to GMT reference frame
    Args:
        positions: geometrical delays
        errors: geometrical delay errors
        longitude: Antenna longitude

    Returns:
    Rotated geometrical delays and associated errors
    """
    xpos, ypos = positions[0:2]
    delta_lon = longitude
    cosdelta = np.cos(delta_lon)
    sindelta = np.sin(delta_lon)
    newpositions = positions
    newpositions[0] = xpos * cosdelta - ypos * sindelta
    newpositions[1] = xpos * sindelta + ypos * cosdelta
    newerrors = errors
    xerr, yerr = errors[0:2]
    newerrors[0] = np.sqrt((xerr * cosdelta) ** 2 + (yerr * sindelta) ** 2)
    newerrors[1] = np.sqrt((yerr * cosdelta) ** 2 + (xerr * sindelta) ** 2)
    return newpositions, newerrors
