import numpy as np
import scipy.signal as scisig
import scipy.constants

from astrohack._utils._panel_classes.telescope import Telescope

import skriba.logger as logger


def _calculate_suggested_grid_parameter(parameter, quantile=0.005):
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


def _find_nearest(array, value):
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


# @njit(cache=False, nogil=True)
def _chunked_average(data, weight, avg_map, avg_freq):
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
            # Most probably will have to unravel assigment
            data_avg[:, avg_index, :] = (data_avg[:, avg_index, :] + weight[:, index, :] * data[:, index, :])
            weight_sum[:, avg_index, :] = weight_sum[:, avg_index, :] + weight[:, index, :]

            index = index + 1

        for time_index in range(n_time):
            for pol_index in range(n_pol):
                if weight_sum[time_index, avg_index, pol_index] == 0:
                    data_avg[time_index, avg_index, pol_index] = 0.0

                else:
                    data_avg[time_index, avg_index, pol_index] = (
                            data_avg[time_index, avg_index, pol_index] / weight_sum[
                        time_index, avg_index, pol_index])

    return data_avg, weight_sum


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

    index, _ = scisig.find_peaks(np.abs(array), height=cutoff)
    x, y = np.unravel_index(index, masked_data.shape)

    center = (masked_data.shape[0] // 2, masked_data.shape[1] // 2)

    distances = _calculate_euclidean_distance(x, y, center)
    index = distances.argmin()

    return masked_data[x[index], y[index]]


def _gauss_elimination_numpy(system, vector):
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


def _least_squares_fit(system, vector):
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

    result, residuals, _, _ = np.linalg.lstsq(system, vector, rcond=None)
    dof = len(vector) - len(result)
    if dof > 0:
        errs = (vector - np.dot(system, result)) / dof
    else:
        errs = (vector - np.dot(system, result))
    sigma2 = np.sum(errs ** 2)
    covar = np.linalg.inv(np.dot(system.T, system))
    variance = np.diagonal(sigma2 * covar)
    return result, variance, residuals


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
        results, variances, _ = _least_squares_fit(system, vector)
    return results, variances


def calculate_optimal_grid_parameters(pnt_map_dict, antenna_name, telescope_name, chan_freq):
    reference_frequency = np.median(chan_freq)
    reference_lambda = scipy.constants.speed_of_light / reference_frequency

    telescope = Telescope(telescope_name)

    # reference_lambda / D is the maximum cell size we should use so reduce is by 85% to get a safer answer.
    # Since this is just an estimate for the situation where the user doesn't specify a values, I am picking
    # a values according to the developer heuristic, ie. it seems to be good.
    cell_size = 0.85 * reference_lambda / telescope.diam

    # Get data range
    data_range = \
        (pnt_map_dict[antenna_name].POINTING_OFFSET.values[:, 1].max()
         - pnt_map_dict[antenna_name].POINTING_OFFSET.values[:, 1].min())

    try:
        n_pix = int(np.ceil(data_range / cell_size)) ** 2

    except ZeroDivisionError as e:
        logger.error(f"Zero division error, there was likely a problem calculating the data range.", verbose=True)
        raise ZeroDivisionError

    return n_pix, cell_size


def _significant_digits(x, digits):
    if np.isscalar(x):
        return _significant_digits_scalar(x, digits)

    else:
        return list(map(_significant_digits, x, [digits] * len(x)))


def _significant_digits_scalar(x, digits):
    if x == 0 or not np.isfinite(x):
        return x

    digits = int(digits - np.ceil(np.log10(abs(x))))

    return round(x, digits)
