import numpy
import numpy as np
import scipy.signal as scisig
import scipy.constants
import xarray as xr
from numba import njit

import toolviper.utils.logger as logger

from astrohack.utils.text import format_angular_distance, create_dataset_label
from astrohack.utils.conversion import convert_unit
from astrohack.utils.constants import pi, twopi


def tokenize_version_number(version_number):
    """
    Tokenize a version number into an array of integers
    Args:
        version_number: The astrohack number version to be tokenized

    Returns:
        Tokenized version number in 3 element numpy array of integers
    """
    if not isinstance(version_number, str):
        raise Exception(f"Version number: {version_number} is not a string")
    split = version_number.split(".")
    if len(split) != 3:
        raise Exception(f"Version number: {version_number} is badly formated")
    tokenized = np.ndarray([3], dtype=int)

    for itoken in range(len(split)):
        try:
            tokenized[itoken] = int(split[itoken])
        except ValueError:
            raise Exception(
                f"Version number: {version_number} is not composed of integers"
            )
    return tokenized


def data_from_version_needs_patch(version_to_check, patched_version):
    """
    Check if data from a version needs to be patched according to a reference patch version
    Args:
        version_to_check: The version that is being tested
        patched_version: Reference version at which the patch is no longer needed

    Returns:
        True if the checked version is from before the patch False elsewise
    """
    check = tokenize_version_number(version_to_check)
    patched = tokenize_version_number(patched_version)
    for itoken in range(3):
        if check[itoken] < patched[itoken]:
            return True
        elif check[itoken] > patched[itoken]:
            return False
        else:
            continue
    return False


def _apply_mask(data, scaling=0.5):
    """Applies a cropping mask to the input data according to the scale factor

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
            1 / x
        )
    )

    start = int(x // 2 - mask // 2)
    return data[start : (start + mask), start : (start + mask)]


def calc_coords(image_size, cell_size):
    """Calculate the center pixel of the image given a cell and image size

    Args:
        image_size (float): image size
        cell_size (float): cell size

    Returns:
        float, float: center pixel location in coordinates x, y
    """

    image_center = image_size // 2

    x = (
        np.arange(-image_center[0], image_size[0] - image_center[0]) * cell_size[0]
        + cell_size[0] / 2
    )
    y = (
        np.arange(-image_center[1], image_size[1] - image_center[1]) * cell_size[1]
        + cell_size[1] / 2
    )

    return x, y


def find_nearest(array, value):
    """Find the nearest entry in array to that of value.

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
            data_avg[:, avg_index, :] = (
                data_avg[:, avg_index, :] + weight[:, index, :] * data[:, index, :]
            )
            weight_sum[:, avg_index, :] = (
                weight_sum[:, avg_index, :] + weight[:, index, :]
            )

            index = index + 1

        for time_index in range(n_time):
            for pol_index in range(n_pol):
                if weight_sum[time_index, avg_index, pol_index] == 0:
                    data_avg[time_index, avg_index, pol_index] = 0.0

                else:
                    data_avg[time_index, avg_index, pol_index] = (
                        data_avg[time_index, avg_index, pol_index]
                        / weight_sum[time_index, avg_index, pol_index]
                    )

    return data_avg, weight_sum


def _calculate_euclidean_distance(x, y, center):
    """Calculates the Euclidean distance between a pair of input points.

    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        center (tuple (float)): float tuple containing the coordinates to the center pixel

    Returns:
        float: euclidean distance of points from center pixel
    """

    return np.sqrt(np.power(x - center[0], 2) + np.power(y - center[1], 2))


def find_peak_beam_value(data, height=0.5, scaling=0.5):
    """Search algorithm to determine the maximal signal peak in the beam pattern.

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
        raise Exception("System must have 2 dimensions")
    if system.shape[0] < system.shape[1]:
        raise Exception(
            "System must have at least the same number of rows as it has of columns"
        )

    result, residuals, _, _ = np.linalg.lstsq(system, vector, rcond=None)
    dof = len(vector) - len(result)
    if dof > 0:
        errs = (vector - np.dot(system, result)) / dof
    else:
        errs = vector - np.dot(system, result)
    sigma2 = np.sum(errs**2)
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
        raise Exception("System must have 2 dimensions")
    if system.shape[0] < system.shape[1]:
        raise Exception(
            "System must have at least the same number of rows as it has of columns"
        )

    result, residuals, _, _ = np.linalg.lstsq(system, vector)
    dof = len(vector) - len(result)
    if dof > 0:
        errs = (vector - np.dot(system, result)) / dof
    else:
        errs = vector - np.dot(system, result)
    sigma2 = np.sum(errs**2)
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
        raise Exception("System block must have at least 2 dimensions")
    if system.shape[-2] < system.shape[-1]:
        raise Exception(
            "Systems must have at least the same number of rows as they have of columns"
        )
    shape = system.shape
    results = np.zeros_like(vector)
    variances = np.zeros_like(vector)
    if len(shape) > 2:
        for it0 in range(shape[0]):
            results[it0], variances[it0] = _least_squares_fit_block(
                system[it0], vector[it0]
            )
    else:
        results, variances, _ = least_squares(system, vector)
    return results, variances


def calculate_optimal_grid_parameters(
    pnt_map_dict, antenna_name, telescope_diameter, chan_freq, ddi
):
    reference_frequency = np.median(chan_freq)
    reference_lambda = scipy.constants.speed_of_light / reference_frequency

    # reference_lambda / D is the maximum cell size we should use so reduce is by 85% to get a safer answer.
    # Since this is just an estimate for the situation where the user doesn't specify a values, I am picking
    # a values according to the developer heuristic, i.e. it seems to be good.
    cell_size = 0.85 * reference_lambda / telescope_diameter

    # Get data range
    data_range = (
        pnt_map_dict[antenna_name].POINTING_OFFSET.values[:, 1].max()
        - pnt_map_dict[antenna_name].POINTING_OFFSET.values[:, 1].min()
    )

    logger.info(
        f"{create_dataset_label(antenna_name, ddi)}: Cell size {format_angular_distance(cell_size)}, "
        f"FOV: {format_angular_distance(data_range)}"
    )
    # logger.info(f"cell_size: {cell_size}")
    # logger.info(f"data_range: {data_range}")

    try:
        n_pix = int(np.ceil(data_range / cell_size)) ** 2

    except ZeroDivisionError:
        logger.error(
            f"Zero division error, there was likely a problem calculating the data range.",
            verbose=True,
        )
        raise ZeroDivisionError

    return n_pix, cell_size


def compute_average_stokes_visibilities(vis, stokes):
    n_chan = len(vis.chan)
    chan_ave_vis = vis.mean(dim="chan", skipna=True)
    amp, pha, sigma_amp, sigma_pha = compute_stokes(
        chan_ave_vis["VIS"].values,
        n_chan * chan_ave_vis["WEIGHT"].values,
        chan_ave_vis.pol,
    )

    coords = {"time": chan_ave_vis.time, "pol": ["I", "Q", "U", "V"]}

    xds = xr.Dataset()
    xds = xds.assign_coords(coords)
    xds["AMPLITUDE"] = xr.DataArray(amp, dims=["time", "pol"], coords=coords)
    xds["PHASE"] = xr.DataArray(pha, dims=["time", "pol"], coords=coords)
    xds["SIGMA_AMP"] = xr.DataArray(sigma_amp, dims=["time", "pol"], coords=coords)
    xds["SIGMA_PHA"] = xr.DataArray(sigma_amp, dims=["time", "pol"], coords=coords)
    xds.attrs["frequency"] = np.mean(vis.chan) / 1e9  # in GHz
    return xds.sel(pol=stokes)


def compute_stokes(data, weight, pol_axis):
    stokes_data = np.zeros_like(data)
    weight[weight == 0] = np.nan
    sigma = np.sqrt(1 / weight)
    sigma_amp = np.zeros_like(weight)
    if "RR" in pol_axis:
        stokes_data[:, 0] = (data[:, 0] + data[:, 3]) / 2
        sigma_amp[:, 0] = (sigma[:, 0] + sigma[:, 3]) / 2
        stokes_data[:, 1] = (data[:, 1] + data[:, 2]) / 2
        sigma_amp[:, 1] = (sigma[:, 1] + sigma[:, 2]) / 2
        stokes_data[:, 2] = 1j * (data[:, 1] - data[:, 2]) / 2
        sigma_amp[:, 2] = sigma_amp[:, 1]
        stokes_data[:, 3] = (data[:, 0] - data[:, 3]) / 2
        sigma_amp[:, 0] = (sigma[:, 0] + sigma[:, 3]) / 2
    elif "XX" in pol_axis:
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
    cst = np.sqrt(9 / (2 * np.pi**3))
    # Both sigmas here are probably wrong because of the uncertainty of how weights are stored.
    sigma_pha = np.pi / np.sqrt(3) * (1 - cst * snr)
    sigma_pha = np.where(snr > 2.5, 1 / snr, sigma_pha)
    sigma_pha *= convert_unit("rad", "deg", "trigonometric")
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
    antenna_off_east = tel_rad * (antenna["longitude"] - tel_lon) * np.cos(tel_lat)
    antenna_off_north = tel_rad * (antenna["latitude"] - tel_lat)
    antenna_off_ele = antenna["radius"] - tel_rad
    antenna_dist = np.sqrt(
        antenna_off_east**2 + antenna_off_north**2 + antenna_off_ele**2
    )
    return (
        antenna_off_east * scaling,
        antenna_off_north * scaling,
        antenna_off_ele * scaling,
        antenna_dist * scaling,
    )


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


def data_statistics(data_array):
    data_stats = {
        "mean": np.nanmean(data_array),
        "median": np.nanmedian(data_array),
        "rms": np.nanstd(data_array),
        "min": np.nanmin(data_array),
        "max": np.nanmax(data_array),
    }
    return data_stats


def phase_wrapping(phase):
    """
    Wraps phase to the -pi to pi interval
    Args:
        phase: phase to be wrapped

    Returns:
    Phase wrapped to the -pi to pi interval
    """
    return (phase + pi) % twopi - pi


def create_coordinate_images(x_axis, y_axis, create_polar_coordinates=False):
    """
    Takes two axes and creates 2D representation of the image coordinates
    Args:
        x_axis: X axis
        y_axis: Y axis
        create_polar_coordinates: Also create polar coordinates images?

    Returns:
        x_mesh and y_mesh, plus radius_mesh and polar_angle_mesh if create_polar_coordinates
    """
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis, indexing="ij")
    if create_polar_coordinates:
        radius_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
        polar_angle_mesh = np.arctan2(x_mesh, -y_mesh) - np.pi / 2
        polar_angle_mesh = np.where(
            polar_angle_mesh < 0, polar_angle_mesh + twopi, polar_angle_mesh
        )
        return x_mesh, y_mesh, radius_mesh, polar_angle_mesh
    else:
        return x_mesh, y_mesh


def create_aperture_mask(
    x_axis,
    y_axis,
    inner_rad,
    outer_rad,
    arm_width=None,
    arm_angle=0,
    return_polar_meshes=False,
):
    """
    Create a basic aperture mask with support for feed supporting arms shadows
    Args:
        x_axis: The X axis of the Aperture
        y_axis: The Y axis of the Aperture
        inner_rad: The innermost radius for valid data in aperture
        outer_rad: The outermost radius for valid data in aperture
        arm_width: The width of the feed arm shadows, can be a list with limiting radii or a single value.
        arm_angle: The angle between the arm shadows and the X axis
        return_polar_meshes: Return the radial and polar meshes to avoid duplicate computations.

    Returns:

    """
    x_mesh, y_mesh, radius_mesh, polar_angle_mesh = create_coordinate_images(
        x_axis, y_axis, create_polar_coordinates=True
    )
    mask = np.full_like(radius_mesh, True, dtype=bool)
    mask = np.where(radius_mesh > outer_rad, False, mask)
    mask = np.where(radius_mesh < inner_rad, False, mask)

    if arm_width is None:
        pass
    elif isinstance(arm_width, (float, int)):
        mask = _arm_shadow_masking(
            mask,
            x_mesh,
            y_mesh,
            radius_mesh,
            inner_rad,
            outer_rad,
            arm_width,
            arm_angle,
        )
    elif isinstance(arm_width, list):
        for section in arm_width:
            minradius, maxradius, width = section
            mask = _arm_shadow_masking(
                mask,
                x_mesh,
                y_mesh,
                radius_mesh,
                minradius,
                maxradius,
                width,
                arm_angle,
            )

    else:
        raise Exception(
            f"Don't know how to handle an arm width of class {type(arm_width)}"
        )

    if return_polar_meshes:
        return mask, radius_mesh, polar_angle_mesh
    else:
        return mask


def _arm_shadow_masking(
    inmask, x_mesh, y_mesh, radius_mesh, minradius, maxradius, width, angle
):
    radial_mask = np.where(radius_mesh < minradius, False, inmask)
    radial_mask = np.where(radius_mesh >= maxradius, False, radial_mask)
    if angle % pi / 2 == 0:
        oumask = np.where(
            np.bitwise_and(np.abs(x_mesh) < width / 2.0, radial_mask), False, inmask
        )
        oumask = np.where(
            np.bitwise_and(np.abs(y_mesh) < width / 2.0, radial_mask), False, oumask
        )
    else:
        # first shadow
        coeff = np.tan(angle % pi)
        distance = np.abs((coeff * x_mesh - y_mesh) / np.sqrt(coeff**2 + 1))
        oumask = np.where(
            np.bitwise_and(distance < width / 2.0, radial_mask), False, inmask
        )
        # second shadow
        coeff = np.tan(angle % pi + pi / 2)
        distance = np.abs((coeff * x_mesh - y_mesh) / np.sqrt(coeff**2 + 1))
        oumask = np.where(
            np.bitwise_and(distance < width / 2.0, radial_mask), False, oumask
        )
    return oumask


def are_axes_equal(axis_a, axis_b):
    if axis_a.shape[0] != axis_b.shape[0]:
        return False
    return np.all(axis_a == axis_b)


def create_2d_array_reconstruction_array(x_axis, y_axis, mask):
    # Creating grid reconstruction
    n_valid = np.sum(mask)
    x_idx = np.arange(x_axis.shape[0], dtype=int)
    y_idx = np.arange(y_axis.shape[0], dtype=int)
    x_idx_grd, y_idx_grd = np.meshgrid(x_idx, y_idx, indexing="ij")
    uv_idx_grid = np.empty([n_valid, 2], dtype=int)
    uv_idx_grid[:, 0] = x_idx_grd[mask]
    uv_idx_grid[:, 1] = y_idx_grd[mask]

    return uv_idx_grid
