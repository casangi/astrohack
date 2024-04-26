import math
import scipy
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from numba import njit
import scipy.fftpack
from matplotlib import pyplot as plt

import graphviper.utils.logger as logger

from skimage.draw import disk
from astrohack.utils.algorithms import calc_coords


def parallactic_derotation(data, parallactic_angle_dict):
    """ Uses samples of parallactic angle (PA) values to correct differences in PA between maps. The reference PA is
    selected to be the first maps median parallactic angle. All values are rotated to this PA value using
    scipy.ndimage.rotate(...)

    Args: data (numpy.ndarray): beam data grid (map, chan, pol, l, m) parallactic_angle_dict (dict): dictionary
    containing antenna selected xds from which the parallactic angle samples are retrieved ==> [map](xds),
    here the map referred to the map values not the map index.

    Returns:
        numpy.ndarray: rotation adjusted beam data grid
    """
    # Find the middle index of the array. This is calculated because there might be a desire to change
    # the array length at some point and I don't want to hard code the middle value.
    #
    # It is assumed, and should be true, that the parallacitc angle array size is consistent over map.
    maps = list(parallactic_angle_dict.keys())

    # Get the median index for the first map (this should be the same for every map).
    median_index = len(parallactic_angle_dict[maps[0]].parallactic_samples) // 2

    # This is the angle we will rotate the maps to.
    # median_angular_reference = parallactic_angle_dict[maps[0]].parallactic_samples[median_index]

    for mapping, map_value in enumerate(maps):
        # median_angular_offset = median_angular_reference - parallactic_angle_dict[map_value].parallactic_samples[
        # median_index] median_angular_offset *= 180/np.pi

        # parallactic_angle = 360 - parallactic_angle_dict[map_value].parallactic_samples[median_index]*180/np.pi

        data[mapping] = scipy.ndimage.rotate(input=data[mapping, ...], angle=90, axes=(3, 2), reshape=False)

    return data


def mask_circular_disk(center, radius, array, mask_value=np.nan):
    """ Create a mask to trim an image

    Args:
        center (tuple): tuple describing the center of the image
        radius (int): disk radius
        array (numpy.ndarray): data array to mask
        mask_value (int, optional): Value to set masked value to. Defaults to 1.

    Returns:
        _type_: _description_
    """
    shape = np.array(array.shape[-2:])

    if center is None:
        center = shape // 2

    r, c = disk(center, radius, shape=shape)
    mask = np.zeros(shape, dtype=array.dtype)
    mask[r, c] = 1

    mask = np.tile(mask, reps=(array.shape[:-2] + (1, 1)))

    mask[mask == 0] = mask_value

    return mask


def calculate_far_field_aperture(grid, delta, padding_factor=50):
    """ Calculates the aperture illumination pattern from the beam data.

    Args:
        grid (numpy.ndarray): gridded beam data
        delta (float): incremental spacing between lm values, ie. delta_l = l_(n+1) - l_(n)
        padding_factor (int, optional): Padding to apply to beam data grid before FFT. Padding is applied on outer edged of 
                                        each beam data grid and not between layers. Defaults to 20.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: aperture grid, u-coordinate array, v-coordinate array
    """
    
    logger.info("Calculating aperture illumination pattern ...")

    assert grid.shape[-1] == grid.shape[-2]  ###To do: why is this expected that l.shape == m.shape
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

    import scipy.fftpack
    shifted = scipy.fftpack.ifftshift(padded_grid)

    grid_fft = scipy.fftpack.fft2(shifted)

    aperture_grid = scipy.fftpack.fftshift(grid_fft)

    u_size = aperture_grid.shape[-2]
    v_size = aperture_grid.shape[-1]

    image_size = np.array([u_size, v_size])

    cell_size = 1 / (image_size * delta)

    u, v = calc_coords(image_size, cell_size)

    return aperture_grid, u, v, cell_size


def calculate_near_field_aperture(grid, sky_cell_size, distance, wavelength, padding_factor=50):
    """" Calculates the aperture illumination pattern from the near_fiedl beam data.

    Args:
        grid (numpy.ndarray): gridded beam data
        sky_cell_size (float): incremental spacing between lm values, ie. delta_l = l_(n+1) - l_(n)
        padding_factor (int, optional): Padding to apply to beam data grid before FFT. Padding is applied on outer edged of
                                        each beam data grid and not between layers. Defaults to 20.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: aperture grid, u-coordinate array, v-coordinate array
    """
    #
    apodizer = apodize_beam(grid[0, 0, 0, ...])
    apodized_grid = grid.copy()
    apodized_grid[0, 0, 0, ...] *= apodizer

    padded_grid = pad_beam_image(apodized_grid, padding_factor)
    uaxis, vaxis, laxis, maxis, aperture_cell_size = compute_axes(padded_grid.shape, sky_cell_size)

    fresnel = True

    aperture_grid = compute_aperture_fft(padded_grid)
    ref = aperture_grid[0, 0, 0, ...].copy()
    if fresnel:
        aperture_grid = compute_fresnel_corrections(padded_grid, aperture_grid, laxis, maxis, uaxis, vaxis, wavelength,
                                                    distance)
        # diff = np.sqrt((aperture_grid[0, 0, 0, ...]-ref)**2)/ref*100
        # print('RMS = ', np.mean(diff))
        # print('RMS STD', np.std(diff))
    else:
        pass

    return aperture_grid, uaxis, vaxis, aperture_cell_size


def calculate_parallactic_angle_chunk(
        time_samples,
        observing_location,
        direction,
        dir_frame="FK5",
        zenith_frame="FK5",
):
    """
    Converts a direction and zenith (frame FK5) to a topocentric Altitude-Azimuth (https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html)
    frame centered at the observing_location (frame ITRF) for a UTC time. The parallactic angles is calculated as the position angle of the Altitude-Azimuth
    direction and zenith.

    Parameters
    ----------
    time_samples: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        UTC time series. Example '2019-10-03T19:00:00.000'.
    observing_location: int np.array, [3], [x,y,z], meters
        ITRF geocentric coordinates.
    direction: float np.array, [n_time,2], [time,ra,dec], radians
        The pointing direction.
    Returns
    -------
    parallactic_angles: float np.array, [n_time], radians
        An array of parallactic angles.
    """

    observing_location = coord.EarthLocation.from_geocentric(
        x=observing_location[0] * u.m,
        y=observing_location[1] * u.m,
        z=observing_location[2] * u.m,
    )

    direction = coord.SkyCoord(
        ra=direction[:, 0] * u.rad, dec=direction[:, 1] * u.rad, frame=dir_frame.lower()
    )
    zenith = coord.SkyCoord(0, 90, unit=u.deg, frame=zenith_frame.lower())

    altaz_frame = coord.AltAz(location=observing_location, obstime=time_samples)
    zenith_altaz = zenith.transform_to(altaz_frame)
    direction_altaz = direction.transform_to(altaz_frame)

    return direction_altaz.position_angle(zenith_altaz).value


@njit(cache=False, nogil=True)
def gaussian_kernel(padded, original):
    mx = padded.shape[-2]//2
    my = padded.shape[-1]//2
    dx = original.shape[-2]//2
    dy = original.shape[-1]//2

    kernel = np.empty_like(padded)
    for it in range(padded.shape[0]):
        for ic in range(padded.shape[1]):
            for ip in range(padded.shape[2]):
                for ix in range(padded.shape[3]):
                    for iy in range(padded.shape[4]):
                        expo = (ix-mx)**2/(2*dx**2) + (iy-my)**2/(2*dy**2)
                        kernel[it, ic, ip, ix, iy] = np.exp(-expo)
                        #kernel[it, ic, ip, ix, iy] = 1.
    return kernel


def gaussian_2d(axes, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xaxis, yaxis = axes
    xo = float(xo)
    yo = float(yo)
    acoeff = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    bcoeff = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    ccoeff = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    expo = acoeff*((xaxis-xo)**2) + 2*bcoeff*(xaxis-xo)*(yaxis-yo) + ccoeff*((yaxis-yo)**2)
    gaussian = offset + amplitude*np.exp(-expo)
    return gaussian.ravel()


def fit_2d_gaussian(ref):

    # use your favorite image processing library to load an image
    nx, ny = ref.shape
    data = ref.ravel()

    xaxis = np.arange(nx)
    yaxis = np.arange(ny)
    xaxis, yaxis = np.meshgrid(xaxis, yaxis)

    # initial guess of parameters
    initial_guess = (1, nx//2, ny//2, nx, ny, 0, 0)

    import scipy.optimize as opt
    # find the optimal Gaussian parameters
    popt, pcov = opt.curve_fit(gaussian_2d, (xaxis, yaxis), data, p0=initial_guess, maxfev=int(1e6))

    # create new data with these parameters
    data_fitted = gaussian_2d((xaxis, yaxis), *popt)
    ref_fit = data_fitted.reshape(nx, ny)

    return ref_fit


def pad_beam_image(grid, padding_factor):
    assert grid.shape[-1] == grid.shape[-2]  ###To do: why is this expected that l.shape == m.shape
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
    return padded_grid


@njit(cache=False, nogil=True)
def apodize_beam(unpadded_beam, degree=2):
    nx, ny = unpadded_beam.shape
    apodizer = np.zeros(unpadded_beam.shape)
    for ix in range(nx):
        xfac = 4*(ix-nx-1)*(ix-1)/(nx**degree)
        for iy in range(ny):
            yfac = 4*(iy-ny-1)*(iy-1)/(ny**degree)
            # if np.sqrt((ix-nx//2)**2 + (iy-ny//2)**2) > nx//2:
            #     apodizer[ix, iy] = 0.0
            # else:
            apodizer[ix, iy] = xfac*yfac
    return apodizer


@njit(cache=False, nogil=True)
def correct_phase_nf_effects(aperture, uaxis, vaxis, distance, focus_offset, focal_length, wavelength):
    wave_vector = 0. + 2*np.pi*1j/wavelength
    for iu, uval in enumerate(uaxis):
        uval *= wavelength
        for iv, vval in enumerate(vaxis):
            vval *= wavelength
            axis_dist2 = uval**2 + vval**2
            z_term = axis_dist2/4/focal_length

            path_variation = (axis_dist2/2/distance - axis_dist2**2/8/distance**3 +
                              np.sqrt(axis_dist2 + (focal_length + focus_offset - z_term)**2) -
                              (focal_length + z_term + focus_offset))
            aperture[0, 0, 0, iu, iv] *= np.exp(wave_vector * path_variation)
    return aperture


def compute_axes(shape, sky_cell_size):
    u_size = shape[-2]
    v_size = shape[-1]
    image_size = np.array([u_size, v_size])
    aperture_cell_size = 1 / (image_size * sky_cell_size)
    uaxis, vaxis = calc_coords(image_size, aperture_cell_size)
    laxis, maxis = calc_coords(image_size, sky_cell_size)
    return uaxis, vaxis, laxis, maxis, aperture_cell_size


def compute_aperture_fft(padded_grid):
    shifted = scipy.fftpack.ifftshift(padded_grid)
    grid_fft = scipy.fftpack.fft2(shifted)
    aperture_grid = scipy.fftpack.fftshift(grid_fft)
    return aperture_grid


def compute_fresnel_corrections(padded_grid, aperture_grid, laxis, maxis, uaxis, vaxis, wavelength, distance, max_it=6):
    logger.info('Applying fresnel corrections...')
    wave_vector = 0. + 2*np.pi*1j/wavelength
    lmesh, mmesh = np.meshgrid(laxis, maxis)
    umesh, vmesh = np.meshgrid(uaxis, vaxis)
    umesh *= wavelength
    vmesh *= wavelength
    u2mesh = np.power(umesh, 2)
    v2mesh = np.power(vmesh, 2)

    it = 1
    while it < max_it:
        fft_work_array = padded_grid[0, 0, 0, ...].copy()
        if it == 1:
            fft_work_array *= lmesh
        elif it == 2:
            fft_work_array *= mmesh
        elif it == 3:
            fft_work_array *= np.power(lmesh, 2)
        elif it == 4:
            fft_work_array *= np.power(mmesh, 2)
        elif it == 5:
            fft_work_array *= lmesh * mmesh

        fresnel_corr = compute_aperture_fft(fft_work_array)

        if it == 1:
            aperture_grid += fresnel_corr * umesh*(u2mesh + v2mesh) / 2 / distance**2 * wave_vector
        elif it == 2:
            aperture_grid += fresnel_corr * vmesh*(u2mesh + v2mesh) / 2 / distance**2 * wave_vector
        elif it == 3:
            aperture_grid += fresnel_corr * u2mesh / 2 / distance * wave_vector
        elif it == 4:
            aperture_grid += fresnel_corr * v2mesh / 2 / distance * wave_vector
        elif it == 5:
            aperture_grid += fresnel_corr * umesh * vmesh / distance * wave_vector
        it += 1

    return aperture_grid
