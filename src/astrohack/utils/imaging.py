import math
import scipy.ndimage
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from numba import njit
import scipy.fftpack
import time

import toolviper.utils.logger as logger

from skimage.draw import disk
from astrohack.utils.algorithms import calc_coords, least_squares
from astrohack.utils.gridding import gridding_correction
from astrohack.utils.constants import clight, sig_2_fwhm


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
    dir_frame: frame of refecerence of the direction
    zenith_frame: Frame of reference of Zenith's coordinates?
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


def calculate_far_field_aperture(grid, padding_factor, freq, telescope, sky_cell_size, apply_grid_correction, label):
    """ Calculates the aperture illumination pattern from the beam data.

    Args:
        grid (numpy.ndarray): gridded beam data
        padding_factor (int, optional): Padding to apply to beam data grid before FFT. Padding is applied on outer edged of
                                        each beam data grid and not between layers. Defaults to 20.
        freq: Beam grid frequency axis
        telescope: telescope object with optical parameters
        sky_cell_size: Sky cell size (radians)
        apply_grid_correction: Apply grid correction (True for gaussian convolution of the beam)

    Returns:
        aperture grid, u-coordinate array, v-coordinate array, aperture cell size, representative wavelength
    """
    start = time.time()
    logger.debug(f"{label}: Calculating far field aperture illumination pattern ...")
    padded_grid = _pad_beam_image(grid, padding_factor)

    aperture_grid = _compute_aperture_fft(padded_grid)

    wavelength = clight/freq[0]
    u_axis, v_axis, _, _, aperture_cell_size = _compute_axes(padded_grid.shape, sky_cell_size, wavelength)

    if apply_grid_correction:
        aperture_grid = gridding_correction(aperture_grid, freq, telescope.diam, sky_cell_size, u_axis, v_axis)
    duration = time.time() - start
    logger.debug(f'{label}: Far field aperture took {duration:.3} seconds')
    return aperture_grid, u_axis, v_axis, aperture_cell_size, wavelength


def calculate_near_field_aperture(grid, sky_cell_size, distance, freq, padding_factor, focus_offset, telescope,
                                  apply_grid_correction, label, apodize=True):
    """" Calculates the aperture illumination pattern from the near_fiedl beam data.

    Args:
        grid (numpy.ndarray): gridded beam data
        sky_cell_size (float): incremental spacing between lm values, ie. delta_l = l_(n+1) - l_(n)
        padding_factor (int, optional): Padding to apply to beam data grid before FFT. Padding is applied on outer edges
                                        of each beam data grid and not between layers. Defaults to 20.
        distance: distance to holographic tower
        focus_offset: Offset from primary focus on holographic receiver
        freq: Beam grid frequency axis
        telescope: telescope object with optical parameters
        apply_grid_correction: Apply grid correction (True for gaussian convolution of the beam)
        apodize: Apodize beam to avoid boxing effects in the FFT (the dashed line cross)

    Returns:
        aperture grid, u-coordinate array, v-coordinate array, aperture cell size, representative wavelength
    """
    logger.debug(f"{label}: Calculating near field aperture illumination pattern ...")
    start = time.time()
    work_grid = grid.copy()

    if apodize:
        apodizer = _apodize_beam(work_grid[0, 0, 0, ...])
        work_grid[0, 0, 0, ...] *= apodizer

    padded_grid = _pad_beam_image(work_grid, padding_factor)
    wavelength = clight / freq[0]
    z_max = (telescope.diam/2)**2/4/telescope.focus
    scale = 1. + (telescope.el_axis_off+z_max/2.)/distance
    u_axis, v_axis, l_axis, m_axis, aperture_cell_size = _compute_axes(padded_grid.shape, sky_cell_size, wavelength,
                                                                       scale=scale)
    aperture_grid = _compute_aperture_fft(padded_grid)

    factor = 2j*np.pi/wavelength
    aperture_grid = _compute_non_fresnel_corrections(padded_grid, aperture_grid, l_axis, m_axis, u_axis, v_axis,
                                                     factor, distance)
    if apply_grid_correction:
        aperture_grid = gridding_correction(aperture_grid, freq, telescope.diam, sky_cell_size, u_axis, v_axis)

    aperture_grid = _correct_phase_nf_effects(aperture_grid, u_axis, v_axis, distance, focus_offset, telescope.focus,
                                              factor)

    #
    phase = np.angle(aperture_grid[0, 0, 0, ...])
    amp = np.absolute(aperture_grid[0, 0, 0, ...])
    # dishhorn_artefact = fit_dishhorn_beam_artefact(amp, telescope.inlim, u_axis, v_axis)
    # amp -= dishhorn_artefact

    phase = _feed_correction(phase, u_axis, v_axis, telescope.focus)
    # fitted_amp = fit_illumination_pattern(amp, u_axis, v_axis, telescope.diam, blockage)
    # aperture_grid[0, 0, 0, ...] = fitted_amp * (np.cos(phase) + 1j * np.sin(phase))
    aperture_grid[0, 0, 0, ...] = amp * (np.cos(phase) + 1j * np.sin(phase))
    duration = time.time() - start
    logger.debug(f'{label}: Near field aperture took {duration:.3} seconds')
    return aperture_grid, u_axis, v_axis, aperture_cell_size, wavelength


def _feed_correction(phase, u_axis, v_axis, focal_length, nk=10):
    """
    Correction to the phases due to the phase change created by the antenna feed
    Args:
        phase: Aperture phase
        u_axis: U axis
        v_axis: V Axis
        focal_length: Telescope focal length
        nk: number of terms to include

    Returns:

    """
    # Tabulated Sigma GE and GH functions:
    gh_tab = [13.33004 - 0.03155j, -1.27077 + 0.00656j, 0.38349 - 0.17755j, 0.78041 - 0.11238j, -0.54821 + 0.16739j,
              -0.68021 + 0.11472j, 1.05341 - 0.01921j, -0.80119 + 0.06443j, 0.36258 - 0.01845j, -0.07905 + 0.00515j]
    ge_tab = [12.79400 + 2.27305j, 1.06279 - 0.56235j, -1.92694 - 1.72309j, 1.79152 - 0.08008j,  0.09406 + 0.46197j,
              -2.82441 - 1.06010j, 2.77077 + 1.02349j, -1.74437 - 0.45956j, 0.62276 + 0.11504j, -0.11176 + 0.01616j]

    umesh, vmesh = np.meshgrid(u_axis, v_axis)
    radius2 = umesh**2 + vmesh**2
    theta2 = radius2 / 4 / focal_length**2
    theta = np.sqrt(theta2)
    z_term = (1-theta2+2j*theta)/(1+theta2)

    zz_term = np.full_like(phase, 1+0j, dtype=complex)
    geth = np.zeros_like(phase, dtype=complex)
    ghth = np.zeros_like(phase, dtype=complex)

    for k in range(nk):
        zz_term *= z_term
        costh = zz_term.real
        geth += ge_tab[k] * costh
        ghth += gh_tab[k] * costh
    phi = np.arctan2(vmesh, umesh)
    gain = geth * np.sin(phi)**2 + ghth * np.cos(phi)**2
    feed_phase = np.angle(gain)

    return phase + feed_phase


def _fit_illumination_pattern(amp, u_axis, v_axis, diameter, blockage):
    """
    Fit aperture amplitude pattern to a gaussian
    Args:
        amp: Aperture amplitude
        u_axis: U axis
        v_axis: V axis
        diameter: Telescope diameter
        blockage: Inner blockage radius

    Returns:
        Best fit gaussian to the ilumination pattern
    """
    amp_max = np.max(amp)
    db_amp = 10.*np.log10(amp/amp_max)

    npar = 5
    umesh, vmesh = np.meshgrid(u_axis, v_axis)
    umesh = umesh.ravel()
    vmesh = vmesh.ravel()
    umesh2 = umesh**2
    vmesh2 = vmesh**2

    oulim2 = (diameter/2)**2
    inlim2 = blockage**2
    dist2 = umesh2 + vmesh2

    mask = np.where(dist2 >= inlim2, True, False)
    mask = np.where(dist2 >= oulim2, False, mask)

    npoints = np.sum(mask)
    matrix = np.empty([npoints, npar])
    matrix[:, 0] = umesh2[mask]
    matrix[:, 1] = vmesh2[mask]
    matrix[:, 2] = umesh[mask]
    matrix[:, 3] = vmesh[mask]
    matrix[:, 4] = 1.0
    vector = db_amp.ravel()[mask]

    result, _, _ = least_squares(matrix, vector)
    db_fitted = umesh2*result[0] + vmesh2*result[1] + umesh*result[2] + vmesh*result[3] + result[4]
    fitted = 10**(db_fitted/10)
    return fitted.reshape(amp.shape)


def _circular_gaussian(axes, amp, x0, y0, sigma, offset):
    """
    Compute a Circular gaussian image
    Args:
        axes: X and Y axes mesh grids
        amp: Gaussian amplitude
        x0: Gaussian center in X
        y0: Gaussian center in Y
        sigma: Gaussian width
        offset: gaussian base offset

    Returns:
        circular gaussian image
    """
    x_axis, y_axis = axes
    x0 = float(x0)
    y0 = float(y0)
    expo = 1*((x_axis-x0) ** 2 + (y_axis - y0) ** 2)
    expo /= 2*sigma**2
    return amp*np.exp(-expo)+offset


def _two_gaussians(axes, x0, y0, amp_narrow, sigma_narrow, amp_broad, sigma_broad, offset):
    """
    Compute the sum of two Circular gaussian images, one narrow the othe broad
    Args:
        axes: X and Y axes mesh grids
        x0: Gaussian center in X
        y0: Gaussian center in Y
        amp_narrow: Narrow Gaussian amplitude
        sigma_narrow: Narrow Gaussian width
        amp_broad: broad Gaussian amplitude
        sigma_broad: broad Gaussian width
        offset: gaussian base offset

    Returns:
        sum of two circular gaussian images
    """
    #offset = 0
    narrow = _circular_gaussian(axes, amp_narrow, x0, y0, sigma_narrow, 0)
    broad = _circular_gaussian(axes, amp_broad, x0, y0, sigma_broad, 0)
    return narrow+broad+offset


def _fit_dishhorn_beam_artefact(amp, blockage, u_axis, v_axis):
    """
    Attempt to fit the artefact that appears at the center of an aperture amplitude iamge
    Args:
        amp: Aperture amplitude
        blockage: Inner blockage radius
        u_axis: U axis
        v_axis: V axis

    Returns:
        Fitted artefact
    """
    logger.info('Fitting feed horn artefact')
    u_mesh, v_mesh = np.meshgrid(u_axis, v_axis)

    dist2 = u_mesh**2+v_mesh**2
    sel = dist2 < (4*blockage) ** 2

    # Ravel data for the fit
    fit_data = amp[sel]
    fit_u = u_mesh[sel]
    fit_v = v_mesh[sel]

    # initial guess of parameters
    initial_guess = (0, 0, np.max(amp), blockage / sig_2_fwhm, 4.0, 2*blockage/sig_2_fwhm, 0)
    import scipy.optimize as opt
    # find the optimal Gaussian parameters
    results = opt.curve_fit(_two_gaussians, (fit_u, fit_v), fit_data, p0=initial_guess, maxfev=int(1e6))

    popt = results[0]
    # create new data with these parameters
    popt[-1] = 0
    data_fitted = _two_gaussians((u_mesh, v_mesh), *popt)

    popt[3] *= sig_2_fwhm
    popt[5] *= sig_2_fwhm
    feed_fit = data_fitted
    return feed_fit


def _pad_beam_image(grid, padding_factor):
    """
    Pad beam image with zeros to avoid aliasing in FFTs
    Args:
        grid: beam grid
        padding_factor: padding factor to determine padded size

    Returns:
        Zero padded beam grid
    """
    assert grid.shape[-1] == grid.shape[-2]  ###To do: why is this expected that l.shape == m.shape
    initial_dimension = grid.shape[-1]

    # Calculate padding as the nearest power of 2
    # k log (2) = log(N) => k = log(N)/log(2)
    # New shape => K = math.ceil(k) => shape = (K, K)
    k_coeff = np.log(initial_dimension * padding_factor) / np.log(2)
    k_integer = math.ceil(k_coeff)
    padded_size = np.power(2, k_integer)
    padding = (padded_size - initial_dimension) // 2
    z_pad = [0, 0]
    if initial_dimension == initial_dimension//2*2:
        pad_wid = [padding, padding]
    else:
        pad_wid = [padding+1, padding]

    pad_width = np.array([z_pad, z_pad, z_pad, pad_wid, pad_wid])
    padded_grid = np.pad(array=grid,   pad_width=pad_width, mode='constant')
    return padded_grid


@njit(cache=False, nogil=True)
def _apodize_beam(unpadded_beam, degree=2):
    """
    Apodize beam image to avoid artefacts in aperture image
    Args:
        unpadded_beam: Unpadded beam image
        degree: Degree of apodization

    Returns:
        Apodizing image
    """
    nx, ny = unpadded_beam.shape
    apodizer = np.zeros(unpadded_beam.shape)
    for ix in range(nx):
        xfac = 4*(ix-nx-1)*(ix-1)/(nx**degree)
        for iy in range(ny):
            yfac = 4*(iy-ny-1)*(iy-1)/(ny**degree)
            apodizer[ix, iy] = xfac*yfac
    return apodizer


def _correct_phase_nf_effects(aperture, u_axis, v_axis, distance, focus_offset, focal_length, factor):
    """
    DOES NOT WORK AS INTENDED
    Compute the near field corrections to be applied to the phases
    Args:
        aperture: Aperture Grid image
        u_axis: U axis
        v_axis: V axis
        distance: Distance to holographic tower
        focus_offset: Defocus used to mitigate NF effects
        focal_length: Telescope focal length
        factor: wave number scaling

    Returns:
        Aperture with phase corrected for the NF effects
    """
    umesh, vmesh = np.meshgrid(u_axis, v_axis)
    wave_vector = factor
    axis_dist2 = umesh**2+vmesh**2  # \epsilon^2 + \eta^2
    z_term = axis_dist2/4/focal_length  # \frac{\epsilon^2 + \eta^2}{4f}
    first_term = axis_dist2/2/distance  # \frac{\epsilon^2 + \eta^2}{2R}
    second_term = axis_dist2**2/8/distance**3  # \frac{(\epsilon^2 + \eta^2)^2}{8R^3}
    dp1 = first_term - second_term
    # \sqrt{(\epsilon^2 + \eta^2) + (f - \frac{\epsilon^2 + \eta^2}{4f} +df)^2}
    first_term = np.sqrt(axis_dist2 + (focal_length + focus_offset - z_term)**2)
    # f - \frac{\epsilon^2 + \eta^2}{4f} +df
    second_term = focal_length + z_term + focus_offset
    dp2 = first_term-second_term
    path_var = dp1+dp2
    aperture[0, 0, 0, ...] *= np.exp(wave_vector * path_var)
    return aperture


def _compute_axes(shape, sky_cell_size, wavelength, scale=1.0):
    """
    Compute the axes of the padded beam image and also the aperture axes
    Args:
        shape: Padded beam image shape
        sky_cell_size: Sky cell size (radians)
        wavelength: representative wavelenght
        scale: axis scaling

    Returns:
        U, V, L and M axis and aperture cell size
    """
    image_size = np.array([shape[-2], shape[-1]])
    aperture_cell_size = wavelength / (image_size * scale * sky_cell_size)
    u_axis, v_axis = calc_coords(image_size, aperture_cell_size)
    l_axis, m_axis = calc_coords(image_size, scale * sky_cell_size)
    return u_axis, v_axis, l_axis, m_axis, aperture_cell_size


def _compute_aperture_fft(padded_grid):
    """
    Compute aperture FFT
    Args:
        padded_grid: Zero padded beam grid

    Returns:
        Aperture
    """
    shifted = scipy.fftpack.ifftshift(padded_grid)
    grid_fft = scipy.fftpack.fft2(shifted)
    aperture_grid = scipy.fftpack.fftshift(grid_fft)
    return aperture_grid


def _compute_non_fresnel_corrections(padded_grid, aperture_grid, l_axis, m_axis, u_axis, v_axis, factor, distance,
                                     max_it=6, verbose=True):
    """
    DOES NOT WORK AS INTENDED
    Compute non fresnel corrections to the aperture
    Args:
        padded_grid: Zero padded beam grid
        aperture_grid: The FFT of the padde beam grid
        l_axis: Beam's L axis
        m_axis: Beam's M axis
        u_axis: Aperture's U axis
        v_axis: Aperture's V axis
        factor: Wave number
        distance: Distance to the hologrphy tower
        max_it: Number of iterations
        verbose: Print messages?
    Returns:
        Aperture with non fresnel corrections
    """
    if verbose:
        logger.info('Applying non-fresnel corrections...')
    wave_vector = factor
    lmesh, mmesh = np.meshgrid(l_axis, m_axis)
    umesh, vmesh = np.meshgrid(u_axis, v_axis)
    u2mesh = np.power(umesh, 2)
    v2mesh = np.power(vmesh, 2)
    dist2 = u2mesh+v2mesh

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

        fft_term = _compute_aperture_fft(fft_work_array)

        if it == 1:
            corr_term = umesh * dist2 / 2 / distance**2 * wave_vector
        elif it == 2:
            corr_term = vmesh * dist2 / 2 / distance**2 * wave_vector
        elif it == 3:
            corr_term = -1 * u2mesh / 2 / distance * wave_vector
        elif it == 4:
            corr_term = -1 * v2mesh / 2 / distance * wave_vector
        elif it == 5:
            corr_term = -1 * umesh * vmesh / distance * wave_vector
        add_term = corr_term * fft_term
        aperture_grid += add_term

        it += 1
    return aperture_grid
