import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from numba import njit
import scipy.fftpack

import graphviper.utils.logger as logger

from skimage.draw import disk
from astrohack.utils.algorithms import calc_coords, least_squares
from astrohack.utils.gridding import gridding_correction
from astrohack.utils.constants import clight, sig_2_fwhm
from astrohack.visualization.plot_tools import create_figure_and_axes, well_positioned_colorbar, get_proper_color_map


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


def calculate_far_field_aperture(grid, padding_factor, freq, telescope, sky_cell_size, apply_grid_correction):
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
    padded_grid = pad_beam_image(grid, padding_factor)

    aperture_grid = compute_aperture_fft(padded_grid)

    wavelength = clight/freq[0]
    u_axis, v_axis, _, _, aperture_cell_size = compute_axes(padded_grid.shape, sky_cell_size, wavelength)

    if apply_grid_correction:
        aperture_grid = gridding_correction(aperture_grid, freq, telescope.diam, sky_cell_size, u_axis, v_axis)

    return aperture_grid, u_axis, v_axis, aperture_cell_size, wavelength


def calculate_near_field_aperture(grid, sky_cell_size, distance, freq, padding_factor, focus_offset, telescope,
                                  apply_grid_correction, apodize=True):
    """" Calculates the aperture illumination pattern from the near_fiedl beam data.

    Args:
        grid (numpy.ndarray): gridded beam data
        sky_cell_size (float): incremental spacing between lm values, ie. delta_l = l_(n+1) - l_(n)
        padding_factor (int, optional): Padding to apply to beam data grid before FFT. Padding is applied on outer edged of
                                        each beam data grid and not between layers. Defaults to 20.
        distance: distance to holographic tower
        wavelength: holography wavelength
        focus_offset: Offset from primary focus on holographic receiver
        focal_length: Antenna focal length
        apodize: Apodize beam to avoid boxing effects in the FFT (the dashed line cross)

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: aperture grid, u-coordinate array, v-coordinate array
    """
    work_grid = grid.copy()

    if apodize:
        apodizer = apodize_beam(work_grid[0, 0, 0, ...])
        work_grid[0, 0, 0, ...] *= apodizer

    padded_grid = pad_beam_image(work_grid, padding_factor)
    wavelength = clight / freq[0]

    u_axis, v_axis, l_axis, m_axis, aperture_cell_size = compute_axes(padded_grid.shape, sky_cell_size, wavelength)
    aperture_grid = compute_aperture_fft(padded_grid)

    # if distance is None:
    #     logger.info('Fitting distance is long and you should feel bad =0')
    #     result = fit_holo_tower_distance(padded_grid, aperture_grid, l_axis, m_axis, u_axis, v_axis, wavelength,
    #                                      focus_offset, focal_length, telescope.diam)
    #
    # else:
    wvl = wavelength
    print(wvl, 2*np.pi)
    factor = 2j*np.pi/wvl
    print(factor, factor*wvl)
    focus_offset *= 1
    aperture_grid = compute_non_fresnel_corrections(padded_grid, aperture_grid, l_axis, m_axis, u_axis, v_axis,
                                                    factor, distance)
    if apply_grid_correction:
        aperture_grid = gridding_correction(aperture_grid, freq, telescope.diam, sky_cell_size, u_axis, v_axis)

    aperture_grid = correct_phase_nf_effects(aperture_grid, u_axis, v_axis, distance, focus_offset, telescope.focus,
                                             factor)

    #
    phase = np.angle(aperture_grid[0, 0, 0, ...])
    amp = np.absolute(aperture_grid[0, 0, 0, ...])
    # dishhorn_artefact = fit_dishhorn_beam_artefact(amp, telescope.inlim, u_axis, v_axis, telescope.diam)
    # amp -= dishhorn_artefact

    phase = feed_correction(phase, u_axis, v_axis, telescope.focus)
    # fitted_amp = fit_illumination_pattern(amp, u_axis, v_axis, telescope.diam, blockage)
    # aperture_grid[0, 0, 0, ...] = fitted_amp * (np.cos(phase) + 1j * np.sin(phase))
    aperture_grid[0, 0, 0, ...] = amp * (np.cos(phase) + 1j * np.sin(phase))

    return aperture_grid, u_axis, v_axis, aperture_cell_size, distance, wavelength


def feed_correction(phase, u_axis, v_axis, focal_length, nk=10):
    # Tabulated Sigma GE and GH functions:
    gh_tab = [13.33004 - 0.03155j, -1.27077 + 0.00656j, 0.38349 - 0.17755j, 0.78041 - 0.11238j, -0.54821 + 0.16739j,
              -0.68021 + 0.11472j, 1.05341 - 0.01921j, -0.80119 + 0.06443j, 0.36258 - 0.01845j, -0.07905 + 0.00515j]
    ge_tab = [12.79400 + 2.27305j, 1.06279 - 0.56235j, -1.92694 - 1.72309j, 1.79152 - 0.08008j,  0.09406 + 0.46197j,
              -2.82441 - 1.06010j, 2.77077 + 1.02349j, -1.74437 - 0.45956j, 0.62276 + 0.11504j, -0.11176 + 0.01616j]

    umesh, vmesh = np.meshgrid(u_axis, v_axis)
    radius2 = umesh**2 + vmesh**2
    theta2 = radius2 / 4 / focal_length**2
    theta = np.sqrt(theta2)
    #z_term = np.cos(theta) + np.sin(theta)*1j
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
    # fig, axes = create_figure_and_axes(None, [1, 1])
    # plot_map_simple(feed_phase, fig, axes, 'feed_correction', u_axis, v_axis)
    # plot_map_simple(expo.imag, fig, axes[0, 1], 'imag', u_axis, v_axis)
    # plot_map_simple(np.angle(expo), fig, axes[1, 0], 'phase', u_axis, v_axis)
    # plot_map_simple(path_var, fig, axes[1, 1], 'path_var', u_axis, v_axis)
    # plt.show()

    return phase + feed_phase


def fit_illumination_pattern(amp, u_axis, v_axis, diameter, blockage):
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
    # mask = np.full_like(umesh, True, dtype=bool)

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


def eliptical_gaussian(axes, amplitude, x0, yo, sigma_x, sigma_y, theta, offset):
    x_axis, y_axis = axes
    x0 = float(x0)
    yo = float(yo)
    acoeff = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    bcoeff = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    ccoeff = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    expo = acoeff*((x_axis-x0)**2) + 2*bcoeff*(x_axis-x0)*(y_axis-yo) + ccoeff*((y_axis-yo)**2)
    gaussian = offset + amplitude*np.exp(-expo)
    return gaussian


def circular_gaussian(axes, amp, x0, yo, sigma, offset):
    x_axis, y_axis = axes
    x0 = float(x0)
    yo = float(yo)
    expo = 1*((x_axis-x0)**2 + (y_axis-yo)**2)
    expo /= 2*sigma**2
    return amp*np.exp(-expo)+offset


def two_gaussians(axes, x0, y0, amp_narrow, sigma_narrow, amp_broad, sigma_broad, offset):
    #offset = 0
    narrow = circular_gaussian(axes, amp_narrow, x0, y0, sigma_narrow, 0)
    broad = circular_gaussian(axes, amp_broad, x0, y0, sigma_broad, 0)
    return narrow+broad+offset


def fit_dishhorn_beam_artefact(amp, blockage, u_axis, v_axis, diameter):
    logger.info('Fitting feed horn artefact')
    nx, ny = amp.shape

    u_mesh, v_mesh = np.meshgrid(u_axis, v_axis)

    dist2 = u_mesh**2+v_mesh**2
    #sel = dist2 < (diameter/2)**2
    sel = dist2 < (4*blockage) ** 2

    # Ravel data for the fit
    fit_data = amp[sel]
    fit_u = u_mesh[sel]
    fit_v = v_mesh[sel]
    print(fit_data.shape, u_mesh.shape)

    # initial guess of parameters
    # initial_guess = (600, nx//2, ny//2, blockage/sig_2_fwhm, blockage/sig_2_fwhm, 0, 0)
    # initial_guess = (np.max(amp), 0, 0, blockage / sig_2_fwhm, 0)
    initial_guess = (0, 0, np.max(amp), blockage / sig_2_fwhm, 4.0, 2*blockage/sig_2_fwhm, 0)
    import scipy.optimize as opt
    # find the optimal Gaussian parameters
    # results = opt.curve_fit(gaussian_2d, (u_mesh, v_mesh), data, p0=initial_guess, maxfev=int(1e6))
    print(np.count_nonzero(np.isnan(fit_data)))
    # results = opt.curve_fit(circular_gaussian, (fit_u, fit_v), fit_data, p0=initial_guess, maxfev=int(1e6))
    results = opt.curve_fit(two_gaussians, (fit_u, fit_v), fit_data, p0=initial_guess, maxfev=int(1e6))

    popt = results[0]
    # create new data with these parameters
    # data_fitted = gaussian_2d((u_mesh, v_mesh), *popt)
    # data_fitted = circular_gaussian((u_mesh, v_mesh), *popt)
    popt[-1] = 0
    data_fitted = two_gaussians((u_mesh, v_mesh), *popt)


    popt[3] *= sig_2_fwhm
    popt[5] *= sig_2_fwhm

    print(popt)

    # data_fitted = two_gaussians((u_mesh, v_mesh), [0, 0, ])
    feed_fit = data_fitted

    print(amp.shape, sel.shape, feed_fit.shape, feed_fit[sel].shape)
    fig, axes = create_figure_and_axes(None, [2, 2])
    axes[0, 0].imshow(feed_fit)
    axes[1, 0].imshow(amp)
    axes[0, 1].plot(u_axis, amp[nx//2, :], color='red')
    axes[0, 1].plot(u_axis, feed_fit[nx//2, :], color='blue')
    axes[0, 1].axvline(x=diameter / 2, color='yellow')
    axes[0, 1].axvline(x=-diameter / 2, color='yellow')
    axes[0, 1].set_xlim([-1.5*diameter/2, 1.5*diameter/2])

    plt.show()

    return feed_fit


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
            apodizer[ix, iy] = xfac*yfac
    return apodizer


def correct_phase_nf_effects(aperture, u_axis, v_axis, distance, focus_offset, focal_length, factor):
    # zm = (diameter/2.)**2/4/focus
    # do i2 = 1, n2
    #   yy = (i2-ref2)*dy
    #   do i1 = 1, n1
    #     xx = (i1-ref1)*dy
    #     r2 = xx**2 + yy**2
    #     z = r2 /4 /focus
    #     dp = r2 /2/distance - r2**2 /8/distance**3   &
    #       +  sqrt(r2+(focus+dfocus-z)**2) - (focus+z +dfocus)
    #     ! do the correction
    #     y(i1,i2) = y(i1,i2) * exp(factor * dp)
    #   !            y(i1,i2) = exp(factor * dp)
    #   enddo
    # enddo
    print(focus_offset, focal_length, distance)
    umesh, vmesh = np.meshgrid(u_axis, v_axis)
    wave_vector = factor
    axis_dist2 = umesh**2+vmesh**2  # \epsilon^2 + \eta^2
    z_term = axis_dist2/4/focal_length  # \frac{\epsilon^2 + \eta^2}{4f}
    first_term = axis_dist2/2/distance  # \frac{\epsilon^2 + \eta^2}{2R}
    second_term = axis_dist2**2/8/distance**3  # \frac{(\epsilon^2 + \eta^2)^2}{8R^3}
    dp1 = first_term - second_term
    # \sqrt{(\epsilon^2 + \eta^2) + (f - \frac{\epsilon^2 + \eta^2}{4f} +df)^2}
    furst_term = np.sqrt(axis_dist2 + (focal_length + focus_offset - z_term)**2)
    # f - \frac{\epsilon^2 + \eta^2}{4f} +df
    sucond_term = focal_length + z_term + focus_offset
    dp2 = furst_term-sucond_term
    path_var = dp1+dp2
    expo = np.exp(wave_vector * path_var)

    # fig, axes = create_figure_and_axes(None, [2, 2])
    # plot_map_simple(expo.real, fig, axes[0, 0], 'real', u_axis, v_axis)
    # plot_map_simple(expo.imag, fig, axes[0, 1], 'imag', u_axis, v_axis)
    # plot_map_simple(np.angle(expo), fig, axes[1, 0], 'phase', u_axis, v_axis)
    # plot_map_simple(path_var, fig, axes[1, 1], 'path_var', u_axis, v_axis)
    # plt.show()
    aperture[0, 0, 0, ...] *= np.exp(wave_vector * path_var)
    return aperture


def plot_map_simple(data, fig, ax, title, u_axis, v_axis):
    extent = [np.min(u_axis), np.max(u_axis), np.min(v_axis), np.max(v_axis)]
    cmap = get_proper_color_map('viridis')
    im = ax.imshow(data, cmap=cmap, extent=extent)
    circ = Circle((0, 0), 6, fill=False, color='black')
    ax.add_patch(circ)
    circ = Circle((0, 0), 3, fill=False, color='black')
    ax.add_patch(circ)
    ax.set_title(title)
    well_positioned_colorbar(ax, fig, im, title)


def compute_axes(shape, sky_cell_size, wavelength):
    image_size = np.array([shape[-2], shape[-1]])
    aperture_cell_size = wavelength / (image_size * sky_cell_size)
    u_axis, v_axis = calc_coords(image_size, aperture_cell_size)
    l_axis, m_axis = calc_coords(image_size, sky_cell_size)
    return u_axis, v_axis, l_axis, m_axis, aperture_cell_size


def _get_img_size(shape):
    return np.array([shape[-2], shape[-1]])


def _compute_aperture_cell_size(wavelength, image_size, sky_cell_size):
    return wavelength / (image_size * sky_cell_size)


def compute_aperture_fft(padded_grid):
    shifted = scipy.fftpack.ifftshift(padded_grid)
    grid_fft = scipy.fftpack.fft2(shifted)
    aperture_grid = scipy.fftpack.fftshift(grid_fft)
    return aperture_grid


def compute_non_fresnel_corrections(padded_grid, aperture_grid, l_axis, m_axis, u_axis, v_axis, factor, distance,
                                    max_it=6, verbose=True):
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

        fft_term = compute_aperture_fft(fft_work_array)

        print(80*'#')
        print(it)
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
        print(np.min(fft_term), np.max(fft_term))
        print(np.min(corr_term), np.max(corr_term))
        add_term = corr_term * fft_term
        print(np.min(add_term), np.max(add_term))
        aperture_grid += add_term
        print(np.min(aperture_grid[0, 0, 0, ...]), np.max(aperture_grid[0, 0, 0, ...]))
        print(80*'#')
        # fig, axes = create_figure_and_axes(None, [2, 4])
        # plot_map_simple(fft_term.real, fig, axes[0, 0], 'real fft', u_axis, v_axis)
        # plot_map_simple(fft_term.imag, fig, axes[1, 0], 'imag fft', u_axis, v_axis)
        # plot_map_simple(corr_term.real, fig, axes[0, 1], 'real corr', u_axis, v_axis)
        # plot_map_simple(corr_term.imag, fig, axes[1, 1], 'imag corr', u_axis, v_axis)
        # plot_map_simple(add_term.real, fig, axes[0, 2], 'real add', u_axis, v_axis)
        # plot_map_simple(add_term.imag, fig, axes[1, 2], 'imag add', u_axis, v_axis)
        # plot_map_simple(aperture_grid[0, 0, 0, ...].real, fig, axes[0, 3], 'real aperture', u_axis, v_axis)
        # plot_map_simple(aperture_grid[0, 0, 0, ...].imag, fig, axes[1, 3], 'imag aperture', u_axis, v_axis)
        # fig.suptitle(f'iteration: {it}')
        # plt.show()


        it += 1
    #print(np.min(aperture_grid), np.max(aperture_grid))
    print(fft_term.shape, aperture_grid.shape, corr_term.shape)
    return aperture_grid


def fit_holo_tower_distance(padded_grid, aperture_grid, l_axis, m_axis, u_axis, v_axis, wavelength, focus_offset,
                            focal_length, diameter):
    from scipy.optimize import minimize
    fixed_par_dict = locals()
    initial_guess = 300  # Tower is ~300m from pads in OSF, gathered from Google Maps
    result = minimize(distance_fitting_function, initial_guess, args=fixed_par_dict)

    return result


def distance_fitting_function(distance, par_dict):
    aperture_grid = compute_non_fresnel_corrections(par_dict["padded_grid"], par_dict["aperture_grid"],
                                                    par_dict["l_axis"], par_dict["m_axis"], par_dict["u_axis"],
                                                    par_dict["v_axis"], par_dict["wavelength"], distance)
    aperture_grid = correct_phase_nf_effects(aperture_grid, par_dict["u_axis"], par_dict["v_axis"],
                                             distance, par_dict["focus_offset"], par_dict["focal_length"],
                                             par_dict["wavelength"])
    rms = compute_phase_rms(aperture_grid, par_dict["diameter"], par_dict["u_axis"], par_dict["v_axis"])
    return rms


def compute_phase_rms(aperture, diameter, u_axis, v_axis):
    phase = np.angle(aperture[0, 0, 0, ...], deg=False)
    umesh, vmesh = np.meshgrid(u_axis, v_axis)
    aper_radius = np.sqrt(umesh**2+vmesh**2)
    ant_radius = diameter/2.
    mask = np.where(aper_radius > ant_radius, False, True)
    rms = np.sqrt(np.mean(phase[mask] ** 2))
    return rms
