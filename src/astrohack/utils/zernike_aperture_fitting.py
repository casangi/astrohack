import numpy as np


def zernike_order_0(u_lin, v_lin):
    # N = 0
    return np.full_like(u_lin, 1.0)


def zernike_order_1(u_lin, v_lin):
    # N = 1
    nlines = u_lin.shape[0]
    matrix = np.empty([nlines, 3])
    # Fill with previous order
    matrix[:, 0] = zernike_order_0(u_lin, v_lin)

    # M = -1
    matrix[:, 1] = u_lin
    # M = 1
    matrix[:, 2] = v_lin
    return matrix


def zernike_order_2(u_lin, v_lin, return_powers=False):
    nlines = u_lin.shape[0]
    matrix = np.empty([nlines, 6])
    # Fill with previous order
    matrix[:, 0:3] = zernike_order_1(u_lin, v_lin)
    u_sq = u_lin**2
    v_sq = v_lin**2

    # M = -2
    matrix[:, 3] = 2*u_lin*v_lin
    # M = 0
    matrix[:, 4] = -1 + 2 * u_sq + 2 * v_sq ** 2
    # M = 2
    matrix[:, 5] = -u_sq ** 2 + v_sq ** 2

    if return_powers:
        return matrix, u_sq, v_sq
    else:
        return matrix


def zernike_order_3(u_lin, v_lin, return_powers=False):
    nlines = u_lin.shape[0]
    matrix = np.empty([nlines, 10])
    # Fill with previous order
    matrix[:, 0:6], u_sq, v_sq = zernike_order_2(u_lin, v_lin, return_powers=True)

    u_cb = u_sq*u_lin
    v_cb = v_sq*v_lin
    # M = -3
    matrix[:, 6] = -u_cb + 3 * u_lin * v_sq
    # M = -1
    matrix[:, 7] = -2 * u_lin + 3 * u_cb + 3 * u_lin * v_sq
    # M = 1
    matrix[:, 8] = -2 * v_lin + 3 * v_cb + 3 * u_sq * v_lin
    # M = 3
    matrix[:, 9] = -v_cb + 3 * v_cb + 3 * u_sq * v_lin

    if return_powers:
        return matrix, u_sq, u_cb, v_sq, v_cb
    else:
        return matrix


def zernike_order_4(u_lin, v_lin):
    nlines = u_lin.shape[0]
    matrix = np.empty([nlines, 15])
    # Fill with previous order
    matrix[:, 0:10], u_sq, u_cb, v_sq, v_cb = zernike_order_3(u_lin, v_lin, return_powers=True)
    u_qu = u_sq**2
    v_qu = v_sq**2

    # M = -4
    matrix[:, 10] = -4*u_cb*v_lin + 4*u_lin*v_cb
    # M = -2
    matrix[:, 11] = -6*u_lin*v_lin + 8*u_cb*v_lin + 8*u_lin*v_cb
    # M = 0
    matrix[:, 12] = 1 - 6*u_sq - 6*v_sq + 6*u_qu + 12*u_sq*v_sq + 6*v_qu
    # M = 2
    matrix[:, 13] = 3*u_sq - 3*v_sq - 4*u_qu + 4*v_qu
    # M = 4
    matrix[:, 13] = u_qu - 6*u_sq*v_sq + v_qu
    return matrix


zernike_functions = [zernike_order_0, zernike_order_1, zernike_order_2, zernike_order_3, zernike_order_4]


def fit_zernike_coefficients(pol_state, aperture, u_axis, v_axis, zernike_order, aperture_radius, aperture_inlim):

    # Creating a unitary radius grid
    u_grid, v_grid = np.meshgrid(u_axis, v_axis)
    u_grid /= aperture_radius
    v_grid /= aperture_radius

    # Creating a mask for valid points
    radius = np.sqrt(u_grid**2+v_grid**2)
    mask = np.where(radius > 1, False, True)
    mask = np.where(radius < aperture_inlim, False, mask)

    # Vectorize grids with only valid points
    u_lin = u_grid[mask]
    v_lin = u_grid[mask]
    aperture_4d = aperture[:, :, :, mask]

