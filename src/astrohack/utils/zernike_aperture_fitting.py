import numpy as np
from scipy import optimize as opt
from astrohack.utils.algorithms import data_statistics
from astrohack.utils.text import statistics_to_text


def zernike_order_0(u_ax, v_ax):
    # N = 0
    return np.full([u_ax.shape[0], 1], 1.0)


def zernike_order_1(u_ax, v_ax):
    # N = 1
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 3])
    # Fill with previous order
    matrix[:, 0] = zernike_order_0(u_ax, v_ax)[:, 0]

    # M = -1
    matrix[:, 1] = u_ax
    # M = 1
    matrix[:, 2] = v_ax
    return matrix


def zernike_order_2(u_ax, v_ax, return_powers=False):
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 6])
    # Fill with previous order
    matrix[:, 0:3] = zernike_order_1(u_ax, v_ax)
    u_pow = [None, None, u_ax**2]
    v_pow = [None, None, v_ax**2]

    # M = -2
    matrix[:, 3] = 2*u_ax*v_ax
    # M = 0
    matrix[:, 4] = -1 + 2 * u_pow[2] + 2 * v_pow[2] ** 2
    # M = 2
    matrix[:, 5] = -u_pow[2] ** 2 + v_pow[2] ** 2

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_3(u_ax, v_ax, return_powers=False):
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 10])
    # Fill with previous order
    matrix[:, 0:6], u_pow, v_pow = zernike_order_2(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[2]*u_ax)
    v_pow.append(v_pow[2]*v_ax)

    # M = -3
    matrix[:, 6] = -u_pow[3] + 3 * u_ax * v_pow[2]
    # M = -1
    matrix[:, 7] = -2 * u_ax + 3 * u_pow[3] + 3 * u_ax * v_pow[2]
    # M = 1
    matrix[:, 8] = -2 * v_ax + 3 * v_pow[3] + 3 * u_pow[2] * v_ax
    # M = 3
    matrix[:, 9] = -v_pow[3] + 3 * v_pow[3] + 3 * u_pow[2] * v_ax

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_4(u_ax, v_ax, return_powers=False):
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 15])
    # Fill with previous order
    matrix[:, 0:10], u_pow, v_pow = zernike_order_3(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[2]**2)
    v_pow.append(v_pow[2]**2)

    # M = -4
    matrix[:, 10] = -4*u_pow[3]*v_ax + 4*u_ax*v_pow[3]
    # M = -2
    matrix[:, 11] = -6*u_ax*v_ax + 8*u_pow[3]*v_ax + 8*u_ax*v_pow[3]
    # M = 0
    matrix[:, 12] = 1 - 6*u_pow[2] - 6*v_pow[2] + 6*u_pow[4] + 12*u_pow[2]*v_pow[2] + 6*v_pow[4]
    # M = 2
    matrix[:, 13] = 3*u_pow[2] - 3*v_pow[2] - 4*u_pow[4] + 4*v_pow[4]
    # M = 4
    matrix[:, 13] = u_pow[4] - 6*u_pow[2]*v_pow[2] + v_pow[4]

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_5(u_ax, v_ax, return_powers=False):
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 21])
    # Fill with previous order
    matrix[:, 0:15], u_pow, v_pow = zernike_order_4(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[4] * u_ax)
    v_pow.append(v_pow[4] * v_ax)

    # M = -5
    matrix[:, 15] = u_pow[5] - 10*u_pow[3]*v_pow[2] + 5*u_ax*v_pow[4]
    # M = -3
    matrix[:, 16] = 4*u_pow[3] - 12*u_ax*v_pow[2] - 5*u_pow[5] + 10*u_pow[3]*v_pow[2] + 15*u_ax*v_pow[4]
    # M = -1
    matrix[:, 17] = (3 * u_ax - 12 * u_pow[3] - 12 * u_ax * v_pow[2] + 10 * u_pow[5] + 20 * u_pow[3] * v_pow[2] +
                     10 * u_ax * v_pow[4])
    # M = 1
    matrix[:, 18] = (3 * v_ax - 12 * v_pow[3] - 12 * v_ax * u_pow[2] + 10 * v_pow[5] + 20 * v_pow[3] * u_pow[2] +
                     10 * v_ax * u_pow[4])
    # M = 3
    matrix[:, 19] = - 4*v_pow[3] + 12*v_ax*u_pow[2] + 5*v_pow[5] - 10*v_pow[3]*u_pow[2] - 15*v_ax*u_pow[4]
    # M = 5
    matrix[:, 20] = v_pow[5] - 10*v_pow[3]*u_pow[2] + 5*v_ax*u_pow[4]

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


def zernike_order_6(u_ax, v_ax, return_powers=False):
    nlines = u_ax.shape[0]
    matrix = np.empty([nlines, 28])
    # Fill with previous order
    matrix[:, 0:21], u_pow, v_pow = zernike_order_5(u_ax, v_ax, return_powers=True)
    u_pow.append(u_pow[5] * u_ax)
    v_pow.append(v_pow[5] * v_ax)
    u3v = u_pow[3]*v_ax
    uv3 = u_ax*v_pow[3]
    u5v = u_pow[5]*v_ax
    uv5 = u_ax*v_pow[5]
    u3v3 = u_pow[3]*v_pow[3]
    u2v4 = u_pow[2]*v_pow[4]
    u4v2 = u_pow[4]*v_pow[2]
    u2v2 = u_pow[2]*v_pow[2]

    # M = -6
    matrix[:, 21] = 6*u5v - 20*u3v3 + 6*uv5
    # M = -4
    matrix[:, 22] = 20*u3v - 20*uv3 - 24*u5v + 24*uv5
    # M = -2
    matrix[:, 23] = 12*u_ax*v_ax - 40*u3v - 40*uv3 + 30*u5v + 60*u3v3 - 30*uv5
    # M = 0
    matrix[:, 24] = (-1 + 12*u_pow[2] + 12*v_pow[2] - 30*u_pow[4] - 60*u2v2 - 30*v_pow[4] + 20*u_pow[6] +
                     60*u4v2 + 60*u2v4 + 20*v_pow[6])
    # M = 2
    matrix[:, 25] = -6*u_pow[2] + 6*v_pow[2] + 20*u_pow[4] - 20*v_pow[4] - 15*u_pow[6] - 15*u4v2 + 15*u2v4 + 15*v_pow[6]
    # M = 4
    matrix[:, 26] = -5*u_pow[4] + 30*u2v2 - 5*v_pow[4] + 6*u_pow[6] - 30*u4v2 - 30*u2v4 + 6*v_pow[6]
    # M = 6
    matrix[:, 27] = -u_pow[6] + 15*u4v2 - 15*u2v4 + v_pow[6]

    print(np.sum(~np.isfinite(matrix)), )

    if return_powers:
        return matrix, u_pow, v_pow
    else:
        return matrix


zernike_matrix_functions = [zernike_order_0, zernike_order_1, zernike_order_2, zernike_order_3, zernike_order_4,
                            zernike_order_5, zernike_order_6]


def fit_zernike_coefficients(fitting_method, aperture, u_axis, v_axis, zernike_order, aperture_radius, aperture_inlim):
    # Creating a unitary radius grid
    u_grid, v_grid = np.meshgrid(u_axis, v_axis, indexing='ij')
    u_grid /= aperture_radius
    v_grid /= aperture_radius

    # Creating a mask for valid points
    radius = np.sqrt(u_grid**2+v_grid**2)
    mask = np.where(radius > 1, False, True)
    mask = np.where(radius < aperture_inlim/aperture_radius, False, mask)

    # Vectorize grids with only valid points
    u_lin = u_grid[mask]
    v_lin = u_grid[mask]
    aperture_4d = aperture[:, :, :, mask]

    # Creating grid reconstruction
    u_idx = np.arange(u_axis.shape[0], dtype=int)
    v_idx = np.arange(v_axis.shape[0], dtype=int)
    u_idx_grd, v_idx_grd = np.meshgrid(u_idx, v_idx, indexing='ij')
    uv_idx_grid = np.empty([u_lin.shape[0], 2], dtype=int)
    uv_idx_grid[:, 0] = u_idx_grd[mask]
    uv_idx_grid[:, 1] = v_idx_grd[mask]

    matrix_func = zernike_matrix_functions[zernike_order]
    matrix = matrix_func(u_lin, v_lin)

    if fitting_method == 'numpy least squares':
        solution_real, rms_real, model_real_lin = (
            _fit_an_aperture_plane_component_np_least_squares(matrix, aperture_4d[0, 0, 0, :].real))
        solution_imag, rms_imag, model_imag_lin = (
            _fit_an_aperture_plane_component_np_least_squares(matrix, aperture_4d[0, 0, 0, :].imag))
    elif fitting_method == 'scipy least squares':
        solution_real, rms_real, model_real_lin = (
            _fit_an_aperture_plane_component_scipy_opt_lst_sq(matrix, aperture_4d[0, 0, 0, :].real))
        solution_imag, rms_imag, model_imag_lin = (
            _fit_an_aperture_plane_component_scipy_opt_lst_sq(matrix, aperture_4d[0, 0, 0, :].imag))
    else:
        raise Exception(f"Unknown fitting method {fitting_method}")

    # Regridding model
    model = np.full_like(aperture, np.nan+np.nan*1j, dtype=complex)
    model[0, 0, 0, uv_idx_grid[:, 0], uv_idx_grid[:, 1]] = model_real_lin[:] + 1j*model_imag_lin[:]
    return solution_real, rms_real, solution_imag, rms_imag, model


def _fit_an_aperture_plane_component_np_least_squares(matrix, aperture_plane_comp):
    max_ap = np.nanmax(aperture_plane_comp)
    result, _, _, _ = np.linalg.lstsq(matrix, aperture_plane_comp/max_ap, rcond=None)
    model = max_ap*np.matmul(matrix, result)
    rms = np.sqrt(np.sum((aperture_plane_comp-model)**2))/model.shape[0]
    return result, rms, model


def _scipy_fitting_func(coeffs, matrix, aperture):
    return aperture - np.matmul(matrix, coeffs)


def _fit_an_aperture_plane_component_scipy_opt_lst_sq(matrix, aperture_plane_comp):
    initial_pars = np.ones(matrix.shape[1])
    max_ap = np.nanmax(aperture_plane_comp)
    norm_aperture = aperture_plane_comp/max_ap
    args = [matrix, norm_aperture]
    results = opt.least_squares(_scipy_fitting_func, initial_pars, args=args)
    model = max_ap*np.matmul(matrix, results.x)
    rms = np.sqrt(np.sum((aperture_plane_comp-model)**2))/model.shape[0]
    return results.x, rms, model






