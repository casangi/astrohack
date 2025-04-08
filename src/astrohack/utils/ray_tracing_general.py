import numpy as np

import toolviper.utils.logger as logger

from astrohack.utils.algorithms import least_squares_jit, least_squares, create_coordinate_images
from numba import njit
from scipy.spatial.distance import cdist


def generalized_dot(vec_map_a, vec_map_b):
    return np.sum(vec_map_a * vec_map_b, axis=-1)


def generalized_norm(vecmap):
    return np.sqrt(generalized_dot(vecmap, vecmap))


def normalize_vector_map(vector_map):
    normalization = np.linalg.norm(vector_map, axis=-1)
    return vector_map / normalization[..., np.newaxis]


def reflect_light(light, normals):
    return light - 2 * generalized_dot(light, normals)[..., np.newaxis] * normals


@njit(cache=True, nogil=True)
def compute_quintic_pseudo_spline_coefficients(point_cloud):
    # QPS definition from Bergman et al. 1994, IEEE Transactions on Antennas and propagation
    def dist_2d(pnt_a, pnt_b):
        return np.sqrt(np.sum((pnt_a-pnt_b)**2))

    n_extra_coeffs = 6
    pcd_xy = point_cloud[:, 0:2]
    pcd_z = point_cloud[:, 2]

    npnt = pcd_xy.shape[0]
    n_var = npnt + n_extra_coeffs
    matrix_shape = (n_var, n_var)

    matrix = np.zeros(matrix_shape)
    vector = np.zeros(n_var)

    # Building system's right-hand side vector
    vector[0:npnt] = pcd_z

    print()
    # Building system matrix
    for irow in range(npnt):
        for icol in range(npnt):
            matrix[irow, icol] = dist_2d(pcd_xy[irow], pcd_xy[icol]) ** 5

        matrix[irow, npnt+0] = pcd_xy[irow, 0] ** 2
        matrix[irow, npnt+1] = pcd_xy[irow, 0] * pcd_xy[irow, 1]
        matrix[irow, npnt+2] = pcd_xy[irow, 1] ** 2
        matrix[irow, npnt+3] = pcd_xy[irow, 0]
        matrix[irow, npnt+4] = pcd_xy[irow, 1]
        matrix[irow, npnt+5] = 1

        print('\033[F', (irow+1)/npnt*100, '%              ')

    for irow in range(npnt, n_var):
        matrix[irow, 0:npnt] = 1

    qps_coeffs, _, _, _ = least_squares_jit(matrix, vector)
    return qps_coeffs


def compute_qps_value(pnt, qps_coeffs, point_cloud):
    # QPS definition from Bergman et al. 1994, IEEE Transactions on Antennas and propagation
    npnt = point_cloud.shape[0]
    pcd_xy = point_cloud[:, 0:2]
    a_coeffs = qps_coeffs[0:npnt]
    b_coeffs = qps_coeffs[npnt:]
    r_term = np.sum(a_coeffs*np.sqrt(np.sum((pnt[np.newaxis, 0:2]-pcd_xy)**2, axis=1))**5)

    z_val  = r_term + b_coeffs[0]*pnt[0]**2 + b_coeffs[1]*pnt[0]*pnt[1] + b_coeffs[2]*pnt[1]**2
    z_val += b_coeffs[3]*pnt[0] + b_coeffs[4]*pnt[1] + b_coeffs[5]

    return z_val


def qps_pcd_fitting(point_cloud_filename, output_coeff_filename, max_rows=None):
    pcd_data = np.loadtxt(point_cloud_filename, max_rows=max_rows)
    qps_coeffs = compute_quintic_pseudo_spline_coefficients(pcd_data)
    np.save(output_coeff_filename, qps_coeffs)

    return qps_coeffs


def find_mid_point(array):
    return (np.max(array)+np.min(array))/2


@njit(cache=True, nogil=True)
def grid_qps_jit(x_axis, y_axis, point_cloud, qps_coeffs, active_radius, x_off, y_off):
    nx = x_axis.shape[0]
    ny = y_axis.shape[0]
    npnt = point_cloud.shape[0]
    gridded = np.empty((nx, ny), dtype=np.float32)
    a_coeffs = qps_coeffs[0:npnt]
    b_coeffs = qps_coeffs[npnt:]
    pcd_xy = point_cloud[:, 0:2]

    for ix, x_val in enumerate(x_axis):
        for iy, y_val in enumerate(y_axis):
            if ((x_val-x_off)**2 + (y_val-y_off)**2) <= active_radius**2:
                pnt = np.array([x_val, y_val])
                r_term = np.sum(a_coeffs*np.sqrt(np.sum((pnt[np.newaxis, 0:2]-pcd_xy)**2, axis=1))**5)
                z_val  = r_term + b_coeffs[0]*pnt[0]**2 + b_coeffs[1]*pnt[0]*pnt[1] + b_coeffs[2]*pnt[1]**2
                z_val += b_coeffs[3]*pnt[0] + b_coeffs[4]*pnt[1] + b_coeffs[5]
                gridded[ix, iy] = z_val
            else:
                gridded[ix, iy] = np.nan

    return gridded


@njit(cache=True, nogil=True)
def compute_qps_one_point(x_val, y_val, a_coeffs, b_coeffs, pcd_xy):
    pnt_xy = np.array([x_val, y_val])
    r_term = np.sum(a_coeffs*np.sqrt(np.sum((pnt_xy[np.newaxis, :]-pcd_xy)**2, axis=1))**5)
    z_val = r_term + b_coeffs[0]*pnt_xy[0]**2 + b_coeffs[1]*pnt_xy[0]*pnt_xy[1] + b_coeffs[2]*pnt_xy[1]**2
    z_val += b_coeffs[3]*pnt_xy[0] + b_coeffs[4]*pnt_xy[1] + b_coeffs[5]
    return z_val


@njit(cache=True, nogil=True)
def grid_qps_jit_1d(x_axis, y_axis, point_cloud, qps_coeffs, active_radius, x_off, y_off):
    nx = x_axis.shape[0]
    ny = y_axis.shape[0]

    new_pcd = np.empty((nx*ny, 3))
    new_idx = np.empty((nx*ny, 2))

    npnt = point_cloud.shape[0]
    a_coeffs = qps_coeffs[0:npnt]
    b_coeffs = qps_coeffs[npnt:]
    pcd_xy = point_cloud[:, 0:2]

    i_pnt = 0
    for ix, x_val in enumerate(x_axis):
        for iy, y_val in enumerate(y_axis):
            if ((x_val-x_off)**2 + (y_val-y_off)**2) <= active_radius**2:
                pnt = np.array([x_val, y_val])
                r_term = np.sum(a_coeffs*np.sqrt(np.sum((pnt[np.newaxis, 0:2]-pcd_xy)**2, axis=1))**5)
                z_val  = r_term + b_coeffs[0]*pnt[0]**2 + b_coeffs[1]*pnt[0]*pnt[1] + b_coeffs[2]*pnt[1]**2
                z_val += b_coeffs[3]*pnt[0] + b_coeffs[4]*pnt[1] + b_coeffs[5]
                new_pcd[i_pnt] = np.array((x_val, y_val, z_val))
                new_idx[i_pnt] = np.array((ix, iy))
                i_pnt += 1

    n_total = i_pnt
    return new_pcd[:n_total, :], new_idx[:n_total, :]

@njit(cache=True, nogil=True)
def grid_qps_plus_fdd_jit_1d(x_axis, y_axis, point_cloud, qps_coeffs, active_radius, x_off, y_off, fdd_epsilon):
    nx = x_axis.shape[0]
    ny = y_axis.shape[0]
    npnt_max = nx*ny

    qps_pcd = np.empty((npnt_max, 3))
    qps_pcd_dx = np.empty((npnt_max, 3))
    qps_pcd_dy = np.empty((npnt_max, 3))
    grid_idx = np.empty((npnt_max, 2))

    npnt_pcd = point_cloud.shape[0]
    a_coeffs = qps_coeffs[0:npnt_pcd]
    b_coeffs = qps_coeffs[npnt_pcd:]
    pcd_xy = point_cloud[:, 0:2]

    i_pnt = 0
    for ix, x_val in enumerate(x_axis):
        for iy, y_val in enumerate(y_axis):
            if ((x_val-x_off)**2 + (y_val-y_off)**2) <= active_radius**2:
                qps_pcd[i_pnt, 0] = x_val
                qps_pcd[i_pnt, 1] = y_val
                qps_pcd[i_pnt, 2] = compute_qps_one_point(x_val, y_val, a_coeffs, b_coeffs, pcd_xy)
                qps_pcd_dx[i_pnt, 0] = 1.
                qps_pcd_dx[i_pnt, 1] = 0.
                qps_pcd_dx[i_pnt, 2] = (compute_qps_one_point(x_val + fdd_epsilon, y_val, a_coeffs, b_coeffs, pcd_xy)
                                        - qps_pcd[i_pnt, 2])/ fdd_epsilon
                qps_pcd_dy[i_pnt, 0] = 0.
                qps_pcd_dy[i_pnt, 1] = 1.
                qps_pcd_dy[i_pnt, 2] = (compute_qps_one_point(x_val, y_val + fdd_epsilon, a_coeffs, b_coeffs, pcd_xy)
                                        - qps_pcd[i_pnt, 2]) / fdd_epsilon
                grid_idx[i_pnt, 0] = ix
                grid_idx[i_pnt, 1] = iy
                i_pnt += 1
    n_total = i_pnt
    return qps_pcd[:n_total, :], qps_pcd_dx[:n_total, :], qps_pcd_dy[:n_total, :], grid_idx[:n_total, :]


def grid_qps_primary(point_cloud, qps_coeffs, sampling, active_radius=9.0, x_off=None, y_off=None):
    if x_off is None:
        x_off = find_mid_point(point_cloud[:, 0])
    if y_off is None:
        y_off = find_mid_point(point_cloud[:, 1])

    x_axis = simple_axis([-active_radius+x_off, active_radius+x_off], sampling)
    y_axis = simple_axis([-active_radius+y_off, active_radius+y_off], sampling)

    gridded_qps = grid_qps_jit(x_axis, y_axis, point_cloud, qps_coeffs, active_radius, x_off, y_off)
    # x_mesh, y_mesh = create_coordinate_images(x_axis, y_axis, create_polar_coordinates=False)
    return gridded_qps


def grid_qps_with_fdd_normals(point_cloud, qps_coeffs, sampling, active_radius=9.0, x_off=None, y_off=None,
                              fdd_epsilon=1e-5):
    if x_off is None:
        x_off = find_mid_point(point_cloud[:, 0])
    if y_off is None:
        y_off = find_mid_point(point_cloud[:, 1])

    x_axis = simple_axis([-active_radius + x_off, active_radius + x_off], sampling)
    y_axis = simple_axis([-active_radius + y_off, active_radius + y_off], sampling)
    qps_pcd, qps_pcd_dx, qps_pcd_dy, grid_idx = grid_qps_plus_fdd_jit_1d(x_axis, y_axis, point_cloud, qps_coeffs,
                                                                         active_radius, x_off, y_off, fdd_epsilon)
    qps_normals = normalize_vector_map(np.cross(qps_pcd_dx, qps_pcd_dy))
    return qps_pcd, qps_normals, grid_idx



def degrade_pcd(pcd_file, new_pcd_file, factor):
    pcd_data = np.loadtxt(pcd_file)
    npnt = pcd_data.shape[0]
    new_size = npnt//factor
    new_pcd = np.empty((new_size, 3))
    for i_new in range(new_size):
        i_old = i_new * factor
        new_pcd[i_new] = pcd_data[i_old]

    ext = new_pcd_file.split('.')[-1]
    if ext == 'npy':
        np.save(new_pcd_file, new_pcd)
    elif ext in ['dat', 'txt']:
        np.savetxt(new_pcd_file, new_pcd)
    else:
        logger.warning(f'Unknown extension {ext} degraded point cloud not saved to disk')

    return new_pcd


def simple_axis(minmax, resolution, margin=0.05):
    """
    Creates an array spaning from min to max (may go over max if resolution is not an integer division) spaced by \
    resolution
    Args:
        minmax: the minimum and maximum of the axis
        resolution: The spacing between array elements
        margin: Add a margin at the edge of the array beyonf min and max
    Returns:
        A numpy array representation of a linear axis.
    """
    mini, maxi = minmax
    ax_range = maxi - mini
    pad = margin * ax_range
    if pad < np.abs(resolution):
        pad = np.abs(resolution)
    mini -= pad
    maxi += pad
    npnt = int(np.ceil((maxi - mini) / resolution))
    axis_array = np.arange(npnt + 1)
    axis_array = resolution * axis_array
    axis_array = axis_array + mini + resolution / 2
    return axis_array
