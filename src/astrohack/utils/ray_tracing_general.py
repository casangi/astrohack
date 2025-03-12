import numpy as np
from astrohack.utils.algorithms import least_squares_jit, least_squares
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
    return qps_coeffs, matrix, vector


def compute_qps_full_np(point_cloud):
    n_extra_coeffs = 6
    pcd_xy = point_cloud[:, 0:2]
    pcd_z = point_cloud[:, 2]

    npnt = pcd_xy.shape[0]
    n_var = npnt + n_extra_coeffs
    matrix_shape = (n_var, n_var)

    matrix = np.empty(matrix_shape)
    vector = np.zeros(n_var)

    matrix[0:npnt, 0:npnt] = cdist(pcd_xy, pcd_xy)**5
    matrix[0:npnt, npnt+0] = pcd_xy[:, 0] ** 2
    matrix[0:npnt, npnt+1] = pcd_xy[:, 0] * pcd_xy[:, 1]
    matrix[0:npnt, npnt+2] = pcd_xy[:, 1] ** 2
    matrix[0:npnt, npnt+3] = pcd_xy[:, 0]
    matrix[0:npnt, npnt+4] = pcd_xy[:, 1]
    matrix[0:npnt, npnt+5] = 1
    matrix[npnt:n_var, 0:npnt] = 1
    matrix[npnt:n_var, npnt:n_var] = 0
    vector[0:npnt] = pcd_z

    qps_coeffs, _, _ = least_squares(matrix, vector)
    return qps_coeffs, matrix, vector


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




