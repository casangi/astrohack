import numpy as np

import toolviper.utils.logger as logger

from astrohack.utils.algorithms import least_squares_jit, least_squares, create_coordinate_images
from numba import njit
from scipy.spatial.distance import cdist
import pickle

nanvec3d = np.array([np.nan, np.nan, np.nan])
intblankval = -1000


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


@njit(cache=True, nogil=True)
def moller_trumbore_algorithm(ray_origin, ray_vector, pa, pb, pc):
    epsilon = 1e-10
    edge1 = pb - pa
    edge2 = pc - pa

    ray_cross_edge2 = np.cross(ray_vector, edge2)
    determinant = np.dot(edge1, ray_cross_edge2)

    if np.abs(determinant) < epsilon:
        return False, nanvec3d

    inv_det = 1.0 / determinant
    s = ray_origin - pa
    u = inv_det * np.dot(s, ray_cross_edge2)

    if (u < 0 and np.abs(u) > epsilon) or (u > 1 and np.abs(u - 1) > epsilon):
        return False, nanvec3d

    s_cross_e1 = np.cross(s, edge1)
    v = inv_det * np.dot(ray_vector, s_cross_e1)

    if (v < 0 and np.abs(v) > epsilon) or (u + v > 1 and np.abs(u + v - 1) > epsilon):
        return False, nanvec3d

    # At this stage we can compute t to find out where the intersection point is on the line.
    t = inv_det * np.dot(edge2, s_cross_e1)

    if t > epsilon: # ray intersection
        return True, ray_origin + ray_vector * t
    else:  # This means that there is a line intersection but not a ray intersection.
        return False, nanvec3d


@njit(cache=True, nogil=True)
def jitted_triangle_find(pr_point, reflection, sc_mesh, sc_pnt):
    for it, triangle in enumerate(sc_mesh):
        va = sc_pnt[int(triangle[0])]
        vb = sc_pnt[int(triangle[1])]
        vc = sc_pnt[int(triangle[2])]
        crosses_triangle, point = moller_trumbore_algorithm(pr_point, reflection, va, vb, vc)
        if crosses_triangle:
            return it, point

    return intblankval, nanvec3d


def find_closest_point_to_ray(ray_origin, ray_direction, pcd):
    point_vectors = pcd - ray_origin[np.newaxis, :]
    direction_norm = generalized_norm(ray_direction)
    # Project point_vector onto self.direction
    projection = generalized_dot(point_vectors, ray_direction[np.newaxis, :])/ direction_norm

    # Calculate the closest point on the line to the given point
    closest_points = ray_origin[np.newaxis, :] + projection[:, np.newaxis] * ray_direction[np.newaxis, :]
    # Calculate the distance**2 between the points and the closest
    # points on the ray
    distances2 = np.sum((pcd-closest_points)**2, axis=-1)
    i_closest = np.argmin(distances2, axis=-1)
    return i_closest, closest_points[i_closest], np.sqrt(distances2[i_closest])


def distance_from_ray_to_point(ray_origin, ray_direction, point):
    pnt_vec = point - ray_origin
    # Project point_vector onto ray_direction
    dir2 = np.sum(ray_direction**2, axis=-1)
    proj = np.sum(pnt_vec*ray_direction, axis=-1) / dir2

    # Calculate the closest point on the line to the given point
    clsst_pnt = ray_origin + proj * ray_direction
    # Calculate the distance**2 between the points and the closest
    # points on the self
    dist = np.sqrt(np.sum((point-clsst_pnt)**2))
    return dist, clsst_pnt


def np_qps_fitting(pcd):
    npnt = pcd.shape[0]
    pcd_xy = pcd[:, 0:2]
    dist_matrix = np.sqrt(np.sum((pcd_xy[np.newaxis, :, :]-pcd_xy[:, np.newaxis, :])**2, axis =-1))

    n_var = npnt+6
    sys_matrix = np.zeros([n_var, n_var])
    sys_vector = np.zeros([n_var])

    sys_matrix[:npnt, :npnt] = dist_matrix**5
    sys_matrix[:npnt, npnt+0] = pcd_xy[:, 0] ** 2
    sys_matrix[:npnt, npnt+1] = pcd_xy[:, 0] * pcd_xy[:, 1]
    sys_matrix[:npnt, npnt+2] = pcd_xy[:, 1] ** 2
    sys_matrix[:npnt, npnt+3] = pcd_xy[:, 0]
    sys_matrix[:npnt, npnt+4] = pcd_xy[:, 1]
    sys_matrix[:npnt, npnt+5] = 1
    sys_matrix[npnt:, :npnt] = 1.0

    sys_vector[:npnt] = pcd[:, 2]
    qps_coeffs, _, _ = least_squares(sys_matrix, sys_vector)
    return qps_coeffs


def qps_compute_point_and_normal(pnt, qps_coeffs, pcd):
    npnt = pcd.shape[0]
    acoeffs = qps_coeffs[:npnt]
    bcoeffs = qps_coeffs[npnt:]
    pnt_xy = pnt[0:2]
    diff = pcd[:, 0:2] - pnt_xy
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    aterm_val = np.sum(acoeffs * dist**5)
    cubic_rterm = acoeffs * dist**3
    aterm_dx = 5 * np.sum(diff[:, 0] * cubic_rterm)
    aterm_dy = 5 * np.sum(diff[:, 1] * cubic_rterm)

    qps_val = aterm_val + bcoeffs[0]*pnt[0]**2 + bcoeffs[1]*pnt[0]*pnt[1] + bcoeffs[2]*pnt[1]**2
    qps_val += bcoeffs[3]*pnt[0] + bcoeffs[4]*pnt[1] + bcoeffs[5]

    dqps_dx = aterm_dx + 2*bcoeffs[0]*pnt[0] + bcoeffs[1]*pnt[1] + bcoeffs[3]

    dqps_dy = aterm_dy + bcoeffs[1]*pnt[0] + 2*bcoeffs[2]*pnt[1] + bcoeffs[4]

    normal = normalize_vector_map(np.array([-dqps_dx, -dqps_dy, 1]))
    new_pnt = np.array([pnt[0], pnt[1], qps_val])
    return new_pnt, normal


def qps_compute_point(pnt, qps_coeffs, pcd):
    npnt = pcd.shape[0]
    acoeffs = qps_coeffs[:npnt]
    bcoeffs = qps_coeffs[npnt:]
    pnt_xy = pnt[0:2]
    diff = pcd[:, 0:2] - pnt_xy
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    aterm_val = np.sum(acoeffs * dist**5)

    qps_val = aterm_val + bcoeffs[0]*pnt[0]**2 + bcoeffs[1]*pnt[0]*pnt[1] + bcoeffs[2]*pnt[1]**2
    qps_val += bcoeffs[3]*pnt[0] + bcoeffs[4]*pnt[1] + bcoeffs[5]

    new_pnt = np.array([pnt[0], pnt[1], qps_val])
    return new_pnt


class LocalQPS:
    n_qps_extra_vars = 6

    def __init__(self):
        # Meta data
        self.npnt = -1
        self.local_qps_n_pnt = -1

        # Data arrays
        self.global_pcd = None
        self.local_qps_coeffs = None
        self.local_pcds = None

    @classmethod
    def from_pcd(cls, pcd_data, local_qps_n_pnt=20):
        new_obj = cls()
        new_obj._init_from_pcd(pcd_data, local_qps_n_pnt)
        return new_obj

    def _init_from_pcd(self, pcd_data, local_qps_n_pnt):
        self.global_pcd = pcd_data
        self.npnt = self.global_pcd.shape[0]
        self.local_qps_n_pnt = local_qps_n_pnt
        self.local_pcds = np.empty([self.npnt, self.local_qps_n_pnt, 3])
        self.local_qps_coeffs = np.empty([self.npnt, local_qps_n_pnt + self.n_qps_extra_vars])
        for ipnt, point in enumerate(self.global_pcd):
            dist2 = np.sum((point[np.newaxis, :]-self.global_pcd)**2, axis=-1)
            n_closest = np.argsort(dist2)[:self.local_qps_n_pnt]
            self.local_pcds[ipnt] = self.global_pcd[n_closest]
            self.local_qps_coeffs[ipnt] = np_qps_fitting(self.global_pcd[n_closest])

    @classmethod
    def from_pickle(cls, filename):
        with open(filename, 'rb') as pickled_file:
            pkl_obj = pickle.load(pickled_file)
            return pkl_obj

    def get_local_qps(self, ipnt):
        return self.local_qps_coeffs[ipnt], self.local_pcds[ipnt]
        # qps_obj = QPS(self.qps_coeffs[ipnt], self.local_pcds[ipnt])
        # return qps_obj

    def __sizeof__(self):
        total_size = 0
        for key, item in self.__dict__.items():
            total_size += item.__sizeof__()
        return total_size

    def to_pickle(self, filename):
        with open(filename, 'wb') as pickle_file:
            # noinspection PyTypeChecker
            pickle.dump(self, pickle_file)

    def grid_primary(self, sampling, active_radius=9.0, x_off=None, y_off=None):
        if x_off is None:
            x_off = find_mid_point(self.global_pcd[:, 0])
        if y_off is None:
            y_off = find_mid_point(self.global_pcd[:, 1])

        x_axis = simple_axis([-active_radius + x_off, active_radius + x_off], sampling)
        y_axis = simple_axis([-active_radius + y_off, active_radius + y_off], sampling)

        nx = x_axis.shape[0]
        ny = y_axis.shape[0]
        npnt_max = nx*ny

        grd_surface = np.empty((npnt_max, 3))
        grd_normal = np.empty((npnt_max, 3))
        grd_idx = np.empty((npnt_max, 2))

        i_pnt = 0
        for ix, x_val in enumerate(x_axis):
            xsel = np.abs(self.global_pcd[:, 0] - x_val) < 3 * sampling
            for iy, y_val in enumerate(y_axis):
                if ((x_val-x_off)**2 + (y_val-y_off)**2) <= active_radius**2:
                    ysel = np.abs(self.global_pcd[:, 1] - y_val) < 3 * sampling
                    fullsel = xsel & ysel
                    sel_qps = self.local_qps_coeffs[fullsel]
                    sel_gl_pcd = self.global_pcd[fullsel]
                    sel_lc_pcd = self.local_pcds[fullsel]

                    i_closest = np.argmin((sel_gl_pcd[:, 0]-x_val)**2+(sel_gl_pcd[:, 1]-y_val)**2)

                    point, normal = qps_compute_point_and_normal(np.array([x_val, y_val]), sel_qps[i_closest],
                                                                 sel_lc_pcd[i_closest])

                    grd_surface[i_pnt, :] = point
                    grd_normal[i_pnt, :] = normal
                    grd_idx[i_pnt, :] = ix, iy

                    i_pnt += 1
        n_total = i_pnt

        return x_axis, y_axis, grd_surface[:n_total, :], grd_normal[:n_total, :], grd_idx[:n_total, :]

    def find_reflection_point(self, ray_origin, ray_direction, wavelength, nitermax=1000):
        epsilon = wavelength / 32.
        i_closest, ray_pnt, dist = find_closest_point_to_ray(ray_origin, ray_direction, self.global_pcd)
        loc_qps_coeff, loc_qps_pcd = self.get_local_qps(i_closest)
        looking = True
        niter = 0
        # Here I am looking for a way to determine movement direction
        # using XY info
        # distance has to be measured in 3D but steps are in 2D
        test_pnt = ray_pnt
        while looking:
            if dist < epsilon:
                looking = False
            elif niter < nitermax:
                test_pnt = (test_pnt + 5 * ray_pnt)/6
                test_pnt = qps_compute_point(test_pnt, loc_qps_coeff, loc_qps_pcd)
                dist, ray_pnt = distance_from_ray_to_point(ray_origin, ray_direction, test_pnt)
            else:
                looking = False
            niter += 1

        return qps_compute_point_and_normal(ray_pnt, loc_qps_coeff, loc_qps_pcd)
