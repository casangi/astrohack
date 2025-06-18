import numpy as np
from astrohack.utils.algorithms import least_squares

nanvec3d = np.array([np.nan, np.nan, np.nan])


def generalized_dot(vec_map_a, vec_map_b):
    return np.sum(vec_map_a * vec_map_b, axis=-1)


def generalized_norm(vecmap):
    return np.sqrt(generalized_dot(vecmap, vecmap))


def generalized_dist(vec_map_a, vec_map_b):
    return np.sqrt(np.sum((vec_map_a - vec_map_b) ** 2, axis=-1))


def normalize_vector_map(vector_map):
    normalization = np.linalg.norm(vector_map, axis=-1)
    return vector_map / normalization[..., np.newaxis]


def reflect_light(light, normals):
    return light - 2 * generalized_dot(light, normals)[..., np.newaxis] * normals


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

def np_qps_fitting(pcd):
    npnt = pcd.shape[0]
    pcd_xy = pcd[:, 0:2]
    dist_matrix = generalized_dist(pcd_xy[np.newaxis, :, :], pcd_xy[:, np.newaxis, :])

    n_var = npnt + 6
    sys_matrix = np.zeros([n_var, n_var])
    sys_vector = np.zeros([n_var])

    sys_matrix[:npnt, :npnt] = dist_matrix**5
    sys_matrix[:npnt, npnt + 0] = pcd_xy[:, 0] ** 2
    sys_matrix[:npnt, npnt + 1] = pcd_xy[:, 0] * pcd_xy[:, 1]
    sys_matrix[:npnt, npnt + 2] = pcd_xy[:, 1] ** 2
    sys_matrix[:npnt, npnt + 3] = pcd_xy[:, 0]
    sys_matrix[:npnt, npnt + 4] = pcd_xy[:, 1]
    sys_matrix[:npnt, npnt + 5] = 1
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

    qps_val = (
        aterm_val
        + bcoeffs[0] * pnt[0] ** 2
        + bcoeffs[1] * pnt[0] * pnt[1]
        + bcoeffs[2] * pnt[1] ** 2
    )
    qps_val += bcoeffs[3] * pnt[0] + bcoeffs[4] * pnt[1] + bcoeffs[5]

    dqps_dx = aterm_dx + 2 * bcoeffs[0] * pnt[0] + bcoeffs[1] * pnt[1] + bcoeffs[3]

    dqps_dy = aterm_dy + bcoeffs[1] * pnt[0] + 2 * bcoeffs[2] * pnt[1] + bcoeffs[4]

    normal = normalize_vector_map(np.array([-dqps_dx, -dqps_dy, 1]))
    new_pnt = np.array([pnt[0], pnt[1], qps_val])
    return new_pnt, normal


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
        self.local_qps_coeffs = np.empty(
            [self.npnt, local_qps_n_pnt + self.n_qps_extra_vars]
        )
        for ipnt, point in enumerate(self.global_pcd):
            dist2 = np.sum((point[np.newaxis, :] - self.global_pcd) ** 2, axis=-1)
            n_closest = np.argsort(dist2)[: self.local_qps_n_pnt]
            self.local_pcds[ipnt] = self.global_pcd[n_closest]
            self.local_qps_coeffs[ipnt] = np_qps_fitting(self.global_pcd[n_closest])

    def export_to_xr_data_variables(self):
        """
        Idea is to return pcd values and qpd coeffs as Xarray data variables for storage.
        Returns: xarray data variables

        """
        return

    @classmethod
    def from_xr_data_variable(cls):
        """
        Idea is to init an object from a few xarray Data variables from storage
        Returns: initialized obj

        """
        return

    def compute_zval_and_z_angle(self, point):
        """
        Idea is to compute value of QPS and angle with boresight
        Args:
            point:

        Returns:

        """
