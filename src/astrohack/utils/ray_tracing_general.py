import xarray as xr
import pickle

from scipy.spatial import distance_matrix
from numba import njit
from numpy.linalg import LinAlgError

from astrohack.utils.algorithms import (
    least_squares,
    create_2d_array_reconstruction_array,
    create_coordinate_images,
    regrid_data_onto_2d_grid,
)
from astrohack.visualization.plot_tools import *

nanvec3d = np.array([np.nan, np.nan, np.nan])
return_line = "\033[F"


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
    try:
        qps_coeffs, _, _ = least_squares(sys_matrix, sys_vector)
    except LinAlgError:
        print(sys_matrix, sys_vector)
        raise LinAlgError
    return qps_coeffs


def degrade_pcd(pcd, degrading_factor):
    """
    Degrades de number of points in a pcd by an integer factor
    Args:
        pcd: pcd data, assumes [:, 3]
        degrading_factor: integer pcd degrading factor

    Returns: degraded pcd
    """
    n_out = int(np.ceil(pcd.shape[0] / degrading_factor))
    rng = np.random.default_rng()
    rand_item = rng.integers(degrading_factor, size=n_out)
    rand_idx = degrading_factor * np.arange(n_out) + rand_item
    if rand_idx[-1] >= pcd.shape[0]:  # Last item goes overboard
        rand_idx[-1] = pcd.shape[0] - 1
    degraded_pcd = pcd[rand_idx, :]
    return degraded_pcd


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


@njit()
def qps_compute_point_and_normal_jit(pnt, qps_coeffs, pcd):
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

    normal = np.array([-dqps_dx, -dqps_dy, 1])
    normal /= np.sqrt(np.sum(normal**2))
    # normal = normalize_vector_map(np.array([-dqps_dx, -dqps_dy, 1]))
    new_pnt = np.array([pnt[0], pnt[1], qps_val])
    return new_pnt, normal


@njit()
def qps_compute_normal_jit(pnt, qps_coeffs, pcd):
    npnt = pcd.shape[0]
    acoeffs = qps_coeffs[:npnt]
    bcoeffs = qps_coeffs[npnt:]
    pnt_xy = pnt[0:2]
    diff = pcd[:, 0:2] - pnt_xy
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    cubic_rterm = acoeffs * dist**3
    aterm_dx = 5 * np.sum(diff[:, 0] * cubic_rterm)
    aterm_dy = 5 * np.sum(diff[:, 1] * cubic_rterm)

    dqps_dx = aterm_dx + 2 * bcoeffs[0] * pnt[0] + bcoeffs[1] * pnt[1] + bcoeffs[3]
    dqps_dy = aterm_dy + bcoeffs[1] * pnt[0] + 2 * bcoeffs[2] * pnt[1] + bcoeffs[4]

    normal = np.array([-dqps_dx, -dqps_dy, 1])
    normal /= np.sqrt(np.sum(normal**2))
    return normal


@njit()
def local_qps_image_jit(global_pcd, local_qps_coeffs, local_pcds, points):
    npnt = points.shape[0]
    new_zval = np.empty(npnt, dtype=np.float64)
    new_norm = np.empty((npnt, 3), dtype=np.float64)
    print()
    for ipnt in range(npnt):
        print(return_line, 100 * ipnt / npnt, "% done     ")
        dist = np.sum((global_pcd[:, 0:2] - points[ipnt]) ** 2, axis=-1)
        i_closest = np.argmin(dist)
        pnt, norm = qps_compute_point_and_normal_jit(
            points[ipnt], local_qps_coeffs[i_closest], local_pcds[i_closest]
        )
        new_zval[ipnt] = pnt[2]
        new_norm[ipnt] = norm
    print("Done")
    return new_zval, new_norm


@njit()
def global_qps_image_jit(pcd, qps_coeffs, points):
    npnt = points.shape[0]
    new_zval = np.empty(npnt, dtype=np.float64)
    new_norm = np.empty((npnt, 3), dtype=np.float64)
    # print()
    for ipnt in range(npnt):
        # if ipnt%100 == 0:
        #     print(return_line, 100*ipnt/npnt, '% done     ')
        pnt, norm = qps_compute_point_and_normal_jit(points[ipnt], qps_coeffs, pcd)
        new_zval[ipnt] = pnt[2]
        new_norm[ipnt] = norm
    # print(return_line, '100% Done                  ')
    return new_zval, new_norm


@njit()
def global_qps_normal_image_jit(pcd, qps_coeffs, points):
    npnt = points.shape[0]
    new_norm = np.empty((npnt, 3), dtype=np.float64)
    # print()
    for ipnt in range(npnt):
        # if ipnt%100 == 0:
        #     print(return_line, 100*ipnt/npnt, '% done     ')
        norm = qps_compute_normal_jit(points[ipnt], qps_coeffs, pcd)
        new_norm[ipnt] = norm
    # print(return_line, '100% Done                  ')
    return new_norm


class GlobalQPS:
    n_qps_extra_vars = 6

    def __init__(self):
        # Meta data
        self.n_points = None

        # Data arrays
        self.point_cloud = None
        self.qps_coefficients = None

    @classmethod
    def from_point_cloud(
        cls, pcd_data, degradation_factor=None, displacement=(0, 0, 0)
    ):
        new_obj = cls()
        new_obj._init_from_point_cloud(pcd_data, degradation_factor, displacement)
        return new_obj

    @classmethod
    def from_point_cloud_and_coefficients(cls, point_cloud, qps_coefficients):
        new_obj = cls()
        new_obj.point_cloud = point_cloud
        new_obj.qps_coefficients = qps_coefficients
        new_obj.n_points = point_cloud.shape[0]
        return new_obj

    def _init_from_point_cloud(self, pcd_data, degradation_factor, displacement):
        if degradation_factor is None:
            self.point_cloud = pcd_data
        else:
            self.point_cloud = degrade_pcd(pcd_data, degradation_factor)

        self.point_cloud[:,] -= np.array(displacement)
        self.qps_coefficients = np_qps_fitting(self.point_cloud)

    @classmethod
    def from_pickle(cls, filename):
        with open(filename, "rb") as pickled_file:
            pkl_obj = pickle.load(pickled_file)
            return pkl_obj

    def to_pickle(self, filename):
        with open(filename, "wb") as pickle_file:
            # noinspection PyTypeChecker
            pickle.dump(self, pickle_file)

    def __sizeof__(self):
        total_size = 0
        for key, item in self.__dict__.items():
            total_size += item.__sizeof__()
        return total_size

    @classmethod
    def from_zarr(cls, filepath):
        new_obj = cls()
        xds = xr.open_zarr(filepath)
        new_obj.n_points = xds.attrs["n_points"]
        new_obj.point_cloud = xds["point_cloud"].values
        new_obj.qps_coefficients = xds["qps_coefficients"].values

    def to_zarr(self, filepath):
        xds = xr.Dataset()

        xds.attrs = {"n_points": self.n_points}

        xds["qps_coefficients"] = xr.DataArray(self.qps_coefficients, dims=["qps_axis"])
        xds["point_cloud"] = xr.DataArray(self.point_cloud, dims=["point_axis", "xyz"])
        qps_axis = [f"A{ipnt}" for ipnt in range(self.n_points)]
        qps_axis.extend([f"B{ipnt}" for ipnt in range(self.n_qps_extra_vars)])

        coords = {
            "xyz": ["x", "y", "z"],
            "point": list(range(self.n_points)),
            "qps": qps_axis,
        }
        xds = xds.assign_coords(coords)
        xds.to_zarr(filepath, mode="w", compute=True, consolidated=True)

    def compute_gridded_z_cos(self, u_axis, v_axis, mask, light=(0, 0, 1)):
        light = np.array(light)

        u_mesh, v_mesh = create_coordinate_images(u_axis, v_axis)
        uv_idx_grid = create_2d_array_reconstruction_array(u_axis, v_axis, mask)

        uv_points = np.empty_like(uv_idx_grid)
        uv_points[:, 0] = u_mesh[mask]
        uv_points[:, 1] = v_mesh[mask]

        z_norm = global_qps_normal_image_jit(
            self.point_cloud, self.qps_coefficients, uv_points
        )

        z_cos = (generalized_dot(z_norm, light)) / (
            generalized_norm(z_norm) * generalized_norm(light)
        )

        z_cos_grid = regrid_data_onto_2d_grid(u_axis, v_axis, z_cos, uv_idx_grid)

        return z_cos_grid

    def compute_gridded_z_val_and_z_cos(self, u_axis, v_axis, mask, light=(0, 0, 1)):
        light = np.array(light)

        u_mesh, v_mesh = create_coordinate_images(u_axis, v_axis)
        uv_idx_grid = create_2d_array_reconstruction_array(u_axis, v_axis, mask)

        uv_points = np.empty_like(uv_idx_grid)
        uv_points[:, 0] = u_mesh[mask]
        uv_points[:, 1] = v_mesh[mask]

        z_val, z_norm = global_qps_image_jit(
            self.point_cloud, self.qps_coefficients, uv_points
        )

        z_angle = (generalized_dot(z_norm, light)) / (
            generalized_norm(z_norm) * generalized_norm(light)
        )

        z_val_grid = regrid_data_onto_2d_grid(u_axis, v_axis, z_val, uv_idx_grid)
        z_cos_grid = regrid_data_onto_2d_grid(u_axis, v_axis, z_angle, uv_idx_grid)

        return z_val_grid, z_cos_grid


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

        self.current_z_val = None
        self.current_z_cos = None
        self.current_u_axis = None
        self.current_v_axis = None

        self.high_res_z_val = None
        self.high_res_z_cos = None
        self.high_res_u_axis = None
        self.high_res_v_axis = None

    @classmethod
    def from_pcd(cls, pcd_data, local_qps_n_pnt=20, displacement=(0, 0, 0)):
        new_obj = cls()
        new_obj._init_from_pcd(pcd_data, local_qps_n_pnt, displacement)
        return new_obj

    def _init_from_pcd(self, pcd_data, local_qps_n_pnt, displacement):
        self.global_pcd = pcd_data
        self.global_pcd[:,] -= np.array(displacement)
        self.npnt = self.global_pcd.shape[0]
        self.local_qps_n_pnt = local_qps_n_pnt
        self.local_pcds = np.empty([self.npnt, self.local_qps_n_pnt, 3])
        self.local_qps_coeffs = np.empty(
            [self.npnt, local_qps_n_pnt + self.n_qps_extra_vars]
        )
        print("0% done")
        for ipnt, point in enumerate(self.global_pcd):
            print(f"{return_line}{100*ipnt/self.npnt:.2f}% done       ")
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

    def compute_z_val_and_z_cos(self, point):
        """
        Idea is to compute value of QPS and angle with boresight
        Args:
            point:

        Returns:

        """
        dist = generalized_dist(self.global_pcd[:, 0:2], point)
        i_closest = np.argmin(dist)
        full_pnt, normal = qps_compute_point_and_normal(
            point, self.local_qps_coeffs[i_closest], self.local_pcds[i_closest]
        )
        z_val = full_pnt[2]
        return z_val, normal

    def vectorized_z_val_and_z_cos(self, point_arr):
        npnt = point_arr.shape[0]
        main_idx = np.arange(npnt)
        vec_qps = np.vectorize(qps_compute_point_and_normal)
        z_val = np.empty(npnt)
        z_norm = np.empty([npnt, 3])

        def execute_selection(pcd_selection, point_sel):
            pnt_retrieve = main_idx[point_sel]
            dist_matrix = distance_matrix(
                self.global_pcd[pcd_selection, 0:2], point_arr[point_sel]
            )
            i_closest_arr = np.argmin(dist_matrix, axis=0)
            print(
                np.sum(pcd_selection),
                np.sum(point_sel),
                i_closest_arr.shape,
                dist_matrix.shape,
            )
            qps_coeffs = self.local_qps_coeffs[pcd_selection, :][i_closest_arr, :]
            local_pcds = self.local_pcds[pcd_selection][i_closest_arr]
            pnt, normals = vec_qps(point_arr, qps_coeffs, local_pcds)
            z_val[pnt_retrieve] = pnt[:, 2]
            z_norm[pnt_retrieve] = normals

        pcd_sel = (self.global_pcd[:, 0] < 0) & (self.global_pcd[:, 1] < 0)
        pnt_sel = (point_arr[:, 0] < 0) & (point_arr[:, 1] < 0)
        execute_selection(pcd_sel, pnt_sel)

        # dist_matrix = distance_matrix(self.global_pcd[:, 0:2], point_arr)
        # i_closest_arr = np.argmin(dist_matrix, axis=-1)
        # qps_coeffs = self.local_qps_coeffs[i_closest_arr]
        # local_pcds = self.local_pcds[i_closest_arr]
        # pnt, normals = vec_qps(point_arr, qps_coeffs, local_pcds)

        return z_val, z_norm

    def plot_z_val_and_z_cos(
        self, colormap="viridis", zlim=None, dpi=300, display=False
    ):
        fig, ax = create_figure_and_axes(None, [1, 2])
        simple_imshow_map_plot(
            ax[0],
            fig,
            self.current_u_axis,
            self.current_v_axis,
            self.current_z_val,
            "Z value",
            colormap,
            zlim,
            z_label="Z value [m]",
            transpose=True,
        )
        simple_imshow_map_plot(
            ax[1],
            fig,
            self.current_u_axis,
            self.current_v_axis,
            self.current_z_cos,
            "Z Cosine",
            colormap,
            zlim,
            z_label="Z cosine []",
            transpose=True,
        )
        close_figure(
            fig,
            "Gridded surface and cosine of surface angle to Z axis",
            "zval_zcos.png",
            dpi,
            display,
        )
        return

    def compute_gridded_z_val_and_z_cos(
        self,
        u_axis,
        v_axis,
        mask,
        gridding_engine="2D regrid",
        light=(0, 0, -1),
        vectorized=True,
    ):
        light = np.array(light)

        u_mesh, v_mesh = create_coordinate_images(u_axis, v_axis)
        uv_idx_grid = create_2d_array_reconstruction_array(u_axis, v_axis, mask)

        uv_points = np.empty_like(uv_idx_grid)
        uv_points[:, 0] = u_mesh[mask]
        uv_points[:, 1] = v_mesh[mask]

        if vectorized:
            z_val, z_norm = local_qps_image_jit(
                self.global_pcd, self.local_qps_coeffs, self.local_pcds, uv_points
            )
        else:
            z_val = np.empty([uv_points.shape[0]])
            z_norm = np.empty([uv_points.shape[0], 3])

            for ip, point in enumerate(uv_points):
                z_val[ip], z_norm[ip] = self.compute_z_val_and_z_cos(point)

        z_angle = (generalized_dot(z_norm, light)) / (
            generalized_norm(z_norm) * generalized_norm(light)
        )

        if gridding_engine == "2D regrid":
            z_val_grid = regrid_data_onto_2d_grid(u_axis, v_axis, z_val, uv_idx_grid)
            z_cos_grid = regrid_data_onto_2d_grid(u_axis, v_axis, z_angle, uv_idx_grid)
        else:
            raise Exception("only 2D regrid available now")

        self.current_z_val = z_val_grid
        self.current_z_cos = z_cos_grid
        self.current_u_axis = u_axis
        self.current_v_axis = v_axis

        return z_val_grid, z_cos_grid

    def downgrid_high_resolution_z_val_and_z_cos(self, u_axis, v_axis, gridding_engine):
        return

    def compute_high_resolution_z_val_and_z_cos(self, x_resolution, y_resolution):
        return

    @classmethod
    def from_pickle(cls, filename):
        with open(filename, "rb") as pickled_file:
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
        with open(filename, "wb") as pickle_file:
            # noinspection PyTypeChecker
            pickle.dump(self, pickle_file)
