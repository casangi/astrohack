import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import griddata, LinearNDInterpolator
from numba import njit
import xarray as xr
import toolviper.utils.logger as logger
import time

from astrohack.utils import convert_unit, twopi, data_statistics, gauss_elimination
from astrohack.visualization.plot_tools import get_proper_color_map, create_figure_and_axes, well_positioned_colorbar, \
    close_figure, compute_extent, scatter_plot

nanvec3d = np.array([np.nan, np.nan, np.nan])
intblankval = -1000
return_line = '\033[F'


def numpy_size(array):
    units = ['B', 'kB', 'MB', 'GB']
    memsize = array.itemsize * array.size
    iu = 0
    while memsize > 1024:
        memsize /= 1024
        iu += 1
    print(f'Image size: {memsize:.2f} {units[iu]}')

def imshow_plot(ax, fig, title, x_axis, y_axis, gridded_data, cmap, minmax, fsize=5, flipxy=True):
    if flipxy:
        plotting_data = np.transpose(gridded_data)
    else:
        plotting_data = gridded_data

    ax.set_title(title, size=1.5 * fsize)
    extent = compute_extent(x_axis, y_axis, margin=0.1)
    im = ax.imshow(plotting_data, cmap=cmap, extent=extent, interpolation="nearest", vmin=minmax[0], vmax=minmax[1],
                   origin='lower')
    well_positioned_colorbar(ax, fig, im, "Phase [deg?]")
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xlabel("X axis [m]")
    ax.set_ylabel("Y axis [m]")


def inner_product_2d(vec_a, vec_b, keep_shape=True):
    """
    This routine expects that vec a and vec b are of the same shape: [n, 3]
    Args:
        vec_a: first vector array
        vec_b: second vector array
        keep_shape: output has the same shape as inputs (easier for subsequent broadcasting)

    Returns:
        the inner product of the vectors in the arrays
    """
    if keep_shape:
        inner = np.empty_like(vec_a)
        inner[:, 0] = np.sum(vec_a * vec_b, axis=1)
        for ipos in range(1, inner.shape[1]):
            inner[:, ipos] = inner[:, 0]
        return inner
    else:
        return np.sum(vec_a * vec_b, axis=1)


@njit(cache=False, nogil=True)
def inner_product_2d_jit(vec_a, vec_b):
    """
    This routine expects that vec a and vec b are of the same shape: [n, 3]
    Args:
        vec_a: first vector array
        vec_b: second vector array

    Returns:
        the inner product of the vectors in the arrays
    """
    inner = np.empty_like(vec_a)
    inner[:, 0] = np.sum(vec_a * vec_b, axis=1)
    for ipos in range(1, inner.shape[1]):
        inner[:, ipos] = inner[:, 0]
    return inner


@njit(cache=False, nogil=True)
def inner_product_1d_jit(vec_a, vec_b):
    return np.sum(vec_a * vec_b, axis=1)


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
def reflect_on_surface(light, normal):
    return light - 2 * np.dot(light, normal) * normal


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


@njit(cache=True, nogil=True)
def cropped_secondary_mesh(pr_reflec, pr_pnt, sc_pnt, sc_cropped_mesh, sc_cropped_mesh_norm, sc_n_triangles):
    sc_reflec = np.empty_like(pr_reflec)
    sc_reflec_pnt = np.empty_like(pr_reflec)
    sc_reflec_triangle = np.empty(pr_reflec.shape[0])

    niter = pr_pnt.shape[0]
    for ipnt in range(niter):
        n_tri = sc_n_triangles[ipnt]
        if n_tri == 0:
            sc_reflec[ipnt] = nanvec3d
            sc_reflec_pnt[ipnt] = nanvec3d
            sc_reflec_triangle[ipnt] = np.nan
        else:
            pr_point = pr_pnt[ipnt]
            pr_reflection = pr_reflec[ipnt]
            mesh_section = sc_cropped_mesh[ipnt, 0:n_tri]
            itriangle, sc_point = jitted_triangle_find(pr_point, pr_reflection, mesh_section, sc_pnt)
            if itriangle == intblankval:
                sc_reflec[ipnt] = nanvec3d
                sc_reflec_pnt[ipnt] = nanvec3d
                sc_reflec_triangle[ipnt] = np.nan
            else:
                sc_reflec_triangle[ipnt] = itriangle
                sc_reflec_pnt[ipnt] = sc_point
                sc_reflec[ipnt] = reflect_on_surface(pr_reflection, sc_cropped_mesh_norm[ipnt, itriangle])

    return sc_reflec, sc_reflec_pnt, sc_reflec_triangle


# @njit(cache=True, nogil=True)
def compute_distances(array_of_vectors, point):
    # assumes that at least array_of_vectors is of shape [:, 3]
    # Point can be an array or a single point
    diff = array_of_vectors-point
    return np.sqrt(np.sum(diff**2, axis=1))


@njit(cache=True, nogil=True)
def compute_distances2(array_of_vectors, point):
    # assumes that at least array_of_vectors is of shape [:, 3]
    # Point can be an array or a single point
    diff = array_of_vectors-point
    return np.sqrt(np.sum(diff**2, axis=1))


@njit(cache=True, nogil=True)
def simple_axis(user_array, resolution, margin=0.05):
    mini, maxi = np.min(user_array), np.max(user_array)
    ax_range = maxi-mini
    pad = margin*ax_range
    if pad < np.abs(resolution):
        pad = np.abs(resolution)
    mini -= pad
    maxi += pad
    npnt = int(np.ceil((maxi - mini) / resolution))
    axis_array = np.arange(npnt + 1)
    axis_array = resolution * axis_array
    axis_array = axis_array + mini + resolution / 2
    return axis_array


class Axis:
    def __init__(self, user_array, resolution):
        mini, maxi = np.min(user_array), np.max(user_array)
        npnt = int(np.ceil((maxi - mini) / resolution))
        axis_array = np.arange(npnt + 1)
        axis_array = resolution * axis_array
        axis_array = axis_array + mini + resolution / 2
        self.np = npnt
        self.res = resolution
        self.mini = mini
        self.maxi = maxi
        self.array = axis_array

    def idx_and_frac(self, coor):
        f_idx = (coor - self.array[0]) / self.res
        i_idx = round(f_idx)
        if i_idx > f_idx:
            idx = [i_idx - 1, i_idx]
            frac = 1 - (i_idx - f_idx)
            fracs = [frac, 1 - frac]
        elif i_idx < f_idx:
            idx = [i_idx, i_idx + 1]
            frac = f_idx - i_idx
            fracs = [frac, 1 - frac]
        else:
            idx = [i_idx]
            fracs = [1.0]

        if len(idx) > 1:
            if idx[0] < 0:
                idx = idx[1:]
                fracs = fracs[1:]
            elif idx[1] > self.np - 1:
                idx = idx[:1]
                fracs = fracs[:1]

        return idx, fracs


class NgvlaRayTracer:

    def __init__(self, wavelength=0.007, incident_light=(0,0,-1),
                 focus_location=(-1.136634465810194, 0, -0.331821128650557),
                 horn_orientation=(0, 0, 1), horn_length=0, horn_diameter=1000, horn_position=(0,0,0),
                 phase_offset=0.0):
        """

        Args:
            wavelength: Light wavelength in meters
            incident_light: Unitary vector describing incident light direction
            focus_location: Focus location relative in mesh coordinate system
            horn_orientation: Unitary vector describing horn orientation
            horn_length: Length of the horn in meters
            horn_diameter: Diameter of the horn in meters
            horn_position: Position of the horn relative to focus in meters
            phase_offset: Phase offset in radians
        """
        self.pr_pnt = self.pr_norm = None
        self.sc_pnt = self.sc_norm = None
        self.focus_offset = np.array(focus_location)
        self.horn_orientation = np.array(horn_orientation)
        self.horn_diameter = horn_diameter
        self.horn_length = horn_length
        self.horn_position = np.array(horn_position)
        self.incident_light = np.array(incident_light)
        self.wavelength = wavelength
        self.phase_offset = phase_offset

        self.pr_reflec = None
        self.sc_reflec = None
        self.sc_reflec_pnt = None
        self.sc_reflec_dist = None
        self.sc_reflec_triangle = None
        self.pr_mesh = None
        self.pr_mesh_norm = None
        self.sc_mesh = None
        self.sc_mesh_norm = None
        self.sc_cropped_mesh = None
        self.sc_cropped_mesh_norm = None
        self.sc_n_triangles = None
        self.horn_intersect = None
        self.full_light_path = None
        self.phase = None
        self.horn_distance = None
        self.n_wavelength = None

    def _shift_to_focus_origin(self):
        # Both dishes are in the same coordinates but this is not the
        # MR reference frame, but its axes are oriented the same
        # This is the expected coordinate of the focus on the MR.
        # focus = [-1.136634465810194, 0, -0.331821128650557]
        # Translation simple shift across the axes as everything is in
        # meters
        for iax, axfocus in enumerate(self.focus_offset):
            if axfocus != 0:
                self.pr_pnt[iax] -= axfocus
                self.sc_pnt[iax] -= axfocus

    def primary_reflection(self):
        light = np.zeros_like(self.pr_pnt)
        light[:] = np.array(self.incident_light)
        self.pr_reflec = light - 2 * inner_product_2d(light, self.pr_norm) * self.pr_norm

    def secondary_reflection_obsolete(self):
        self.sc_reflec = np.empty_like(self.pr_reflec)
        self.sc_reflec_pnt = np.empty_like(self.pr_reflec)
        print()
        for it, point, in enumerate(self.pr_pnt):
            print(f'\033[F{100 * it / self.pr_pnt.shape[0]:.2f}%')
            pnt_reflec = self.pr_reflec[it]
            pnt_diff = point - self.sc_pnt
            dist_vec = pnt_diff - inner_product_2d(pnt_diff, pnt_reflec) * pnt_reflec
            dist_matrix = np.sqrt(inner_product_2d(dist_vec, dist_vec, keep_shape=False))
            isec_loc = np.argmin(dist_matrix)
            self.sc_reflec_pnt[it] = self.sc_pnt[isec_loc]
            self.sc_reflec[it] = pnt_reflec - 2 * np.inner(pnt_reflec, self.sc_norm[isec_loc]) * self.sc_norm[isec_loc]
        print()

    def _grid_with_griddata(self, data_array, resolution, label):
        x_pnt = self.pr_pnt[:, 0]
        y_pnt = self.pr_pnt[:, 1]
        x_axis = Axis(x_pnt, resolution)
        y_axis = Axis(y_pnt, resolution)
        x_mesh, y_mesh = np.meshgrid(x_axis.array, y_axis.array)
        selection = np.isfinite(data_array)
        if np.sum(selection) == 0:
            logger.warning(f'No data to display for {label}')
            return None
        gridded_array = griddata((x_pnt[selection], y_pnt[selection]),
                                 data_array[selection], (x_mesh, y_mesh), 'linear')

        return gridded_array, x_axis, y_axis

    def _grid_with_lndi(self, data_array, resolution, label):
        x_pnt = self.pr_pnt[:, 0]
        y_pnt = self.pr_pnt[:, 1]
        x_axis = Axis(x_pnt, resolution)
        y_axis = Axis(y_pnt, resolution)
        x_mesh, y_mesh = np.meshgrid(x_axis.array, y_axis.array)
        selection = np.isfinite(data_array)
        if np.sum(selection) == 0:
            logger.warning(f'No data to display for {label}')
            return None
        interp = LinearNDInterpolator(list(zip(x_pnt, y_pnt)), data_array)
        gridded_array = interp(x_mesh, y_mesh)
        return gridded_array, x_axis, y_axis

    def _plot_map(self, data_array, prog_res, title, filename, colormap, zlim, fsize=5):
        grid = self._grid_with_griddata(data_array, prog_res, title)

        if grid is None:
            return
        else:
            gridded_data, x_axis, y_axis = grid

        if zlim is None:
            minmax = [np.nanmin(gridded_data), np.nanmax(gridded_data)]
        else:
            minmax = zlim

        fig, ax = create_figure_and_axes([10, 8], [1, 1])
        cmap = get_proper_color_map(colormap)

        ax.set_title(title, size=1.5 * fsize)
        extent = compute_extent(x_axis.array, y_axis.array, margin=0.1)
        im = ax.imshow(gridded_data, cmap=cmap, extent=extent, interpolation="nearest", vmin=minmax[0], vmax=minmax[1])
        well_positioned_colorbar(ax, fig, im, "Z Scale")
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")

        close_figure(fig, '', filename, 300, False)

    def compare_phase_gridding(self, phase_unit='deg', resolution=5, length_unit='cm', colormap='RdBu_r', zlim=None):
        if self.phase is None:
            raise Exception("Can't plot phase if phase is not present...")
        resolution *= convert_unit(length_unit, 'm', 'length')
        phase_fac = convert_unit('rad', phase_unit, 'trigonometric')

        start = time.time()
        griddata_phase, x_axis, y_axis = self._grid_with_griddata(self.phase, resolution, 'phase')
        stop = time.time()
        print(f'Gridding with griddata took {stop - start} seconds')

        start = time.time()
        lndi_phase, x_axis, y_axis = self._grid_with_lndi(self.phase, resolution, 'phase')
        stop = time.time()
        print(f'Gridding with LNDI took {stop - start} seconds')

        diff = lndi_phase - griddata_phase
        print('Statistics of the difference')
        print(data_statistics(diff))

        if zlim is None:
            minmax = [-np.pi * phase_fac, np.pi * phase_fac]
        elif zlim == 'minmax':
            minmax = [np.nanmin(griddata_phase), np.nanmax(griddata_phase)]
        else:
            minmax = zlim


        fig, ax = create_figure_and_axes([16, 8], [1, 3])
        cmap = get_proper_color_map(colormap)
        imshow_plot(ax[0], fig, 'Griddata phase map', x_axis.array, y_axis.array, griddata_phase*phase_fac, cmap, minmax, fsize=10)
        imshow_plot(ax[1], fig, 'LNDI phase map', x_axis.array, y_axis.array, lndi_phase*phase_fac, cmap, minmax, fsize=10)
        imshow_plot(ax[2], fig, 'Difference phase map', x_axis.array, y_axis.array, diff*phase_fac, cmap, minmax, fsize=10)
        close_figure(fig, 'Phase gridding comparison', f'phase_comparison_{resolution:.2}.png', 300, False)

    def _select_data_for_plot(self, data_type):
        zlim = None
        if data_type == 'x reflec pnt':
            data_array = self.sc_reflec_pnt[:, 0]
            title = f'X coordinate of point touched on secondary'
        elif data_type == 'y reflec pnt':
            data_array = self.sc_reflec_pnt[:, 1]
            title = f'Y coordinate of point touched on secondary'
        elif data_type == 'z reflec pnt':
            data_array = self.sc_reflec_pnt[:, 2]
            title = f'Z coordinate of point touched on secondary'
        elif data_type == 'x reflec dir':
            data_array = self.sc_reflec[:, 0]
            title = f'X component of reflection on secondary'
        elif data_type == 'y reflec dir':
            data_array = self.sc_reflec[:, 1]
            title = f'Y component of reflection on secondary'
        elif data_type == 'z reflec dir':
            data_array = self.sc_reflec[:, 2]
            title = f'Z component of reflection on secondary'
        elif data_type == 'x prim normal':
            data_array = self.pr_norm[:, 0]
            title = f'X component of primary mirror normal'
        elif data_type == 'y prim normal':
            data_array = self.pr_norm[:, 1]
            title = f'Y component of primary mirror normal'
        elif data_type == 'z prim normal':
            data_array = self.pr_norm[:, 2]
            title = f'Z component of primary mirror normal'
        elif data_type == 'reflec dist':
            data_array = self.sc_reflec_dist
            title = f'Distance to the point of reflection on the secondary'
            zlim = [0, 0.05]
        elif data_type == 'light path':
            data_array = self.full_light_path
            title = f'Full light path'
        elif data_type == 'phase':
            data_array = self.phase
            title = f'Phase at detection'
            zlim = [-np.pi, np.pi]
        elif data_type == 'horn distance':
            data_array = self.horn_distance
            title = f'Distance to horn center'
            zlim = [1, 2]
        else:
            raise Exception(f'Unrecognized data type {data_type}')
        return data_array, title, zlim

    def plot_simple_2d(self, data_types, rootname, resolution, resolution_unit, colormap='viridis'):
        prog_res = convert_unit(resolution_unit, 'm', 'length') * resolution
        for data_type in data_types:
            data_array, title, zlim = self._select_data_for_plot(data_type)
            filename = f"{rootname}-gridded-{data_type.replace(' ', '-')}.png"
            self._plot_map(data_array, prog_res, title, filename, colormap, zlim)

    def save_to_zarr(self, filename):
        xds = xr.Dataset()

        for key, item in vars(self).items():
            try:
                if isinstance(item, np.ndarray):
                    if key in ['sc_pnt', 'sc_norm']:
                        xds[key] = xr.DataArray(item, dims=['sc_pnt', 'xyz'])
                    elif key in ['pr_mesh', 'pr_mesh_norm']:
                        xds[key] = xr.DataArray(item, dims=['pr_tri', 'tri_corners'])
                    elif key in ['sc_mesh', 'sc_mesh_norm']:
                        xds[key] = xr.DataArray(item, dims=['sc_tri', 'tri_corners'])
                    elif key in ['focus_offset', 'horn_orientation', 'horn_position', 'incident_light']:
                        xds.attrs[key] = item
                    elif len(item.shape) == 2:
                        xds[key] = xr.DataArray(item, dims=['pr_pnt', 'xyz'])
                    elif len(item.shape) == 1:
                        xds[key] = xr.DataArray(item, dims=['pr_pnt'])
                    elif key in ['sc_cropped_mesh']:
                        xds[key] = xr.DataArray(item, dims=['pr_pnt', 'crop_tri', 'tri_corners'])
                    elif key in ['sc_cropped_mesh_norm']:
                        xds[key] = xr.DataArray(item, dims=['pr_pnt', 'crop_tri', 'xyz'])
                    else:
                        raise Exception(f"Don't know what to do with {key}")
                elif item is None:
                    pass
                else:
                    xds.attrs[key] = item
            except ValueError:
                msg = f'{key} => {item.shape}'
                raise Exception(msg)
        print(xds)
        xds.to_zarr(filename, mode='w')

    def reread(self, xds_name, fetch_attributes=False):
        xds = xr.open_zarr(xds_name)
        if fetch_attributes:
            for key, item in xds.attrs.items():
                self.__setattr__(key, item)
        for key, item in xds.items():
            self.__setattr__(str(key), item.values)
        self.pr_pnt = xds.pr_pnt.values
        self.sc_pnt = xds.sc_pnt.values

    def read_from_zarr_mesh(self, mesh_file):
        in_xds = xr.open_zarr(mesh_file)
        self.pr_pnt = in_xds['primary_point_cloud'].values
        self.pr_norm = in_xds['primary_pcd_normals'].values
        self.pr_mesh = in_xds['primary_mesh'].values
        self.pr_mesh_norm = in_xds['primary_mesh_normals'].values

        self.sc_pnt = in_xds['secondary_point_cloud'].values
        self.sc_norm = in_xds['secondary_pcd_normals'].values
        self.sc_mesh = in_xds['secondary_mesh'].values
        self.sc_mesh_norm = in_xds['secondary_mesh_normals'].values

        self._shift_to_focus_origin()

    def __repr__(self):
        outstr = 'Ray tracer contents:\n'
        for key, item in vars(self).items():
            outstr += f'{key:15s} = {item}\n'
        return outstr

    def _find_triangle_on_secondary(self, pr_point, reflection):
        """
        This returns the triangle index and the point in the triangle.
        """
        return jitted_triangle_find(pr_point, reflection, self.sc_mesh, self.sc_pnt)

    def _find_triangle_on_cropped_mesh(self, pr_point, reflection, mesh_section):
        return jitted_triangle_find(pr_point, reflection, mesh_section, self.sc_pnt)

    def secondary_reflection_on_mesh_brute_force(self, short_circuit=False):
        self.sc_reflec = np.empty_like(self.pr_reflec)
        self.sc_reflec_pnt = np.empty_like(self.pr_reflec)
        self.sc_reflec_triangle = np.empty(self.pr_reflec.shape[0])
        print()

        if short_circuit:
            niter = 100
        else:
            niter = self.pr_pnt.shape[0]
        for ipnt in range(niter):
            pr_point = self.pr_pnt[ipnt]
            pr_reflection = self.pr_reflec[ipnt]
            itriangle, sc_point = self._find_triangle_on_secondary(pr_point, pr_reflection)
            self.sc_reflec_triangle[ipnt] = itriangle
            self.sc_reflec_pnt[ipnt] = sc_point
            if np.isnan(itriangle):
                self.sc_reflec[ipnt] = nanvec3d
            else:
                self.sc_reflec[ipnt] = reflect_on_surface(pr_reflection, self.sc_mesh_norm[itriangle])
            print(f'\033[FMesh reflections: {100 * ipnt / niter:.2f}%')

        self.sc_reflec_triangle = np.where(np.abs(self.sc_reflec_triangle-intblankval)<1e-7, np.nan,
                                           self.sc_reflec_triangle)

    def cropped_reflec_jit(self):
        self.sc_reflec, self.sc_reflec_pnt, self.sc_reflec_triangle = \
            cropped_secondary_mesh(self.pr_reflec, self.pr_pnt, self.sc_pnt, self.sc_cropped_mesh,
                                   self.sc_cropped_mesh_norm, self.sc_n_triangles)

    def secondary_to_horn(self, epsilon=1e-7):
        # Horn orientation must be unitary
        horn_mouth_center = self.horn_length * self.horn_orientation
        sec_to_horn = horn_mouth_center - self.sc_reflec_pnt
        dot_horn_plane = np.dot(sec_to_horn, self.horn_orientation)
        line_par = np.where(np.abs(dot_horn_plane) < epsilon, np.nan,
                            dot_horn_plane/np.dot(self.sc_reflec, self.horn_orientation))
        intersect_point = self.sc_reflec_pnt + line_par[:,np.newaxis]*self.sc_reflec
        dist_horn_mouth = compute_distances(intersect_point, horn_mouth_center)
        self.horn_distance = dist_horn_mouth
        self.horn_intersect = np.where(dist_horn_mouth[:,np.newaxis] <= self.horn_diameter, intersect_point, np.nan)
        return

    def compute_full_light_path(self, show_stats=False):
        pr_z_val = self.pr_pnt[:, 2]
        pr_z_max = np.max(pr_z_val)
        light_z = self.incident_light[2]
        # From the plane defined by the leading edge of the primary reflector to primary reflector point along light ray
        zeroth_distance = (pr_z_val - pr_z_max) / light_z
        # From point in the primary to point in the secondary along light ray
        first_distance = compute_distances(self.pr_pnt, self.sc_reflec_pnt)
        # From point in the secondary to horn mouth
        second_distance = compute_distances(self.sc_reflec_pnt, self.horn_intersect)
        # from horn mouth to receptor inside horn
        third_distance = compute_distances(self.horn_intersect, self.horn_position)

        self.full_light_path = zeroth_distance + first_distance + second_distance + third_distance
        self.n_wavelength = self.full_light_path/self.wavelength
        floor_n_wave = np.floor(self.n_wavelength)
        self.phase = (self.n_wavelength-floor_n_wave) * twopi - np.pi
        self.phase += self.phase_offset


        if show_stats:
            print('first')
            print(data_statistics(first_distance))
            print('second')
            print(data_statistics(second_distance))
            print('third')
            print(data_statistics(third_distance))
            print('full light path')
            print(data_statistics(self.full_light_path))
            print('phase')
            print(data_statistics(self.phase))
            print('horn distance')
            print(data_statistics(self.horn_distance))

        return

    def triangle_area(self):
        for it, triangle in enumerate(self.sc_mesh):
            va = self.sc_pnt[int(triangle[0])]
            vb = self.sc_pnt[int(triangle[1])]
            vc = self.sc_pnt[int(triangle[2])]
            vba = vb - va
            vbc = vb - vc
            cross = np.cross(vba, vbc)
            print(it, '=', np.sqrt(np.inner(cross, cross)) / 2)

    def _plot_no_gridding(self, data_array, title, filename, zlim, colormap='viridis', resolution=0.1):
        cmap = get_proper_color_map(colormap)
        fig, ax = create_figure_and_axes([10, 8], [1, 1])
        x_pnt = self.pr_pnt[:, 0]
        y_pnt = self.pr_pnt[:, 1]
        x_axis = Axis(x_pnt, resolution)
        y_axis = Axis(y_pnt, resolution)
        extent = compute_extent(x_axis.array, y_axis.array, margin=0.1)

        if zlim is None:
            vmin = np.nanmin(data_array)
            vmax = np.nanmax(data_array)
        else:
            vmin = zlim[0]
            vmax = zlim[1]
        norm = Normalize(vmin=vmin, vmax=vmax)

        notnan = ~ np.isnan(data_array)
        colors = cmap(norm(data_array[notnan]))
        ax.scatter(x_pnt[notnan], y_pnt[notnan], color=colors, s=resolution)

        ax.set_title(title)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        well_positioned_colorbar(ax, fig, cmap, "Z Scale")
        close_figure(fig, '', filename, 300, False)

    def plot_no_grid(self, data_types, rootname, colormap='viridis'):
        for data_type in data_types:
            data_array, title, zlim = self._select_data_for_plot(data_type)
            filename = f"{rootname}-no-grid-{data_type.replace(' ', '-')}.png"
            self._plot_no_gridding(data_array, title, filename, zlim)

    def crop_secondary_mesh(self, max_distances, i_sel=None):
        sc_cropped_mesh = []
        sc_cropped_mesh_norm = []
        n_pnt = self.pr_pnt.shape[0]
        sc_n_triang = np.full(n_pnt, 0, dtype=int)

        if i_sel is None:
            loop_list = range(n_pnt)
        else:
            if isinstance(i_sel, list):
                loop_list = i_sel
            else:
                loop_list = [i_sel]

        selec_dist = 0.1
        sel_large_dists = max_distances>selec_dist

        for ipnt in loop_list:

            if not np.isfinite(max_distances[ipnt]):
                sc_n_triang[ipnt] = 0
                sc_cropped_mesh.append([])
                sc_cropped_mesh_norm.append([])
                continue

            sc_point = self.sc_reflec_pnt[ipnt]
            max_dist = max_distances[ipnt]

            pnt_distances = np.sqrt(np.sum((self.sc_pnt - sc_point)**2, axis=1))
            isc_pnt = np.arange(pnt_distances.shape[0])

            sel_dist = pnt_distances <= max_dist
            multiplier = 1.1

            while np.sum(sel_dist) < 3:
                # if i_sel is not None:
                #     print(ipnt, np.sum(sel_dist), max_dist)
                #     print()
                max_dist *= multiplier
                sel_dist = pnt_distances <= max_dist

            selected_triangles = []
            triangle_normals = []
            for point in isc_pnt[sel_dist]:
                for ix in range(3):
                    selec = self.sc_mesh[:, ix] == point
                    selected_triangles.extend(self.sc_mesh[selec])
                    triangle_normals.extend(self.sc_mesh_norm[selec])

            sc_n_triang[ipnt] = len(selected_triangles)
            # print('triangulitos', sc_n_triang[ipnt], len(triangle_normals))
            # print()
            sc_cropped_mesh.append(np.array(selected_triangles))
            sc_cropped_mesh_norm.append(np.array(triangle_normals))

            print(f'{return_line}Cropping: {100 * ipnt / self.pr_pnt.shape[0]:.2f}%      ')

        max_length = np.max(sc_n_triang)
        imax = np.argmax(sc_n_triang)
        print(f'Max lenght is {max_length} Triangles at {imax} {sc_n_triang[imax]}')

        self.sc_n_triangles = sc_n_triang
        cropped_shape = [n_pnt, max_length, 3]
        self.sc_cropped_mesh = np.full(cropped_shape, np.nan)
        self.sc_cropped_mesh_norm = np.full(cropped_shape, np.nan)
        for ipnt in range(n_pnt):
            n_triang = sc_n_triang[ipnt]
            if n_triang > 0:
                self.sc_cropped_mesh[ipnt, 0:n_triang] = sc_cropped_mesh[ipnt]
                self.sc_cropped_mesh_norm[ipnt, 0:n_triang] = sc_cropped_mesh_norm[ipnt]

        numpy_size(self.sc_cropped_mesh)

    def find_focus(self, ipnt):
        p0x, p0y = self.sc_reflec_pnt[ipnt, 0:2]
        l0x, l0y = self.sc_reflec[ipnt, 0:2]

        matrix = np.ndarray([2, 2])
        vector = np.ndarray(2)

        crossings = np.empty_like(self.sc_reflec_pnt)

        for jpnt in range(self.pr_pnt.shape[0]):
            if jpnt == ipnt:
                crossings[jpnt] = nanvec3d
            elif np.all(np.isfinite(self.sc_reflec[jpnt])):
                lpntx, lpnty = self.sc_reflec_pnt[jpnt, 0:2]
                ppntx, ppnty = self.sc_reflec[jpnt, 0:2]
                matrix[0, :] = [l0x, -lpntx]
                matrix[1, :] = [l0y, -lpnty]
                vector[:] = [ppntx-p0x, ppnty-p0y]

                try:
                    solution = gauss_elimination(matrix, vector)
                    crossings[jpnt] = self.sc_reflec_pnt[ipnt] + solution[0]*self.sc_reflec[jpnt]
                except KeyError:
                    crossings[jpnt] = nanvec3d
            else:
                crossings[jpnt] = nanvec3d

        # Selecting crossings within 2 meters of the origin which is supposed to be the focus
        selection_min = self.horn_distance < 1.6
        selection_max = self.horn_distance > 1.0
        selection = selection_min & selection_max
        crossings = crossings[selection]

        print('x coord')
        print(data_statistics(crossings[:, 0]))
        print('y coord')
        print(data_statistics(crossings[:, 1]))
        print('z coord')
        print(data_statistics(crossings[:, 2]))

    def grid_phase_image(self, resolution):
        griddata_phase, x_axis, y_axis = self._grid_with_griddata(self.phase, resolution, 'phase')
        return griddata_phase, x_axis, y_axis


def ngvla_rt_pipeline(cropped_mesh_zarr_file, wavelength=0.007, incident_light=(0,0,-1),
                      focus_location=(-1.136634465810194, 0, -0.331821128650557),
                      horn_orientation=(0, 0, 1), horn_length=0, horn_diameter=1000, horn_position=(0,0,0),
                      phase_offset=0.0, resolution=0.1):

    rtobj = NgvlaRayTracer(wavelength=wavelength, incident_light=incident_light, focus_location=focus_location,
                           horn_orientation=horn_orientation, horn_length=horn_length, horn_diameter=horn_diameter,
                           horn_position=horn_position, phase_offset=phase_offset)
    rtobj.reread(cropped_mesh_zarr_file)
    rtobj.primary_reflection()
    rtobj.cropped_reflec_jit()
    rtobj.secondary_to_horn()
    rtobj.compute_full_light_path(show_stats=False)

    return rtobj.grid_phase_image(resolution)


def full_proc_pipeline(cropped_mesh_zarr_file, wavelength=0.007, incident_light=(0, 0, -1),
                       focus_location=(-1.136634465810194, 0, -0.331821128650557),
                       horn_orientation=(0, 0, 1), horn_length=0, horn_diameter=1000, horn_position=(0, 0, 0),
                       phase_offset=0.0, resolution=0.1, epsilon=1e-7):

    # Converting List/Tuples to np arrays
    incident_light = np.array(incident_light)
    horn_orientation = np.array(horn_orientation)
    horn_position = np.array(horn_position)

    # Opening XDS
    in_xds = xr.open_zarr(cropped_mesh_zarr_file)

    # Primary reflections
    pr_pnt = in_xds['pr_pnt'].values
    pr_norm = in_xds['pr_norm'].values
    light = np.zeros_like(pr_pnt)
    light[:] = incident_light
    pr_reflec = light - 2 * inner_product_2d(light, pr_norm) * pr_norm

    # Secondary reflections
    sc_pnt = in_xds['sc_pnt'].values
    sc_cropped_mesh = in_xds['sc_cropped_mesh'].values
    sc_cropped_mesh_norm = in_xds['sc_cropped_mesh_norm'].values
    sc_n_triangles = in_xds['sc_n_triangles'].values
    sc_reflec, sc_reflec_pnt, sc_reflec_triangle = cropped_secondary_mesh(pr_reflec, pr_pnt, sc_pnt, sc_cropped_mesh,
                                                                          sc_cropped_mesh_norm, sc_n_triangles)

    # Secondary to horn
    horn_mouth_center = horn_length * horn_orientation
    sec_to_horn = horn_mouth_center - sc_reflec_pnt
    dot_horn_plane = np.dot(sec_to_horn, horn_orientation)
    line_par = np.where(np.abs(dot_horn_plane) < epsilon, np.nan, dot_horn_plane/np.dot(sc_reflec, horn_orientation))
    intersect_point = sc_reflec_pnt + line_par[:, np.newaxis]*sc_reflec
    dist_horn_mouth = compute_distances(intersect_point, horn_mouth_center)
    horn_intersect = np.where(dist_horn_mouth[:, np.newaxis] <= horn_diameter, intersect_point, np.nan)

    # Compute Full light path
    pr_z_val = pr_pnt[:, 2]
    pr_z_max = np.max(pr_z_val)
    light_z = incident_light[2]
    # From the plane defined by the leading edge of the primary reflector to primary reflector point along light ray
    zeroth_distance = (pr_z_val - pr_z_max) / light_z
    # From point in the primary to point in the secondary along light ray
    first_distance = compute_distances(pr_pnt, sc_reflec_pnt)
    # From point in the secondary to horn mouth
    second_distance = compute_distances(sc_reflec_pnt, horn_intersect)
    # from horn mouth to receptor inside horn
    third_distance = compute_distances(horn_intersect, horn_position)
    full_light_path = zeroth_distance + first_distance + second_distance + third_distance

    # Compute phase from light path
    n_wavelength = full_light_path/wavelength
    floor_n_wave = np.floor(n_wavelength)
    phase = (n_wavelength-floor_n_wave) * twopi - np.pi
    phase += phase_offset

    # Grid Phase
    x_pnt = pr_pnt[:, 0]
    y_pnt = pr_pnt[:, 1]
    x_axis = Axis(x_pnt, resolution)
    y_axis = Axis(y_pnt, resolution)
    x_mesh, y_mesh = np.meshgrid(x_axis.array, y_axis.array)
    selection = np.isfinite(phase)
    if np.sum(selection) == 0:
        logger.warning(f'Phase is empty')
        return None
    gridded_phase = griddata((x_pnt[selection], y_pnt[selection]), phase[selection], (x_mesh, y_mesh), 'linear')

    return gridded_phase, x_axis, y_axis


def pipeline_full_jit(cropped_mesh_zarr_file, wavelength=0.007, incident_light=(0, 0, -1),
                      focus_location=(-1.136634465810194, 0, -0.331821128650557),
                      horn_orientation=(0, 0, 1), horn_length=0, horn_diameter=1000, horn_position=(0, 0, 0),
                      phase_offset=0.0, resolution=0.1, epsilon=1e-7):

    # Converting List/Tuples to np arrays
    incident_light = np.array(incident_light, dtype=float)
    horn_orientation = np.array(horn_orientation, dtype=float)
    horn_position = np.array(horn_position, dtype=float)

    # Opening XDS
    in_xds = xr.open_zarr(cropped_mesh_zarr_file)
    pr_pnt = in_xds['pr_pnt'].values
    pr_norm = in_xds['pr_norm'].values
    sc_pnt = in_xds['sc_pnt'].values
    sc_cropped_mesh = in_xds['sc_cropped_mesh'].values
    sc_cropped_mesh_norm = in_xds['sc_cropped_mesh_norm'].values
    sc_n_triangles = in_xds['sc_n_triangles'].values

    phase = actual_jitted_pipeline(wavelength, incident_light, horn_orientation, horn_length, horn_diameter,
                                   horn_position, phase_offset, epsilon, pr_pnt, pr_norm, sc_pnt,
                                   sc_cropped_mesh, sc_cropped_mesh_norm, sc_n_triangles)

    # Grid Phase
    x_pnt = pr_pnt[:, 0]
    y_pnt = pr_pnt[:, 1]
    x_axis = simple_axis(x_pnt, resolution)
    y_axis = simple_axis(y_pnt, resolution)
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
    selection = np.isfinite(phase)
    if np.sum(selection) == 0:
        logger.warning(f'Phase is empty')
        return None

    gridded_phase = griddata((x_pnt[selection], y_pnt[selection]), phase[selection], (x_mesh, y_mesh), 'linear')

    return gridded_phase, x_axis, y_axis


@njit(cache=True, nogil=True)
def actual_jitted_pipeline(wavelength, incident_light, horn_orientation, horn_length, horn_diameter, horn_position,
                           phase_offset, epsilon, pr_pnt, pr_norm, sc_pnt, sc_cropped_mesh,
                           sc_cropped_mesh_norm, sc_n_triangles):
    # Primary reflections
    light = np.zeros_like(pr_pnt)
    light[:] = incident_light
    pr_reflec = light - 2 * inner_product_2d_jit(light, pr_norm) * pr_norm

    # Secondary reflections
    sc_reflec, sc_reflec_pnt, sc_reflec_triangle = cropped_secondary_mesh(pr_reflec, pr_pnt, sc_pnt, sc_cropped_mesh,
                                                                          sc_cropped_mesh_norm, sc_n_triangles)

    # Secondary to horn
    horn_mouth_center = horn_length * horn_orientation
    sec_to_horn = horn_mouth_center - sc_reflec_pnt
    # horn_orien_2d = np.zeros_like(pr_pnt)
    # horn_orien_2d[:] = horn_orientation
    dot_horn_plane = np.dot(sec_to_horn, horn_orientation)
    reflec_horn = np.dot(sc_reflec, horn_orientation)
    line_par = np.where(np.abs(dot_horn_plane) < epsilon, np.nan, dot_horn_plane/reflec_horn)
    intersect_point = sc_reflec_pnt + line_par[:, np.newaxis]*sc_reflec
    dist_horn_mouth = compute_distances2(intersect_point, horn_mouth_center)
    horn_intersect = np.where(dist_horn_mouth[:, np.newaxis] <= horn_diameter, intersect_point, np.nan)

    # Compute Full light path
    pr_z_val = pr_pnt[:, 2]
    pr_z_max = np.max(pr_z_val)
    light_z = incident_light[2]
    # From the plane defined by the leading edge of the primary reflector to primary reflector point along light ray
    zeroth_distance = (pr_z_val - pr_z_max) / light_z
    # From point in the primary to point in the secondary along light ray
    first_distance = compute_distances2(pr_pnt, sc_reflec_pnt)
    # From point in the secondary to horn mouth
    second_distance = compute_distances2(sc_reflec_pnt, horn_intersect)
    # from horn mouth to receptor inside horn
    third_distance = compute_distances2(horn_intersect, horn_position)
    full_light_path = zeroth_distance + first_distance + second_distance + third_distance

    # Compute phase from light path
    n_wavelength = full_light_path/wavelength
    floor_n_wave = np.floor(n_wavelength)
    phase = (n_wavelength-floor_n_wave) * twopi - np.pi
    phase += phase_offset

    return phase


@njit(cache=True, nogil=True)
def actual_gridding(x_axis, y_axis, nx, ny, pr_pcd, pr_mesh, pr_mesh_norm, pr_pnt, pr_norm):

    #pcd_2d = pr_pcd[:, 0:2]
    ray_origin = np.array([0.0, 0.0, 0.0])
    ray_vector = np.array([0.0, 0.0, -1.0])

    print()
    ntot = nx * ny
    for ix in range(nx):
        ray_origin[0] = x_axis[ix]
        for iy in range(ny):
            ray_origin[1] = y_axis[iy]

            for itriangle, triangle in enumerate(pr_mesh):
                inside, intersect_pnt = moller_trumbore_algorithm(ray_origin, ray_vector, pr_pcd[triangle[0]],
                                                                  pr_pcd[triangle[1]], pr_pcd[triangle[2]])
                if inside:
                    pr_pnt[ix, iy, :] = intersect_pnt
                    pr_norm[ix, iy, :] = pr_mesh_norm[itriangle]
                    break

            # distances = np.sqrt(np.sum((pcd_2d - ray_origin[0:2]) ** 2, axis=1))
            # nearest_pnt_idx = np.argmin(distances)
            # if distances[nearest_pnt_idx] < distance_limit:
            #     for itriangle, triangle in enumerate(pr_mesh):
            #         if nearest_pnt_idx in triangle:
            #             inside, intersect_pnt = moller_trumbore_algorithm(ray_origin, ray_vector,
            #                                                               pr_pcd[triangle[0]], pr_pcd[triangle[1]],
            #                                                               pr_pcd[triangle[2]])
            #             if inside:
            #                 pr_pnt[ix, iy, :] = intersect_pnt
            #                 pr_norm[ix, iy, :] = pr_mesh_norm[itriangle]
            #                 break


            itot = ix * ny + iy +1
            print(f'{return_line}{itot}/{ntot}             ')

    return pr_pnt, pr_norm


def create_gridded_primary_dish(mesh_zarr_file, resolution, plot=False, raw_mesh=True):
    # Opening XDS
    focus_location = np.array((-1.136634465810194, 0, -0.331821128650557))
    in_xds = xr.open_zarr(mesh_zarr_file)
    if raw_mesh:
        pr_mesh = in_xds['primary_mesh'].values
        pr_mesh_norm = in_xds['primary_mesh_normals'].values
        pr_pcd = in_xds['primary_point_cloud'].values
        pr_pcd -= focus_location
    else:
        pr_mesh = in_xds['pr_mesh'].values
        pr_mesh_norm = in_xds['pr_mesh_norm'].values
        pr_pcd = in_xds['pr_pnt'].values

    # This needs to be tested
    # del in_xds

    x_axis = simple_axis(pr_pcd[:, 0], resolution)
    y_axis = simple_axis(pr_pcd[:, 1], resolution)


    # Grid Normals and points
    nx, ny = x_axis.shape[0], y_axis.shape[0]
    tri_shape = np.array([nx, ny, 3])
    pr_norm = np.full(tri_shape, np.nan)
    pr_pnt = np.full(tri_shape, np.nan)
    # pr_pnt, pr_norm =
    print(f'Gridding {nx} by {ny} points')
    actual_gridding(x_axis, y_axis, nx, ny, pr_pcd, pr_mesh, pr_mesh_norm, pr_pnt, pr_norm)


    x_pnt = pr_pcd[:, 0]
    y_pnt = pr_pcd[:, 1]
    z_pnt = pr_pcd[:, 2]
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
    gridded_z = griddata((x_pnt, y_pnt), z_pnt, (x_mesh, y_mesh), 'linear')



    if plot:
        axes = ['x', 'y', 'z']
        fig, ax = create_figure_and_axes([24, 8], [1, 3])
        for iax in range(3):
            title = f'{axes[iax]} coord'
            print(title)
            print(data_statistics(pr_pnt[:, :, iax]))
            imshow_plot(ax[iax], fig, title, x_axis, y_axis, pr_pnt[:, :, iax], 'viridis', [-20, 20],
                        fsize=5)
        close_figure(fig, 'Coord of gridded primary', 'gridded_pnt.png', 300, False, tight_layout=True)

        fig, ax = create_figure_and_axes([24, 8], [1, 3])
        for iax in range(3):
            title = f'{axes[iax]} normal'
            print(title)
            print(data_statistics(pr_norm[:, :, iax]))
            imshow_plot(ax[iax], fig, title, x_axis, y_axis, pr_norm[:, :, iax], 'viridis', [-1, 1],
                    fsize=5)
        close_figure(fig, 'Normal of gridded primary', 'gridded_norm.png', 300, False, tight_layout=True)

        fig, ax = create_figure_and_axes([10, 8], [1, 1])
        imshow_plot(ax, fig, 'Z gridded griddata', x_axis, y_axis, pr_norm[:, :, iax], 'viridis', [-20, 20], fsize=5)
        close_figure(fig, 'Z gridded griddata', 'z_griddata.png', 300, False, tight_layout=True)

        fig, ax = create_figure_and_axes([10, 8], [1, 1])
        scatter_plot(ax, x_pnt, 'X axis (m)', y_pnt, 'Y axis (m)')
        close_figure(fig, 'Ungrided X & Y', 'xy_ungridded.png', 300, False, tight_layout=True)

    return pr_pnt, pr_norm
