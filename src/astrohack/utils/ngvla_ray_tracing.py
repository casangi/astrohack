import numpy as np
from scipy.interpolate import griddata
from numba import njit
import xarray as xr

from astrohack.utils import convert_unit
from astrohack.visualization.plot_tools import get_proper_color_map, create_figure_and_axes, well_positioned_colorbar, \
    close_figure, compute_extent


def numpy_size(array):
    units = ['B', 'kB', 'MB', 'GB']
    memsize = array.itemsize * array.size
    iu = 0
    while memsize > 1024:
        memsize /= 1024
        iu += 1
    print(f'Image size: {memsize:.2f} {units[iu]}')


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


@njit(cache=False, nogil=True)
def secondary_reflec_jit(pr_pnt, pr_reflec, sc_pnt, sc_norm):
    sc_reflec = np.empty_like(pr_reflec)
    sc_reflec_pnt = np.empty_like(pr_reflec)
    sc_reflec_dist = np.empty(pr_reflec.shape[0])
    for it, point, in enumerate(pr_pnt):
        print('\033[F', 100 * it / pr_pnt.shape[0], '%          ')
        pnt_reflec = pr_reflec[it]
        pnt_diff = point - sc_pnt
        dist_vec = pnt_diff - inner_product_2d_jit(pnt_diff, pnt_reflec) * pnt_reflec
        dist_matrix = np.sqrt(np.sum(dist_vec ** 2, axis=1))
        isec_loc = np.argmin(dist_matrix)
        sc_reflec_dist[it] = dist_matrix[isec_loc]
        sc_reflec_pnt[it] = sc_pnt[isec_loc]
        sc_reflec[it] = pnt_reflec - 2 * np.sum(pnt_reflec * sc_norm[isec_loc]) * sc_norm[isec_loc]

    return sc_reflec, sc_reflec_pnt, sc_reflec_dist


# @njit(cache=False, nogil=True)
def triangle_intersection(va, vb, vc, t_norm, start_pnt, direction):
    vba = vb - va
    vca = vc - va
    vao = start_pnt - va
    vcao = np.cross(vao, direction)
    det = np.dot(direction, t_norm)
    uval = np.dot(vca, vcao) / det
    vval = -np.dot(vba, vcao) / det
    line_par = np.dot(vao, t_norm) / np.dot(direction, t_norm)
    # line_par = np.dot(vao, t_norm) / det
    intersect_pnt = start_pnt + line_par * direction
    intersect = abs(det) >= 1e-6 and line_par > 0 and uval >= 0 and vval >= 0 and uval+vval <= 1.0
    print(intersect, intersect_pnt, va, vb, vc)
    return intersect, intersect_pnt


@njit(cache=False, nogil=True)
def intersect_line_triangle(start_point, direction, t_norm, p1, p2, p3):
    def signed_tetra_volume(a, b, c, d):
        return np.sign(np.dot(np.cross(b-a, c-a), d-a)/6.0)

    point_over_ray = start_point + direction
    s1 = signed_tetra_volume(start_point, p1, p2, p3)
    s2 = signed_tetra_volume(point_over_ray, p1, p2, p3)

    if s1 != s2:
        s3 = signed_tetra_volume(start_point, point_over_ray, p1, p2)
        s4 = signed_tetra_volume(start_point, point_over_ray, p2, p3)
        s5 = signed_tetra_volume(start_point, point_over_ray, p3, p1)
        if s3 == s4 and s4 == s5:
            line_par = -np.dot(start_point, t_norm) / np.dot(start_point, direction)
            return True, start_point + line_par * direction
    return False, np.array([np.nan, np.nan, np.nan])


@njit(cache=False, nogil=True)
def reflect_on_surface(light, normal):
    return light - 2 * np.dot(light, normal)*normal

@njit(cache=False, nogil=True)
def jitted_trinagle_find(pr_point, reflection, sc_mesh, sc_pnt, sc_mesh_norm):
    for it, triangle in enumerate(sc_mesh):
        va = sc_pnt[int(triangle[0])]
        vb = sc_pnt[int(triangle[1])]
        vc = sc_pnt[int(triangle[2])]
        t_norm = sc_mesh_norm[it]
        # crosses_triangle, point = triangle_intersection(va, vb, vc, t_norm, pr_point, reflection)
        crosses_triangle, point = intersect_line_triangle(pr_point, reflection, t_norm, va, vb, vc)
        # triangle_intersection(va, vb, vc, t_norm, pr_point, reflection)
        if crosses_triangle:
            print(it, point)
            print()
            return it, point

    return np.nan, point

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

    def __init__(self, focus_location=(-1.136634465810194, 0, -0.331821128650557)):
        self.pr_pnt = self.pr_norm = None
        self.sc_pnt = self.sc_norm = None
        self.focus_offset = np.array(focus_location)
        self.pr_reflec = None
        self.sc_reflec = None
        self.sc_reflec_pnt = None
        self.sc_reflec_dist = None
        self.sc_reflec_triangle = None
        self.pr_mesh = None
        self.pr_mesh_norm = None
        self.sc_mesh = None
        self.sc_mesh_norm = None

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

    def primary_reflection(self, incident_light):
        light = np.zeros_like(self.pr_pnt)
        light[:] = np.array(incident_light)
        self.pr_reflec = light - 2 * inner_product_2d(light, self.pr_norm) * self.pr_norm

    def secondary_reflection(self):
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

    def secondary_reflection_jit(self):
        print()
        self.sc_reflec, self.sc_reflec_pnt, self.sc_reflec_dist = \
            secondary_reflec_jit(self.pr_pnt, self.pr_reflec, self.sc_pnt, self.sc_norm)
        print()

    def _grid_for_plotting(self, data_array, resolution):
        x_pnt = self.pr_pnt[:, 0]
        y_pnt = self.pr_pnt[:, 1]
        x_axis = Axis(x_pnt, resolution)
        y_axis = Axis(y_pnt, resolution)
        x_mesh, y_mesh = np.meshgrid(x_axis.array, y_axis.array)
        gridded_array = griddata((x_pnt, y_pnt),
                                 data_array, (x_mesh, y_mesh), 'cubic')

        return gridded_array, x_axis, y_axis

    def _plot_map(self, data_array, prog_res, title, filename, colormap, zlim, fsize=5):
        gridded_data, x_axis, y_axis = self._grid_for_plotting(data_array, prog_res)

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
        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")

        close_figure(fig, '', filename, 300, False)

    def plot_simple_2d(self, data_types, rootname, resolution, resolution_unit, colormap='viridis'):
        prog_res = convert_unit(resolution_unit, 'm', 'length') * resolution

        for data_type in data_types:
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
            elif data_type == 'reflec triangle':
                data_array = self.sc_reflec_triangle
                title = f'Triangle of reflection on the secondary'
            else:
                raise Exception(f'Unrecognized data type {data_type}')

            filename = f"{rootname}-{data_type.replace(' ', '-')}.png"
            self._plot_map(data_array, prog_res, title, filename, colormap, zlim)

    def save_to_zarr(self, filename):
        xds = xr.Dataset()

        for key, item in vars(self).items():
            if isinstance(item, np.ndarray):
                if key in ['sc_pnt', 'sc_norm']:
                    xds[key] = xr.DataArray(item, dims=['sc_pnt', 'xyz'])
                elif key == 'focus_offset':
                    xds[key] = xr.DataArray(item, dims=['xyz'])
                elif len(item.shape) == 2:
                    xds[key] = xr.DataArray(item, dims=['pr_pnt', 'xyz'])
                elif len(item.shape) == 1:
                    xds[key] = xr.DataArray(item, dims=['pr_pnt'])
                else:
                    raise Exception(f"Don't know what to do with {key}")
            elif item is None:
                pass
            else:
                xds.attrs[key] = item

        xds.to_zarr(filename, mode='w')

    def reread(self, xds_name):
        xds = xr.open_zarr(xds_name)
        for key, item in xds.attrs.items():
            self.__setattr__(key, item)
        for key, item in xds.items():
            self.__setattr__(str(key), item.values)
        self.pr_pnt = xds.pr_pnt.values
        self.sc_pnt = xds.sc_pnt.values

    def from_zarr_mesh(self, mesh_file):
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

        print('Primary cloud size:')
        numpy_size(self.pr_pnt)
        print('Secondary cloud size:')
        numpy_size(self.sc_pnt)

    def __repr__(self):
        outstr = 'Ray tracer contents:\n'
        for key, item in vars(self).items():
            outstr += f'{key:15s} = {item}\n'
        return outstr

    def _find_triangle_on_secondary(self, pr_point, reflection):
        """
        This returns the triangle index and the point in the triangle.
        """
        return jitted_trinagle_find(pr_point, reflection, self.sc_mesh, self.sc_pnt, self.sc_mesh_norm)
        # for it, triangle in enumerate(self.sc_mesh):
        #     va = self.sc_pnt[int(triangle[0])]
        #     vb = self.sc_pnt[int(triangle[1])]
        #     vc = self.sc_pnt[int(triangle[2])]
        #     t_norm = self.sc_mesh_norm[it]
        #     # crosses_triangle, point = triangle_intersection(va, vb, vc, t_norm, pr_point, reflection)
        #     crosses_triangle, point = intersect_line_triangle(pr_point, reflection, t_norm, va, vb, vc)
        #     # triangle_intersection(va, vb, vc, t_norm, pr_point, reflection)
        #     if crosses_triangle:
        #         print(it, point)
        #         print()
        #         return it, point
        #
        # return np.nan, np.full([3], np.nan)

    def secondary_reflection_on_mesh(self):
        self.sc_reflec = np.empty_like(self.pr_reflec)
        self.sc_reflec_pnt = np.empty_like(self.pr_reflec)
        self.sc_reflec_triangle = np.empty(self.pr_reflec.shape[0])
        print()

        for ipnt, pr_point in enumerate(self.pr_pnt):
        # if True:
            #ipnt = 0
            #pr_point = self.pr_pnt[0]
            print(f'\033[FMesh reflections: {100 * ipnt / self.pr_pnt.shape[0]:.2f}%')
            pr_reflection = self.pr_reflec[ipnt]
            triangle, sc_point = self._find_triangle_on_secondary(pr_point, pr_reflection)
            self.sc_reflec_triangle[ipnt] = triangle
            self.sc_reflec_pnt[ipnt] = sc_point

            # print(triangle, sc_point)
            if np.isnan(triangle):
                self.sc_reflec[ipnt] = np.full([3], np.nan)
            else:
                self.sc_reflec[ipnt] = reflect_on_surface(pr_reflection, self.sc_mesh_norm[triangle])

    def triangle_area(self):
        for it, triangle in enumerate(self.sc_mesh):
            va = self.sc_pnt[int(triangle[0])]
            vb = self.sc_pnt[int(triangle[1])]
            vc = self.sc_pnt[int(triangle[2])]
            vba = vb-va
            vbc = vb-vc
            cross = np.cross(vba, vbc)
            print(it, '=', np.sqrt(np.inner(cross, cross))/2)
