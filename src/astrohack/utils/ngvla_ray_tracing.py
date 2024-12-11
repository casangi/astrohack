import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from numba import njit
import random
import xarray as xr

from astrohack.utils import convert_unit
from astrohack.visualization.plot_tools import get_proper_color_map, create_figure_and_axes, well_positioned_colorbar, \
    close_figure, compute_extent


def numpy_size(array):
    units = ['B', 'kB', 'MB', 'GB']
    memsize = array.itemsize*array.size
    iu = 0
    while memsize > 1024:
        memsize /= 1024
        iu += 1
    print(f'Image size: {memsize:.2f} {units[iu]}')

@njit(cache=False, nogil=True)
def grad_calc(axis, vals):
    npnt = 0
    grad = 0
    for idx in range(len(axis)-1):
        if np.isnan(vals[idx]) or np.isnan(vals[idx+1]):
            pass
        else:
            dx = axis[idx+1] - axis[idx]
            df = vals[idx+1] - vals[idx]
            grad += df/dx
            npnt += 1

    if npnt == 0:
        return np.nan
    else:
        return grad/npnt


@njit(cache=False, nogil=True)
def grid_grad_jit(zgrid, xaxis, yaxis):
    x_grad = np.full_like(zgrid, np.nan)
    y_grad = np.full_like(zgrid, np.nan)

    for i_x in range(zgrid.shape[0]):
        for i_y in range(zgrid.shape[1]):
            if np.isnan(zgrid[i_x, i_y]):
                pass
            else:
                x_grad[i_x, i_y] = grad_calc(xaxis[i_x-1 : i_x+2],
                                             zgrid[i_x-1 : i_x+2, i_y])
                y_grad[i_x, i_y] = grad_calc(yaxis[i_y-1 : i_y+2],
                                             zgrid[i_x, i_y-1 : i_y+2])
    return x_grad, y_grad


def grid_grad_np(zgrid, xaxis, yaxis):
    x_grad = np.gradient(zgrid, xaxis.res, axis=0)
    y_grad = np.gradient(zgrid, yaxis.res, axis=1)
    return x_grad, y_grad


def read_cloud_with_normals(cloud_file, comment_char='#'):
    points = np.loadtxt(cloud_file, usecols=[0, 1, 2], unpack=False, comments=comment_char)
    normals = np.loadtxt(cloud_file, usecols=[3, 4, 5], unpack=False, comments=comment_char)
    return points, normals


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
def inner_product_2d_jit(vec_a, vec_b, keep_shape=True):
    """
    This routine expects that vec a and vec b are of the same shape: [n, 3]
    Args:
        vec_a: first vector array
        vec_b: second vector array
        keep_shape: output has the same shape as inputs (easier for subsequent broadcasting)

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
        print('\033[F', 100*it/pr_pnt.shape[0], '%\r')
        pnt_reflec = pr_reflec[it]
        pnt_diff = point-sc_pnt
        dist_vec = pnt_diff - inner_product_2d_jit(pnt_diff, pnt_reflec) * pnt_reflec
        dist_matrix = np.sqrt(np.sum(dist_vec**2, axis=1))
        isec_loc = np.argmin(dist_matrix)
        sc_reflec_dist[it] = dist_matrix[isec_loc]
        sc_reflec_pnt[it] = sc_pnt[isec_loc]
        sc_reflec[it] = pnt_reflec - 2*np.sum(pnt_reflec*sc_norm[isec_loc]) * sc_norm[isec_loc]

    return sc_reflec, sc_reflec_pnt, sc_reflec_dist


class Axis:
    def __init__(self, user_array, resolution):
        mini, maxi = np.min(user_array), np.max(user_array)
        npnt = int(np.ceil((maxi-mini) / resolution))
        axis_array = np.arange(npnt + 1)
        axis_array = resolution * axis_array
        axis_array = axis_array + mini + resolution / 2
        self.np = npnt
        self.res = resolution
        self.mini = mini
        self.maxi = maxi
        self.array = axis_array

    def idx_and_frac(self, coor):
        f_idx = (coor-self.array[0])/self.res
        i_idx = round(f_idx)
        if i_idx > f_idx:
            idx = [i_idx-1, i_idx]
            frac = 1-(i_idx-f_idx)
            fracs = [frac, 1-frac]
        elif i_idx < f_idx:
            idx = [i_idx, i_idx+1]
            frac = f_idx-i_idx
            fracs = [frac, 1-frac]
        else:
            idx = [i_idx]
            fracs = [1.0]

        if len(idx) > 1:
            if idx[0] < 0:
                idx = idx[1:]
                fracs = fracs[1:]
            elif idx[1] > self.np-1:
                idx = idx[:1]
                fracs = fracs[:1]

        return idx, fracs


class ReflectiveSurface:
    idx={'xy': [0, 1, 2],
         'zy': [2, 1, 0],
         'xz': [0, 2, 1],
         'yx': [1, 0, 2]
    }
    axes={'xy': ['X', 'Y', 'Z'],
          'zy': ['Z', 'Y', 'X'],
          'xz': ['X', 'Z', 'Y'],
          'yx': ['Y', 'X', 'Z']
    }
    keys = {'xy': np.array([0, 0]),
            'zy': [0, 1],
            'xz': [1, 0],
            'yx': [1, 1]
            }

    types = ['primary', 'secondary']

    def __init__(self, primary_cloud, secondary_cloud):
        self.primary_cloud = np.loadtxt(primary_cloud, unpack=True)
        self.secondary_cloud = np.loadtxt(secondary_cloud, unpack=True)
        self.x_axis = None
        self.y_axis = None
        self.zgridded = None
        self.x_grad = None
        self.y_grad = None
        self.vec_shape = None
        self.norm_vector = None
        self.reflection = None
        self._shift_to_focus_origin()

    def _shift_to_focus_origin(self):
        # Both dishes are in the same coordinates but this is not the
        # MR reference frame, but its axes are oriented the same
        # This is the coordinate of the focus on the MR.
        focus = [-1.136634465810194, 0, -0.331821128650557]
        # Translation simple shift across the axes as everything is in
        # meters
        for iax, axfocus in enumerate(focus):
            if axfocus != 0:
                self.primary_cloud[iax] -= axfocus
                self.secondary_cloud[iax] -= axfocus


    def grid_points(self, resolution=1e-3):
        # REMEMBER X is 0, Y is 1!!!
        self.x_axis = Axis(self.primary_cloud[0], resolution, margin=0)
        self.y_axis = Axis(self.primary_cloud[1], resolution, margin=0)

        x_mesh, y_mesh = np.meshgrid(self.x_axis.array, self.y_axis.array)
        self.zgridded = griddata((self.primary_cloud[0], self.primary_cloud[1]),
                                 self.primary_cloud[2], (x_mesh, y_mesh), 'cubic')
        numpy_size(self.zgridded)

    def find_closest_on_secondary(self):
        for ix in range(self.vec_shape[0]):
            for iy in range(self.vec_shape[1]):
                print(ix, iy)


    def compute_gradients(self):
        self.x_grad, self.y_grad = grid_grad_jit(self.zgridded,
                                                 self.x_axis.array,
                                                 self.y_axis.array)

    def compute_gradients_np(self):
        self.x_grad = np.gradient(self.zgridded, self.x_axis.res, axis=0)
        self.y_grad = np.gradient(self.zgridded, self.y_axis.res, axis=1)

    def compute_normal_vector(self):
        self.vec_shape = list(self.x_grad.shape)
        self.vec_shape.append(3)
        self.norm_vector = np.ndarray(self.vec_shape)
        vec_amp = np.sqrt(self.x_grad**2+self.y_grad**2+1)
        self.norm_vector[:, :, 0] = -self.x_grad/vec_amp
        self.norm_vector[:, :, 1] = -self.y_grad/vec_amp
        self.norm_vector[:, :, 2] = 1/vec_amp


    def compute_reflected_parallel(self):
        """
        Default light direction is parallel to the Z axis
        Args:
            light_direction: The unit vector representing light propagation

        Returns:
            the reflections for each elelement are stored in self.reflection
        """
        self.reflection = np.ndarray(self.vec_shape)
        nx = self.norm_vector[:, :, 0]
        ny = self.norm_vector[:, :, 1]
        nz = self.norm_vector[:, :, 2]
        ang_xz = np.arccos(nz/np.sqrt(nx**2+nz**2))
        ang_yz = np.arccos(nz/np.sqrt(ny**2+nz**2))
        # this is a rotation matrix, needs to be generalized for the
        # case of light not coming Z direction
        rx = np.sin(2*ang_xz)
        ry = -np.sin(2*ang_yz)
        rz = np.cos(2*ang_yz)*np.cos(2*ang_xz)
        # Reflections along y-axis have to be reflected for Y > 0
        pos_y = self.y_axis.array > 0
        rx[pos_y, :] *= -1

        # store in a single array
        self.reflection[:, :, 0] = rx
        self.reflection[:, :, 1] = ry
        self.reflection[:, :, 2] = rz

    def compute_general_reflection(self, light_direction):
        # Reflection is: i−2(i·n)n
        light = np.zeros_like(self.norm_vector)
        light[:, :] = np.array(light_direction)
        inner = np.empty_like(light)
        inner[:, :, 0] = np.sum(light*self.norm_vector, axis=2)
        inner[:, :, 1] = inner[:, :, 0]
        inner[:, :, 2] = inner[:, :, 0]
        self.reflection = light - 2*inner*self.norm_vector

    def plot_reflection(self, filename, nreflec=5):
        fig, ax = plt.subplots(1,2)
        ix = self.vec_shape[0]//2
        self._plot_reflection_cut(ix, 1, nreflec, ax[0], self.x_axis.array, self.zgridded[ix, :])
        iy = self.vec_shape[1]//2
        self._plot_reflection_cut(iy, 0, nreflec, ax[1], self.y_axis.array, self.zgridded[:, iy])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=8)

        fig.set_tight_layout(True)
        plt.savefig(filename, dpi=300)


    def _plot_reflection_cut(self, icut, cut_dim, nreflec, ax, xaxis, mirror_cut):
        # These plots need to present the projection of the reflection
        # onto the cut plane otherwise representation is misleading.

        ax.plot(xaxis, mirror_cut, color='blue', label='ngVLA primary reflector')
        ax.plot(0, 0, marker='x', color='yellow', label='Focus', ls='')

        for irefle in range(nreflec):
            inorm = random.randint(0, self.vec_shape[cut_dim]-1)
            if cut_dim == 1:
                norm_vec = self.norm_vector[icut, inorm]
                reflect_vec = self.reflection[icut, inorm]
            else:
                norm_vec = self.norm_vector[inorm, icut]
                reflect_vec = self.reflection[inorm, icut]

            xp = xaxis[inorm]
            zp = mirror_cut[inorm]
            incident = np.array([[xp, xp], [2, zp]])
            point = [xp, zp]
            ax.quiver(*point, norm_vec[cut_dim], norm_vec[2], color='black',
                       label='normal')

            ax.plot(incident[0], incident[1], label='incident light', color='red')
            ax.quiver(*point, reflect_vec[cut_dim], reflect_vec[2], color='green',
                       label='reflected light', scale=1e-5, width=0.005)

        if cut_dim == 1:
            ax.set_xlabel("Antenna X axis (m)")
            ax.set_title("Cut along main chord")
            ax.set_xlim([-4, 22])
        else:
            ax.set_xlabel("Antenna Y axis (m)")
            ax.set_title("Cut along X == 0")
            ax.set_xlim([-10, 10])
        ax.set_ylabel("Antenna Z Axis (m)")
        ax.set_ylim(-10, 2)


    def _plot_proj(self, proj, fig, ax, secondary_mirror=None, size=0.03, fsize=5):
        i1, i2, i3 = self.idx[proj]
        xax, yax, zax = self.axes[proj]
        minmax = [np.min(self.primary_cloud[i3]), np.max(self.primary_cloud[i3])]
        ax.scatter(self.primary_cloud[i1], self.primary_cloud[i2], c=self.primary_cloud[i3],
                   cmap='viridis', s=size)
        if secondary_mirror is None:
            ax.set_title(f'ngVLA {self.rtype} {proj.upper()} projection',
                         size=1.5*fsize)
        else:
            ax.scatter(secondary_mirror.primary_cloud[i1], secondary_mirror.primary_cloud[i2],
                       c=secondary_mirror.primary_cloud[i3], cmap='viridis', s=size)
            sminmax = [np.min(secondary_mirror.primary_cloud[i3]), np.max(secondary_mirror.primary_cloud[i3])]
            ax.set_title(f'ngVLA prototype {proj.upper()} projection',
                         size=1.5*fsize)
            if sminmax[0] < minmax[0]:
                minmax[0] = sminmax[0]
            if sminmax[1] > minmax[1]:
                minmax[1] = sminmax[1]
        ax.scatter(0,0, c='black', s=0.3, label='focus')
        ax.set_xlabel(f'{xax} axis [m]', size=fsize)
        ax.set_ylabel(f'{yax} axis [m]', size=fsize)

        norm = plt.Normalize(minmax[0], minmax[1])
        smap = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        cbar = fig.colorbar(smap, ax=ax, fraction=0.05)
        cbar.set_label(label=f'{zax} axis [m]',size=fsize)
        cbar.ax.tick_params(labelsize=fsize)
        ax.axis('equal')
        ax.tick_params(axis='both', which='major', labelsize=fsize)
        ax.legend(fontsize=fsize)

    def plot_2d(self, filename, secondary_mirror=None):
        fig, ax = plt.subplots(2,2)
        for key in self.idx.keys():
            ia1, ia2 = self.keys[key]
            self._plot_proj(key, fig, ax[ia1, ia2], secondary_mirror)
        fig.set_tight_layout(True)
        fig.savefig(filename, dpi=300)

    def plot_3d(self, filename, secondary_mirror=None, size=0.03):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        i1, i2, i3 = self.idx['xy']
        ax.scatter3D(self.primary_cloud[i1], self.primary_cloud[i2], self.primary_cloud[i3],
                     s=size)
        if secondary_mirror is not None:
            ax.scatter3D(secondary_mirror.primary_cloud[i1], secondary_mirror.primary_cloud[i2],
                         secondary_mirror.primary_cloud[i3], s=size)
        ax.scatter3D(0,0,0, s=0.3, label='Focus')
        ax.legend()
        fig.savefig(filename, dpi=300)

    def plot_grid(self, filename):
        fig, ax = plt.subplots(2,2)
        labels = ['grid', 'x grad', 'y grad']
        images = [self.zgridded, self.x_grad, self.y_grad]

        for i_im in range(len(images)):
            ix = i_im//2
            iy = i_im%2
            im = ax[ix, iy].imshow(images[i_im], cmap='viridis')
            ax[ix, iy].set_title(labels[i_im])
            fig.colorbar(im, ax=ax[ix, iy], fraction=0.03)

        fig.set_tight_layout(True)
        fig.savefig(filename, dpi=300)


class NgvlaRayTracer:

    def __init__(self, focus_location=(-1.136634465810194, 0, -0.331821128650557)):
        self.pr_pnt = self.pr_norm = None
        self.sc_pnt = self.sc_norm = None
        self.focus_offset = np.array(focus_location)
        self.pr_reflec = None
        self.sc_reflec = None
        self.sc_reflec_pnt = None
        self.sc_reflec_dist = None

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
        self.pr_reflec = light - 2*inner_product_2d(light, self.pr_norm) * self.pr_norm

    def secondary_reflection(self):
        self.sc_reflec = np.empty_like(self.pr_reflec)
        self.sc_reflec_pnt = np.empty_like(self.pr_reflec)
        print()
        for it, point, in enumerate(self.pr_pnt):
            print(f'\033[F{100*it/self.pr_pnt.shape[0]:.2f}%')
            pnt_reflec = self.pr_reflec[it]
            pnt_diff = point-self.sc_pnt
            dist_vec = pnt_diff - inner_product_2d(pnt_diff, pnt_reflec) * pnt_reflec
            dist_matrix = np.sqrt(inner_product_2d(dist_vec, dist_vec, keep_shape=False))
            isec_loc = np.argmin(dist_matrix)
            self.sc_reflec_pnt[it] = self.sc_pnt[isec_loc]
            self.sc_reflec[it] = pnt_reflec - 2*np.inner(pnt_reflec, self.sc_norm[isec_loc]) * self.sc_norm[isec_loc]
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

        ax.set_title(title,size=1.5*fsize)
        extent = compute_extent(x_axis.array, y_axis.array, margin=0.1)
        im = ax.imshow(gridded_data, cmap=cmap, extent=extent, interpolation="nearest", vmin=minmax[0], vmax=minmax[1])
        well_positioned_colorbar(ax, fig, im, "Z Scale")
        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")

        close_figure(fig, '', filename, 300, False)

    def plot_simple_2d(self, data_types, rootname, resolution, resolution_unit, colormap='viridis'):
        prog_res = convert_unit(resolution_unit, 'm', 'length')*resolution

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
                zlim=[0, 0.05]
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

    def from_xds(self, xds_name):
        xds = xr.open_zarr(xds_name)
        for key, item in xds.attrs.items():
            self.__setattr__(key, item)
        for key, item in xds.items():
            self.__setattr__(key, item.values)
        self.pr_pnt = xds.pr_pnt.values
        self.sc_pnt = xds.sc_pnt.values

    def from_point_cloud(self, primary_pcd, secondary_pcd):
        self.pr_pnt, self.pr_norm = read_cloud_with_normals(primary_pcd)
        self.sc_pnt, self.sc_norm = read_cloud_with_normals(secondary_pcd)
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








