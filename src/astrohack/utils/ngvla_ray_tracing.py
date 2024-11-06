import numpy as np
from numpy.linalg import norm as norm
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import time
from numba import njit
import random

def numpy_size(array):
    units = ['B', 'kB', 'MB', 'GB']
    memsize = array.itemsize*array.size
    iu = 0
    while memsize > 1024:
        memsize /= 1024
        iu += 1
    print(f'Image size: {memsize:.2f} {units[iu]}')

def resolution_str(res):
    if res >= 1:
        return f"{res} m"
    elif res >= 0.01:
        return f"{100*res} cm"
    elif res >= 0.001:
        return f"{1000*res} mm"
    else:
        return f"{1e6*res} um"



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



def print_elapsed(start, label):
    stop = time.time()
    elap = stop-start
    print(f'{label}: {elap:.2f} s')


class Axis:
    def __init__(self, array, res):
        mini, maxi = np.min(array), np.max(array)
        npnt = int(np.ceil((maxi-mini)/res))
        array = np.arange(npnt+1)
        array = res*array
        array += mini+res/2
        self.np = npnt
        self.res = res
        self.mini = mini
        self.maxi = maxi
        self.array = array

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
            frac = [1.0]

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

    def __init__(self, filename, rtype):
        self.cloud = np.loadtxt(filename, unpack=True)
        self.np = self.cloud.shape[1]
        #self.cloud[2] = -self.cloud[2]
        if rtype in self.types:
            self.rtype = rtype
        else:
            raise Exception(f'Unknown reflector type {rtype}')
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
                self.cloud[iax] -= axfocus


    def grid_points(self, resolution=1e-3):
        # REMEMBER X is 0, Y is 1!!!
        self.x_axis = Axis(self.cloud[0], resolution)
        self.y_axis = Axis(self.cloud[1], resolution)

        x_mesh, y_mesh = np.meshgrid(self.x_axis.array, self.y_axis.array)
        self.zgridded = griddata((self.cloud[0], self.cloud[1]),
                                 self.cloud[2], (x_mesh, y_mesh), 'cubic')
        numpy_size(self.zgridded)


    def compute_gradients(self):
        self.x_grad, self.y_grad = grid_grad_jit(self.zgridded,
                                                 self.x_axis.array,
                                                 self.y_axis.array)

    def compute_normal_vector(self):
        self.vec_shape = list(self.x_grad.shape)
        self.vec_shape.append(3)
        self.norm_vector = np.ndarray(self.vec_shape)
        vec_amp = np.sqrt(self.x_grad**2+self.y_grad**2+1)
        self.norm_vector[:,:,0] = -self.x_grad/vec_amp
        self.norm_vector[:,:,1] = -self.y_grad/vec_amp
        self.norm_vector[:,:,2] = 1/vec_amp


    def compute_reflected_parallel(self):
        self.reflection = np.ndarray(self.vec_shape)
        inf_light = np.array([0,0,1])
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
        # Reflections along y axis have to be reflected for Y > 0
        pos_y = self.y_axis.array > 0
        rx[pos_y, :] *= -1

        # store in a single array
        self.reflection[:, :, 0] = rx
        self.reflection[:, :, 1] = ry
        self.reflection[:, :, 2] = rz


    def plot_reflection(self, nreflec=5):
        fig, ax = plt.subplots(1,2)
        ix = self.vec_shape[0]//2
        self._plot_reflection_cut(ix, 1, nreflec, ax[0], self.x_axis.array, self.zgridded[ix, :])
        iy = self.vec_shape[1]//2
        self._plot_reflection_cut(iy, 0, nreflec, ax[1], self.y_axis.array, self.zgridded[:, iy])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=8)

        fig.set_tight_layout(True)
        plt.savefig('test_reflection.png', dpi=300)


    def _plot_reflection_cut(self, icut, cut_dim, nreflec, ax, xaxis, mirror_cut):
        # These plots need to present the projection of the reflection
        # onto the cut plane other wise representation is misleading.

        ax.plot(xaxis, mirror_cut, color='blue', label='ngVLA primary reflector')
        ax.plot(0, 0, marker='x', color='yellow', label='Focus')

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
            ax.set_xlim([-2, 22])
        else:
            ax.set_xlabel("Antenna Y axis (m)")
            ax.set_title("Cut along X == 0")
            ax.set_xlim([-10, 10])
        ax.set_ylabel("Antenna Z Axis (m)")
        ax.set_ylim(-9, 2)


    def _plot_proj(self, proj, fig, ax, secondary=None, size=0.03, fsize=5):
        i1, i2, i3 = self.idx[proj]
        xax, yax, zax = self.axes[proj]
        minmax = [np.min(self.cloud[i3]), np.max(self.cloud[i3])]
        ax.scatter(self.cloud[i1], self.cloud[i2], c=self.cloud[i3],
                   cmap='viridis', s=size)
        if secondary is None:
            ax.set_title(f'ngVLA {self.rtype} {proj.upper()} projection',
                         size=1.5*fsize)
        else:
            ax.scatter(secondary.cloud[i1], secondary.cloud[i2],
                       c=secondary.cloud[i3], cmap='viridis', s=size)
            sminmax = [np.min(secondary.cloud[i3]), np.max(secondary.cloud[i3])]
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

    def plot_2d(self, filename, secondary=None):
        fig, ax = plt.subplots(2,2)
        for key in self.idx.keys():
            ia1, ia2 = self.keys[key]
            self._plot_proj(key, fig, ax[ia1, ia2], secondary)
        fig.set_tight_layout(True)
        fig.savefig(filename, dpi=300)

    def plot_3d(self, filename, secondary=None, size=0.03):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        i1, i2, i3 = self.idx['xy']
        ax.scatter3D(self.cloud[i1], self.cloud[i2], self.cloud[i3],
                     s=size)
        if secondary is not None:
            ax.scatter3D(secondary.cloud[i1], secondary.cloud[i2],
                         secondary.cloud[i3], s=size)
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




start = time.time()
primary = ReflectiveSurface('primary_mirror.dat', 'primary')
print_elapsed(start, 'Init')
secondary = ReflectiveSurface('secondary_mirror.dat', 'secondary')

# primary.plot_2d('primary_2d.png')
# primary.plot_3d('primary_3d.png')
# primary.plot_3d('both_3d.png', secondary)
# primary.plot_2d('both_2d.png', secondary)
# secondary.plot_2d('secondary_2d.png')
# secondary.plot_3d('secondary_3d.png')


resos = [5e-2, 1e-2, 5e-3, 1e-3]
resos = [1e-2]

for res in resos:
    start = time.time()
    primary.grid_points(resolution=res)
    print_elapsed(start, f'gridding with {resolution_str(res)}')
    start = time.time()
    primary.compute_gradients()
    print_elapsed(start, f'gradients with {resolution_str(res)}')
    # primary.plot_grid(f'primary_gridded_res_{res:.3f}.png')
    start = time.time()
    primary.compute_normal_vector()
    print_elapsed(start, f'Normal vectors with {resolution_str(res)}')
    start = time.time()
    primary.compute_reflected_parallel()
    print_elapsed(start, f'parallel reflection with {resolution_str(res)}')

    primary.plot_reflection(nreflec=20)
