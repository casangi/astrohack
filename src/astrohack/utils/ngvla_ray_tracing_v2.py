from typing import Union

import numpy as np

from astrohack.utils.ray_tracing_general import *
from astrohack.visualization.plot_tools import *
from astrohack.utils.algorithms import phase_wrapping
from astrohack.utils.constants import twopi


class NgvlaRayTracer:

    def __init__(self):
        # Metadata
        self.holds_qps = False
        self.npnt = -1
        self.wavelength = None
        self.tel_pars = None

        # Local QPS objects
        self.primary_qps = LocalQPS()
        self.secondary_qps = LocalQPS()

        # Data arrays
        self.x_axis = np.empty([0])
        self.y_axis = np.empty([0])
        self.incident_ray = np.empty([0])

        self.grd_pm_points = np.empty([0])
        self.grd_pm_normals = np.empty([0])
        self.grd_idx = np.empty([0])
        self.grd_pm_reflections = np.empty([0])
        self.grd_sc_points = np.empty([0])
        self.grd_sc_normals = np.empty([0])
        self.grd_sc_reflections = np.empty([0])
        self.grd_length_sc_to_horn = np.empty([0])
        self.grd_horn_intercept  = np.empty([0])
        self.grd_total_path = np.empty([0])
        self.grd_phase = np.empty([0])

    @classmethod
    def from_local_qps_objects(cls, primary_local_qps, secondary_local_qps, wavelength, telescope_parameters):
        # if not isinstance(primary_local_qps, LocalQPS) or not isinstance(secondary_local_qps, LocalQPS):
        #     raise Exception('Local QPS descriptions should be LocalQPS instances')
        new_obj = cls()
        new_obj.primary_qps = primary_local_qps
        new_obj.secondary_qps = secondary_local_qps
        new_obj.holds_qps = True
        new_obj.wavelength = wavelength
        new_obj.tel_pars = telescope_parameters
        return new_obj

    def grid_primary_mirror(self, sampling, active_radius=9.0, x_off=None, y_off=None):
        self.x_axis, self.y_axis, self.grd_pm_points, self.grd_pm_normals, self.grd_idx = \
            self.primary_qps.grid_primary(sampling, active_radius, x_off, y_off)
        self.npnt = self.grd_pm_points.shape[0]

    def reflect_off_primary(self, incident_light):
        # Left at 1D to save memory on purpose
        self.incident_ray = normalize_vector_map(incident_light)
        self.grd_pm_reflections = reflect_light(self.incident_ray[np.newaxis, :], self.grd_pm_normals)

    def reflect_off_secondary(self):
        self.grd_sc_points = np.empty_like(self.grd_pm_points)
        self.grd_sc_normals = np.empty_like(self.grd_pm_normals)

        for ipnt in range(self.npnt):
            self.grd_sc_points[ipnt], self.grd_sc_normals[ipnt] = \
                self.secondary_qps.find_reflection_point(self.grd_pm_points[ipnt], self.grd_pm_reflections[ipnt],
                                                         self.wavelength)

        self.grd_sc_reflections = reflect_light(self.grd_pm_reflections, self.grd_sc_normals)

    def detect_rays(self):
        horn_orientation = np.empty_like(self.grd_sc_reflections)
        horn_position = np.empty_like(self.grd_sc_reflections)
        horn_orientation[:] = self.tel_pars['horn_orientation']
        horn_position[:] = self.tel_pars['horn_position']
        horn_diameter = self.tel_pars['horn_inner_diameter']

        self.grd_length_sc_to_horn = (generalized_dot((horn_position - self.grd_sc_points), horn_orientation) /
                                      generalized_dot(self.grd_sc_reflections, horn_orientation))
        self.grd_horn_intercept = (self.grd_sc_points + self.grd_length_sc_to_horn[..., np.newaxis] *
                                   self.grd_sc_reflections)
        distance_to_horn_center = generalized_norm(self.grd_horn_intercept - horn_position)

        selection = distance_to_horn_center > horn_diameter
        self.grd_horn_intercept[selection, :] = nanvec3d

    def compute_phase(self, phase_offset=0):
        distance_pr_to_sc = generalized_dist(self.grd_pm_points, self.grd_sc_points)
        primary_points_z = self.grd_pm_points[:, 2]
        distance_pr_horn = distance_pr_to_sc + self.grd_length_sc_to_horn
        incident_light = np.empty_like(self.grd_pm_points)
        incident_light[:] = self.incident_ray

        maxheight = np.max(primary_points_z)
        boresight = np.empty_like(incident_light)
        boresight[:] = [0, 0, -1]  # strictly vertical
        cosbeta = generalized_dot(boresight, incident_light)
        path_diff_before_dish = (maxheight - primary_points_z) / cosbeta
        self.grd_total_path = np.where(np.isnan(self.grd_horn_intercept[:, 0]), np.nan, distance_pr_horn +
                                       path_diff_before_dish)

        wavenumber = self.grd_total_path / self.wavelength
        self.grd_phase = phase_wrapping(twopi * wavenumber + phase_offset)

    def plot_key(self, key, colormap='viridis', zlim=None, dpi=300):
        axis_labels = ['X', 'Y', 'Z']
        if 'grd_' not in key:
            raise Exception(f'{key} is not a plottable map.')

        degridded = self.__dict__[key]

        regridded = regrid_array(self.x_axis, self.y_axis, degridded, self.grd_idx)
        cmap = get_proper_color_map(colormap)

        if regridded.ndim == 3:  # [nx,ny, 2|3]
            third_dim = regridded.shape[2]
            if third_dim == 3:
                fig_size = [18, 10]
            else:  # third_dim == 2
                fig_size = [14, 10]
            fig, ax = create_figure_and_axes(fig_size, [1, third_dim])

            for iax in range(third_dim):
                simple_imshow_map_plot(ax[iax], fig, self.x_axis, self.y_axis, regridded[:, :, iax],
                                       f'{axis_labels[iax]} axis', cmap, zlim)
        else:  # [nx, ny]
            fig, ax = create_figure_and_axes(None, [1, 1])
            simple_imshow_map_plot(ax, fig, self.x_axis, self.y_axis, regridded, '', cmap, zlim)

        close_figure(fig, f'{key} map', f'{key}_map.png', dpi, False)

    def plot_3d_ray(self, choosen_ray=None, display=True, dpi=300, n_rays=None):
        if choosen_ray is None:
            if n_rays is None:
                i_rays = [np.random.randint(0, self.npnt)]
            else:
                i_rays = []
                for i_ray in range(n_rays):
                    i_rays.append(np.random.randint(0, self.npnt))
        else:
            if isinstance(choosen_ray, Union[list, tuple]):
                i_rays = choosen_ray
            else:
                i_rays = [choosen_ray]

        fig, ax = create_figure_and_axes(None, [1, 1], plot_is_3d=True)

        ax.scatter(self.grd_pm_points[:, 0], self.grd_pm_points[:, 1], self.grd_pm_points[:, 2], marker='.',
                   color='black', label='Primary mirror')
        sc_pcd = self.secondary_qps.global_pcd
        # ax.scatter(self.grd_sc_points[:, 0], self.grd_sc_points[:, 1], self.grd_sc_points[:, 2], marker='.',
        #            color='blue', label='Secondary mirror')
        ax.scatter(sc_pcd[:, 0], sc_pcd[:, 1], sc_pcd[:, 2], marker='.',
                   color='blue', label='Secondary mirror')
        focus_pos = self.tel_pars['horn_position']
        ax.scatter(focus_pos[0], focus_pos[1], focus_pos[2], marker='v', color='red', label='Focus')

        for i_ray in i_rays:
            self._add_3d_ray_to_ax(ax, i_ray)

        ax.set_xlabel('X axis [m]')
        ax.set_ylabel('Y axis [m]')
        ax.set_zlabel('Z axis [m]')
        ax.legend()
        close_figure(fig, f'3D visualization of rays', f'ray_visualization.png', dpi, display)

    def _add_3d_ray_to_ax(self, ax, i_ray, style='--', plot_normals=False):
        pm_point = self.grd_pm_points[i_ray]
        sc_point = self.grd_sc_points[i_ray]
        horn_pnt = self.grd_horn_intercept[i_ray]
        light_origin = pm_point - 20 * self.incident_ray

        if np.isnan(sc_point[0]):  # Did not touch secondary
            inf_dir = pm_point+self.grd_pm_reflections[i_ray]*20
            ray_x = [light_origin[0], pm_point[0], inf_dir[0]]
            ray_y = [light_origin[1], pm_point[1], inf_dir[1]]
            ray_z = [light_origin[2], pm_point[2], inf_dir[2]]
            ray_color = 'red'

        else:  # Touches Secondary
            if np.isnan(horn_pnt[0]):  # Did not reach horn
                inf_dir = sc_point+self.grd_sc_reflections[i_ray]*20
                ray_x = [light_origin[0], pm_point[0], sc_point[0], inf_dir[0]]
                ray_y = [light_origin[1], pm_point[1], sc_point[1], inf_dir[1]]
                ray_z = [light_origin[2], pm_point[2], sc_point[2], inf_dir[2]]
                ray_color = 'purple'
            else:  # fully detected ray
                ray_x = [light_origin[0], pm_point[0], sc_point[0], horn_pnt[0]]
                ray_y = [light_origin[1], pm_point[1], sc_point[1], horn_pnt[1]]
                ray_z = [light_origin[2], pm_point[2], sc_point[2], horn_pnt[2]]
                ray_color = 'green'

        ax.plot(ray_x, ray_y, ray_z, color=ray_color, ls=style, label=f'Ray #{i_ray}')
        if plot_normals:
            ax.quiver(*pm_point, *self.grd_pm_normals[i_ray], label=f'Primary normal', color='black')
            if not np.isnan(sc_point[0]):
                ax.quiver(*sc_point, *self.grd_sc_normals[i_ray], label=f'Secondary normal', color='yellow')
        return

    def plot_detailed_single_ray(self, chosen_ray, dpi=300, display=True):
        fig, ax = create_figure_and_axes(None, [1, 1], plot_is_3d=True)
        focus_pos = self.tel_pars['horn_position']
        ax.scatter(focus_pos[0], focus_pos[1], focus_pos[2], marker='v', color='red', label='Focus')

        self._add_3d_ray_to_ax(ax, chosen_ray, plot_normals=True)

        ax.set_xlabel('X axis [m]')
        ax.set_ylabel('Y axis [m]')
        ax.set_zlabel('Z axis [m]')
        ax.legend()
        close_figure(fig, f'Detailed visualization of rays', f'detailed_{chosen_ray}_ray_vis.png', dpi, display)
        return






