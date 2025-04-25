from astrohack.utils.ray_tracing_general import *
from astrohack.visualization.plot_tools import *

class NgvlaRayTracer:

    def __init__(self):
        # Metadata
        self.holds_qps = False
        self.npnt = -1
        self.wavelength = None

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

    @classmethod
    def from_local_qps_objects(cls, primary_local_qps, secondary_local_qps, wavelength):
        # if not isinstance(primary_local_qps, LocalQPS) or not isinstance(secondary_local_qps, LocalQPS):
        #     raise Exception('Local QPS descriptions should be LocalQPS instances')
        new_obj = cls()
        new_obj.primary_qps = primary_local_qps
        new_obj.secondary_qps = secondary_local_qps
        new_obj.holds_qps = True
        new_obj.wavelength = wavelength
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

    def plot_key(self, key, colormap='viridis', zlim=None, dpi=300):
        axis_labels = ['X', 'Y', 'Z']
        if 'grd_' not in key:
            raise Exception(f'{key} is not a plottable map.')

        degridded = self.__dict__[key]

        regridded = regrid_array(self.x_axis, self.y_axis, degridded, self.grd_idx)
        cmap = get_proper_color_map(colormap)

        if regridded.ndim == 3: # [nx,ny, 2|3]
            third_dim = regridded.shape[2]
            if third_dim == 3:
                fig_size = [18, 10]
            else: # third_dim == 2
                fig_size = [14, 10]
            fig, ax = create_figure_and_axes(fig_size, [1, third_dim])

            for iax in range(third_dim):
                simple_imshow_map_plot(ax[iax], fig, self.x_axis, self.y_axis, regridded[:, :, iax],
                                       f'{axis_labels[iax]} axis', cmap, zlim)
        else: # [nx, ny]
            fig, ax = create_figure_and_axes(None, [1, 1])
            simple_imshow_map_plot(ax, fig, self.x_axis, self.y_axis, regridded, '', cmap, zlim)

        close_figure(fig, f'{key} map', f'{key}_map.png', dpi, False)






