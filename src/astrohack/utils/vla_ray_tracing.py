import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import xarray as xr

from astrohack.utils import data_statistics
from astrohack.utils.constants import twopi
from astrohack.utils.conversion import convert_unit
from astrohack.utils.algorithms import phase_wrapping
from astrohack.utils.ray_tracing_general import generalized_dot, generalized_norm, normalize_vector_map, reflect_light
from astrohack.visualization.plot_tools import get_proper_color_map, create_figure_and_axes, well_positioned_colorbar, \
    close_figure, compute_extent

vla_pars = {
    'primary_diameter': 25.0,
    'secondary_diameter': 2.5146,
    'focal_length': 9.0,
    'z_intercept': 3.140,
    'foci_half_distance': 3.662,
    'inner_radius': 2.0,
    # Assuming a 10 cm Horn for now
    'horn_diameter': 0.2,
    # Assumed to be at the Secondary focus i.e.: f - 2c
    'horn_position': [0, 0, 9.0 - 2 * 3.662],
    # Horn looks straight up
    'horn_orientation': [0, 0, 1],
}

nanvec3d = np.full([3], np.nan)


######################################################################
# Setup routines and Mathematical description of the secondary shape #
######################################################################
def simple_axis(minmax, resolution, margin=0.05):
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


def create_coordinate_images(x_axis, y_axis):
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis, indexing='ij')
    img_radius = np.sqrt(x_mesh ** 2 + y_mesh ** 2)
    return x_mesh, y_mesh, img_radius



def create_radial_mask(radius, inner_rad, outer_rad):
    mask = np.full_like(radius, True, dtype=bool)
    mask = np.where(radius > outer_rad, False, mask)
    mask = np.where(radius < inner_rad, False, mask)
    return mask


def make_gridded_vla_primary(grid_size, resolution, telescope_pars):
    grid_minmax = [-grid_size / 2, grid_size / 2]
    axis = simple_axis(grid_minmax, resolution, margin=0.0)
    image_size = axis.shape[0]
    axis_idx = np.arange(image_size, dtype=int)

    # It is imperative to put indexing='ij' so that the x and Y axes are not flipped in this step.
    x_mesh, y_mesh, img_radius =  create_coordinate_images(axis, axis)
    x_idx_mesh, y_idx_mesh = np.meshgrid(axis_idx, axis_idx, indexing='ij')
    radial_mask = create_radial_mask(img_radius, telescope_pars['inner_radius'], telescope_pars['primary_diameter'] / 2)
    img_radius = img_radius[radial_mask]
    npnt_1d = img_radius.shape[0]
    idx_1d = np.empty([npnt_1d, 2], dtype=int)
    idx_1d[:, 0] = x_idx_mesh[radial_mask]
    idx_1d[:, 1] = y_idx_mesh[radial_mask]
    x_mesh_1d = x_mesh[radial_mask]
    y_mesh_1d = y_mesh[radial_mask]

    vec_shape = [npnt_1d, 3]
    focal_length = telescope_pars['focal_length']
    # Parabola formula = (x**2 + y**2)/4/focal_length
    gridded_primary = img_radius ** 2 / 4 / focal_length
    x_grad = np.zeros(vec_shape)
    y_grad = np.zeros(vec_shape)
    x_grad[:, 0] = 1.0
    x_grad[:, 2] = 2 * x_mesh_1d / 4 / focal_length
    y_grad[:, 1] = 1.0
    y_grad[:, 2] = 2 * y_mesh_1d / 4 / focal_length

    primary_normals = np.cross(x_grad, y_grad)
    primary_normals = normalize_vector_map(primary_normals)
    primary_points = np.empty_like(x_grad)
    primary_points[:, 0] = x_mesh_1d
    primary_points[:, 1] = y_mesh_1d
    primary_points[:, 2] = gridded_primary

    rt_xds = xr.Dataset()

    rt_xds.attrs['image_size'] = image_size
    rt_xds.attrs['npnt_1d'] = npnt_1d
    rt_xds.attrs['telescope_parameters'] = telescope_pars

    rt_xds["primary_points"] = xr.DataArray(primary_points, dims=['points', 'xyz'])
    rt_xds["primary_normals"] = xr.DataArray(primary_normals, dims=['points', 'vxyz'])
    rt_xds["image_indexes"] = xr.DataArray(idx_1d, dims=['points', 'idx'])
    rt_xds["x_axis"] = xr.DataArray(axis, dims=['x'])
    rt_xds["y_axis"] = xr.DataArray(axis, dims=['y'])

    return rt_xds


def secondary_hyperboloid_root_func(tval, fargs):
    pnt, ray, acoef, fcoef, ccoef, offsets = fargs
    # The offset is a simple displacement of the secondary
    newpnt = (pnt + tval * ray) - offsets
    rad2 = newpnt[0] ** 2 + newpnt[1] ** 2
    pntz = newpnt[2]
    value = fcoef - ccoef + acoef * np.sqrt(1 + rad2 / (ccoef ** 2 - acoef ** 2)) - pntz
    return value


##########################################################
# Actual ray tracing steps in order of light propagation #
##########################################################
def reflect_off_primary(rt_xds, incident_light):
    incident_light = normalize_vector_map(incident_light)
    primary_normals = rt_xds['primary_normals'].values
    light = np.zeros_like(primary_normals)
    light[:] = incident_light
    reflection = reflect_light(light, primary_normals)
    rt_xds['primary_reflections'] = xr.DataArray(reflection, dims=['points', 'vxyz'])
    rt_xds['light'] = xr.DataArray(light, dims=['points', 'vxyz'])
    return rt_xds


def reflect_off_analytical_secondary(rt_xds, offset=np.array((0, 0, 0))):
    primary_points = rt_xds['primary_points'].values
    primary_reflections = rt_xds['primary_reflections'].values
    telescope_pars = rt_xds.attrs['telescope_parameters']

    # this is simply 1D
    distance_to_secondary = np.empty_like(primary_points[:, 0])

    fargs = [None, None, telescope_pars['z_intercept'], telescope_pars['focal_length'],
             telescope_pars['foci_half_distance'], offset]

    for ipnt in range(rt_xds.attrs['npnt_1d']):
        fargs[0] = primary_points[ipnt]
        fargs[1] = primary_reflections[ipnt]
        # Focal length plus the height of departing point (distance from point to primary focus)
        initial_guess = telescope_pars['focal_length'] + primary_points[ipnt][2]
        val, _, ier, _ = fsolve(secondary_hyperboloid_root_func, initial_guess, args=fargs, maxfev=100,
                                full_output=True, xtol=1e-8)
        if ier == 1:
            distance_to_secondary[ipnt] = val
        else:
            distance_to_secondary[ipnt] = np.nan

    secondary_points = primary_points + distance_to_secondary[..., np.newaxis] * primary_reflections
    # Compute Gradients to compute normals at touched points
    x_grad = np.zeros_like(primary_points)
    y_grad = np.zeros_like(primary_points)
    dcoeff = telescope_pars['foci_half_distance'] ** 2 - telescope_pars['z_intercept'] ** 2
    px, py = secondary_points[:, 0], secondary_points[:, 1]
    root_term = telescope_pars['z_intercept'] / (dcoeff * np.sqrt(1 + (px ** 2 + py ** 2) / dcoeff))
    x_grad[:, 0] = 1.0
    y_grad[:, 1] = 1.0
    x_grad[:, 2] = px * root_term
    y_grad[:, 2] = py * root_term
    secondary_normals = normalize_vector_map(np.cross(x_grad, y_grad))
    secondary_reflections = reflect_light(primary_reflections, secondary_normals)

    rt_xds['distance_primary_to_secundary'] = xr.DataArray(distance_to_secondary, dims=['points'])
    rt_xds['secondary_points'] = xr.DataArray(secondary_points, dims=['points', 'xyz'])
    rt_xds['secondary_normals'] = xr.DataArray(secondary_normals, dims=['points', 'vxyz'])
    rt_xds['secondary_reflections'] = xr.DataArray(secondary_reflections, dims=['points', 'vxyz'])

    return rt_xds


def detect_light(rt_xds):
    secondary_reflections = rt_xds['secondary_reflections'].values
    secondary_points = rt_xds['secondary_points'].values
    telescope_pars = rt_xds.attrs['telescope_parameters']

    horn_orientation = np.empty_like(secondary_reflections)
    horn_position = np.empty_like(secondary_reflections)
    horn_orientation[:] = telescope_pars['horn_orientation']
    horn_position[:] = telescope_pars['horn_position']
    horn_diameter = telescope_pars['horn_diameter']

    distance_secondary_to_horn = (generalized_dot((horn_position - secondary_points), horn_orientation) /
                                  generalized_dot(secondary_reflections, horn_orientation))
    horn_intercept = secondary_points + distance_secondary_to_horn[..., np.newaxis] * secondary_reflections
    distance_to_horn_center = generalized_norm(horn_intercept - horn_position)

    selection = distance_to_horn_center > horn_diameter
    horn_intercept[selection, :] = nanvec3d

    rt_xds['distance_secondary_to_horn'] = xr.DataArray(distance_secondary_to_horn, dims=['points'])
    rt_xds['horn_intercept'] = xr.DataArray(horn_intercept, dims=['points', 'xyz'])
    return rt_xds


def compute_phase(rt_xds, wavelength, phase_offset):
    incident_light = rt_xds['light']
    primary_points_z = rt_xds['primary_points'].values[:, 2]
    distance_pr_horn = rt_xds['distance_secondary_to_horn'].values + rt_xds['distance_primary_to_secundary'].values

    maxheight = np.max(primary_points_z)
    boresight = np.empty_like(incident_light)
    boresight[:] = [0, 0, -1]  # strictly vertical
    cosbeta = generalized_dot(boresight, incident_light)
    path_diff_before_dish = (maxheight - primary_points_z) / cosbeta
    total_path = np.where(np.isnan(rt_xds['horn_intercept'].values[:, 0]), np.nan, distance_pr_horn +
                          path_diff_before_dish)

    wavenumber = total_path / wavelength
    phase = phase_wrapping(twopi * wavenumber + phase_offset)

    rt_xds['total_path'] = xr.DataArray(total_path, dims=['points'])
    rt_xds['phase'] = xr.DataArray(phase, dims=['points'])
    return rt_xds


###########################################################
# Plotting routines and plotting aids, such as regridding #
###########################################################
def regrid_data_onto_2d_grid(npnt, data, indexes):
    npnt_1d = data.shape[0]
    if len(data.shape) == 2:
        gridded_2d = np.full([npnt, npnt, data.shape[1]], np.nan)
        for ipnt in range(npnt_1d):
            ix, iy = indexes[ipnt]
            gridded_2d[ix, iy, :] = data[ipnt, :]
    else:
        gridded_2d = np.full([npnt, npnt], np.nan)
        for ipnt in range(npnt_1d):
            ix, iy = indexes[ipnt]
            gridded_2d[ix, iy] = data[ipnt]
    return gridded_2d


def title_from_input_parameters(inpt_dict):
    title = 'VLA ray Tracing model for:\n'
    title += (f'pnt off = ({inpt_dict["x_pnt_off"]}, {inpt_dict["y_pnt_off"]}) '
              f'{inpt_dict["pnt_off_unit"]}, ')
    title += (f'Focus offset = ({inpt_dict["x_focus_off"]}, {inpt_dict["y_focus_off"]}, '
              f'{inpt_dict["z_focus_off"]}) {inpt_dict["focus_off_unit"]}, ')
    title += f'Phase offset = {inpt_dict["phase_offset"]} {inpt_dict["phase_unit"]}, '
    title += f'$\lambda$ = {inpt_dict["observing_wavelength"]} {inpt_dict["wavelength_unit"]}'
    return title


def plot_2d_maps_from_rt_xds(rt_xds, keys, rootname, colormap='viridis'):
    suptitle = title_from_input_parameters(rt_xds.attrs['input_parameters'])
    for key in keys:
        filename = f'{rootname}_{key}.png'
        gridded_array = regrid_data_onto_2d_grid(rt_xds.attrs['image_size'], rt_xds[key].values,
                                                 rt_xds['image_indexes'].values)

        zlabel = key.capitalize()
        if key == 'phase':
            zlabel += ' [rad]'
            zlim = [-np.pi, np.pi]
        else:
            zlabel += ' [m]'
            zlim = None

        plot_2d_map(gridded_array, rt_xds["x_axis"].values, rt_xds.attrs['telescope_parameters'], suptitle, filename,
                    zlabel, colormap, zlim)
    return


def imshow_2d_map(ax, fig, gridded_array, title, extent, zlabel, colormap, inner_radius, outer_radius, zlim):
    cmap = get_proper_color_map(colormap)
    if zlim is None:
        minmax = [np.nanmin(gridded_array), np.nanmax(gridded_array)]
    else:
        minmax = zlim
    fsize = 10
    if title is not None:
        ax.set_title(title, size=1.5 * fsize)

    im = ax.imshow(gridded_array.T, cmap=cmap, extent=extent, interpolation="nearest", vmin=minmax[0], vmax=minmax[1],
                   origin='lower')

    if zlabel is None:
        well_positioned_colorbar(ax, fig, im, "Z Scale")
    else:
        well_positioned_colorbar(ax, fig, im, zlabel)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xlabel("X axis [m]")
    ax.set_ylabel("Y axis [m]")

    innercircle = plt.Circle((0, 0), inner_radius, color='black', fill=None)
    outercircle = plt.Circle((0, 0), outer_radius, color='black', fill=None)
    ax.add_patch(outercircle)
    ax.add_patch(innercircle)


def plot_2d_map(gridded_array, axis, telescope_parameters, suptitle, filename, zlabel, colormap, zlim):
    inner_radius = telescope_parameters['inner_radius']
    outer_radius = telescope_parameters['primary_diameter'] / 2
    axes = ['X', 'Y', 'Z']
    extent = compute_extent(axis, axis, margin=0.0)

    if len(gridded_array.shape) == 3:  # XYZ Plots
        fig, ax = create_figure_and_axes([18, 8], [1, 3])
        for iax in range(3):
            imshow_2d_map(ax[iax], fig, gridded_array[..., iax], f'{axes[iax]} Component', extent, zlabel, colormap,
                          inner_radius, outer_radius, zlim)
    else:
        fig, ax = create_figure_and_axes([10, 8], [1, 1])
        imshow_2d_map(ax, fig, gridded_array, None, extent, zlabel, colormap, inner_radius, outer_radius, zlim)

    close_figure(fig, suptitle, filename, 300, False)


def add_rz_ray_to_plot(ax, origin, destiny, color, ls, label, sign):
    radcoord = [sign * generalized_norm(origin[0:2]), sign * generalized_norm(destiny[0:2])]
    zcoord = [origin[2], destiny[2]]
    ax.plot(radcoord, zcoord, color=color, label=label, ls=ls)


def plot_radial_projection_from_rt_xds(rt_xds, filename, nrays=20):
    telescope_pararameters = rt_xds.attrs['telescope_parameters']
    primary_diameter = telescope_pararameters['primary_diameter']
    secondary_diameter = telescope_pararameters['secondary_diameter']
    focal_length = telescope_pararameters['focal_length']
    foci_half_distance = telescope_pararameters['foci_half_distance']
    z_intercept = telescope_pararameters['z_intercept']

    pr_rad = primary_diameter / 2
    sc_rad = secondary_diameter / 2
    radarr = np.arange(-pr_rad, pr_rad, primary_diameter / 1e3)
    primary = radarr ** 2 / 4 / focal_length
    secondary = focal_length - foci_half_distance + z_intercept * np.sqrt(
        1 + radarr ** 2 / (foci_half_distance ** 2 - z_intercept ** 2))
    secondary = np.where(np.abs(radarr) < sc_rad, secondary, np.nan)
    fig, ax = create_figure_and_axes([16, 8], [1, 1])
    ax.plot(radarr, primary, color='black', label='Pr mirror')
    ax.plot(radarr, secondary, color='blue', label='Sc mirror')
    ax.scatter([0], [focal_length], color='black', label='Pr focus')
    ax.scatter([0], [focal_length - 2 * foci_half_distance], color='blue', label='Sc focus')

    primary_points = rt_xds['primary_points'].values
    secondary_points = rt_xds['secondary_points'].values
    horn_intercepts = rt_xds['horn_intercept'].values
    incoming_light = rt_xds['light'].values
    secondary_reflections = rt_xds['secondary_reflections'].values
    primary_reflections = rt_xds['primary_reflections'].values

    npnt = primary_points.shape[0]
    sign = -1
    inf = 1e3
    for isamp in range(nrays):
        sign *= -1
        ipnt = np.random.randint(0, high=npnt)

        # Data Selection
        sc_pnt = secondary_points[ipnt]
        pr_pnt = primary_points[ipnt]
        pr_ref = primary_reflections[ipnt]
        sc_ref = secondary_reflections[ipnt]
        horn_inter = horn_intercepts[ipnt]
        incoming = incoming_light[ipnt]

        # Plot incident light
        origin = pr_pnt - inf * incoming
        add_rz_ray_to_plot(ax, origin, pr_pnt, 'yellow', '-', '$\infty$->Pr', sign)

        # Plot primary reflection
        if np.all(np.isnan(sc_pnt)):  # Ray does not touch secondary
            dest = pr_pnt + inf * pr_ref
            add_rz_ray_to_plot(ax, pr_pnt, dest, 'red', '--', 'Pr->$\infty$', sign)
        else:
            add_rz_ray_to_plot(ax, pr_pnt, sc_pnt, 'yellow', '--', 'Pr->Sc', sign)

            # Plot secondary reflection
            if np.all(np.isnan(horn_inter)):  # Ray does not touch horn
                dest = sc_pnt + inf * sc_ref
                add_rz_ray_to_plot(ax, sc_pnt, dest, 'red', '-.', 'sc->$\infty$', sign)
            else:
                add_rz_ray_to_plot(ax, sc_pnt, horn_inter, 'yellow', '-.', 'Sc->Horn', sign)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax.legend(by_label.values(), by_label.keys())
    ax.set_aspect('equal')
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('Height [m]')
    ax.set_ylim([-0.5, 9.5])
    ax.set_xlim([-13, 13])
    ax.set_title('VLA Ray tracing 2D Schematic')
    close_figure(fig, title_from_input_parameters(rt_xds.attrs['input_parameters']), filename, 300, False)


######################################
# Master routine for the ray Tracing #
######################################
def vla_ray_tracing_pipeline(telescope_parameters, grid_size, grid_resolution, grid_unit,
                             x_pnt_off, y_pnt_off, pnt_off_unit, x_focus_off, y_focus_off, z_focus_off, focus_off_unit,
                             phase_offset, phase_unit, observing_wavelength, wavelength_unit, filename):
    input_pars = locals()
    del input_pars['telescope_parameters']

    # Convert user units and build proper RT inputs
    grid_fac = convert_unit(grid_unit, 'm', 'length')
    grid_size *= grid_fac
    grid_resolution *= grid_fac

    focus_fac = convert_unit(focus_off_unit, 'm', 'length')
    focus_offset = focus_fac * np.array([x_focus_off, y_focus_off, z_focus_off])

    pnt_fac = convert_unit(pnt_off_unit, 'rad', 'trigonometric')
    x_pnt_off *= pnt_fac
    y_pnt_off *= pnt_fac
    # Using small angles approximation here
    pnt_off = np.sqrt(x_pnt_off ** 2 + y_pnt_off ** 2)
    incident_light = np.array([np.sin(x_pnt_off), np.sin(y_pnt_off), -np.cos(pnt_off)])

    # Actual Ray Tracing starts here
    rt_xds = make_gridded_vla_primary(grid_size, grid_resolution, telescope_parameters)
    rt_xds = reflect_off_primary(rt_xds, incident_light)
    rt_xds = reflect_off_analytical_secondary(rt_xds, focus_offset)
    rt_xds = detect_light(rt_xds)
    rt_xds = compute_phase(rt_xds, observing_wavelength * convert_unit(wavelength_unit, 'm', 'length'),
                           phase_offset * convert_unit(phase_unit, 'rad', 'trigonometric'))

    rt_xds.attrs['input_parameters'] = input_pars

    rt_xds.to_zarr(filename, mode="w", compute=True, consolidated=True)
    return rt_xds


############################
# VLA Phase fitting plugin #
############################
def apply_vla_phase_fitting_to_xds(rt_xds, ntime=1, npol=1, nfreq=1):
    npnt = rt_xds.attrs['image_size']
    telescope_pars = rt_xds.attrs['telescope_parameters']
    x_axis = rt_xds['x_axis'].values
    y_axis = rt_xds['y_axis'].values

    shape_5d = [ntime, npol, nfreq, npnt, npnt]
    amplitude_5d = np.empty(shape_5d)
    phase_2d = regrid_data_onto_2d_grid(npnt, rt_xds['phase'].values, rt_xds['image_indexes'].values)
    phase_5d = np.empty_like(amplitude_5d)
    phase_5d[..., :, :] = phase_2d
    _, _, radius = create_coordinate_images(x_axis, y_axis)
    radial_mask = create_radial_mask(radius, telescope_pars['inner_radius'], telescope_pars['primary_diameter'] / 2)
    amplitude_5d[..., :, :] = np.where(radial_mask, 1.0, np.nan)


    # execute_phase_fitting(amplitude, phase, pol_axis, freq_axis, telescope, uv_cell_size, phase_fit_parameter,
    #                       to_stokes, is_near_field, focus_offset, uaxis, vaxis, label)
    print(data_statistics(phase_2d))
    print(data_statistics(phase_5d))
