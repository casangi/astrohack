import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import xarray as xr
import toolviper.utils.logger as logger

from astrohack import overwrite_file
from astrohack.utils import data_statistics, clight, statistics_to_text
from astrohack.utils.constants import twopi
from astrohack.utils.conversion import convert_unit
from astrohack.utils.algorithms import phase_wrapping
from astrohack.utils.ray_tracing_general import generalized_dot, generalized_norm, normalize_vector_map, reflect_light
from astrohack.visualization.plot_tools import get_proper_color_map, create_figure_and_axes, well_positioned_colorbar, \
    close_figure, compute_extent
from astrohack.visualization.textual_data import create_pretty_table
from astrohack.utils.text import format_value_error, format_label

nanvec3d = np.full([3], np.nan)


####################
# Data IO routines #
####################
def open_rt_zarr(zarr_filename):
    try:
        rt_xds = xr.open_zarr(zarr_filename)
        return rt_xds
    except Exception as error:
        logger.error(f"There was an exception opening {zarr_filename}: {error}")


def write_rt_xds_to_zarr(rt_xds, zarr_filename, overwrite):
    overwrite_file(zarr_filename, overwrite)
    rt_xds.to_zarr(zarr_filename, mode="w", compute=True, consolidated=True)


######################################################################
# Setup routines and Mathematical description of the secondary shape #
######################################################################
def _simple_axis(minmax, resolution, margin=0.05):
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
    axis = _simple_axis(grid_minmax, resolution, margin=0.0)
    image_size = axis.shape[0]
    axis_idx = np.arange(image_size, dtype=int)

    # It is imperative to put indexing='ij' so that the x and Y axes are not flipped in this step.
    x_mesh, y_mesh, img_radius = create_coordinate_images(axis, axis)
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


def _secondary_hyperboloid_root_func(tval, fargs):
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
        val, _, ier, _ = fsolve(_secondary_hyperboloid_root_func, initial_guess, args=fargs, maxfev=100,
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
    title = ''
    title += (f'Pointing offset = ({inpt_dict["x_pointing_offset"]}, {inpt_dict["y_pointing_offset"]}) '
              f'[{inpt_dict["pointing_offset_unit"]}], ')
    title += (f'Focus offset = ({inpt_dict["x_focus_offset"]}, {inpt_dict["y_focus_offset"]}, '
              f'{inpt_dict["z_focus_offset"]}) [{inpt_dict["focus_offset_unit"]}], ')
    title += f'Phase offset = {inpt_dict["phase_offset"]} [{inpt_dict["phase_unit"]}], '
    title += f'$\lambda$ = {inpt_dict["observing_wavelength"]} [{inpt_dict["wavelength_unit"]}]'
    return title


def _imshow_2d_map(ax, fig, gridded_array, title, extent, zlabel, colormap, inner_radius, outer_radius, zlim):
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


def plot_2d_map(gridded_array, axis, telescope_parameters, suptitle, filename, zlabel, colormap, zlim, display, dpi):
    inner_radius = telescope_parameters['inner_radius']
    outer_radius = telescope_parameters['primary_diameter'] / 2
    axes = ['X', 'Y', 'Z']
    extent = compute_extent(axis, axis, margin=0.0)

    if len(gridded_array.shape) == 3:  # XYZ Plots
        fig, ax = create_figure_and_axes([20, 8], [1, 3])
        for iax in range(3):
            _imshow_2d_map(ax[iax], fig, gridded_array[..., iax], f'{axes[iax]} Component', extent, zlabel, colormap,
                           inner_radius, outer_radius, zlim)
    else:
        fig, ax = create_figure_and_axes([10, 8], [1, 1])
        _imshow_2d_map(ax, fig, gridded_array, None, extent, zlabel, colormap, inner_radius, outer_radius, zlim)

    close_figure(fig, suptitle, filename, dpi, display)


def add_rz_ray_to_plot(ax, origin, destiny, color, ls, label, sign):
    radcoord = [sign * generalized_norm(origin[0:2]), sign * generalized_norm(destiny[0:2])]
    zcoord = [origin[2], destiny[2]]
    ax.plot(radcoord, zcoord, color=color, label=label, ls=ls)


def compare_ray_tracing_to_phase_fit_results(rt_xds, phase_fit_results, phase_5d, phase_corrected_angle, filename,
                                             phase_unit='deg', colormap='viridis', display=False, dpi=300):
    xds_inp = rt_xds.attrs['input_parameters']
    angle_unit = xds_inp['pointing_offset_unit']
    length_unit = xds_inp['focus_offset_unit']
    field_names = ['Parameter', 'Value', 'Reference', 'Difference', 'unit']
    alignment = 'c'
    outstr = ''
    wavelength = xds_inp['observing_wavelength']*convert_unit(xds_inp['wavelength_unit'], 'm', 'length')
    valid_pars = ['phase_offset', 'x_point_offset', 'y_point_offset', 'x_focus_offset', 'y_focus_offset',
                  'z_focus_offset']
    unit_types = ['trigonometric', 'trigonometric', 'trigonometric', 'length', 'length', 'length']
    units = ['deg', angle_unit, angle_unit, length_unit, length_unit, length_unit]
    reference_values = [0.0, xds_inp['x_pointing_offset'], xds_inp['y_pointing_offset'],
                        xds_inp['x_focus_offset'], xds_inp['y_focus_offset'], xds_inp['z_focus_offset']]

    outstr += 'Comparison between input and fitted values'
    freq = clight/wavelength
    cropped_dict = phase_fit_results['map_0'][freq]['I']
    table = create_pretty_table(field_names, alignment)
    for ip, par_name in enumerate(valid_pars):
        item = cropped_dict[par_name]
        val = item['value']
        err = item['error']
        unitin = item['unit']
        ref = reference_values[ip]
        fac = convert_unit(unitin, units[ip], unit_types[ip])
        val *= fac
        err *= fac
        diff = val-ref
        row = [format_label(par_name), format_value_error(val, err, 1.0, 1e-4), f'{ref}', f'{diff}', units[ip]]
        table.add_row(row)

    outstr += table.get_string() + '\n\n'
    print(outstr)

    phase_2d = phase_5d[0, 0, 0]
    residuals_2d = phase_corrected_angle[0, 0, 0]
    correction = residuals_2d - phase_2d

    axis = rt_xds['x_axis'].values
    extent = compute_extent(axis, axis, margin=0.0)
    telescope_parameters = rt_xds.attrs['telescope_parameters']
    inner_radius = telescope_parameters['inner_radius']
    outer_radius = telescope_parameters['primary_diameter'] / 2
    fac = convert_unit('rad', phase_unit, 'trigonometric')
    zlim = [-np.pi*fac, np.pi*fac]

    fig, ax = create_figure_and_axes([20, 8], [1, 3])
    statkeys = ['mean', 'median', 'rms']

    _imshow_2d_map(ax[0], fig, fac * phase_2d, f'RT phase model\n{statistics_to_text(data_statistics(fac * phase_2d), statkeys)}',
                   extent, f'Phase [{phase_unit}]', colormap, inner_radius, outer_radius, zlim)
    _imshow_2d_map(ax[1], fig, fac * correction, f'Fitted correction', extent, f'Phase [{phase_unit}]', colormap,
                   inner_radius, outer_radius, zlim)
    _imshow_2d_map(ax[2], fig, fac * residuals_2d, f'Residuals\n{statistics_to_text(data_statistics(fac * residuals_2d), statkeys)}',
                   extent, f'Phase [{phase_unit}]', colormap, inner_radius, outer_radius, zlim)
    close_figure(fig, 'Cassegrain RT model fitting for \n'+title_from_input_parameters(rt_xds.attrs['input_parameters']),
                 filename, dpi, display)
