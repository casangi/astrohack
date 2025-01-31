import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

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
    'z_intercept':3.140,
    'foci_half_distance':3.662,
    'inner_radius': 2.0,
    # Assuming a 10 cm Horn for now
    'horn_diameter': 0.2,
    # Assumed to be at the Secondary focus i.e.: f - 2c
    'horn_position': np.array([0, 0, 9.0 - 2 * 3.662]),
    # Horn looks straight up
    'horn_orientation': np.array([0, 0, 1])
}

nanvec3d = np.full([3], np.nan)

######################################################################
# Setup routines and Mathematical description of the secondary shape #
######################################################################
def simple_axis(minmax, resolution, margin=0.05):
    mini, maxi = minmax
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


def create_radial_mask(radius, inner_rad, outer_rad):
    mask = np.full_like(radius, True, dtype=bool)
    mask = np.where(radius > outer_rad, False, mask)
    mask = np.where(radius < inner_rad, False, mask)
    return mask


def make_gridded_vla_primary(grid_size, resolution, telescope_pars):
    grid_minmax = [-grid_size/2, grid_size/2]
    axis = simple_axis(grid_minmax, resolution, margin=0.0)
    image_size = axis.shape[0]
    axis_idx = np.arange(image_size, dtype=int)

    # It is imperative to put indexing='ij' so that the x and Y axes are not flipped in this step.
    x_mesh, y_mesh = np.meshgrid(axis, axis,indexing='ij')
    x_idx_mesh, y_idx_mesh = np.meshgrid(axis_idx, axis_idx, indexing='ij')
    img_radius = np.sqrt(x_mesh**2+y_mesh**2)
    radial_mask = create_radial_mask(img_radius, telescope_pars['inner_radius'], telescope_pars['primary_diameter']/2)
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
    gridded_primary = img_radius**2/4/focal_length
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
    ray_tracing_dict = {
        'pr_pnt': primary_points,
        'pr_norm': primary_normals,
        'pr_idx': idx_1d,
        'image_size': image_size,
        'axis': axis,
        'radius': img_radius,
        'npnt_1d': npnt_1d,
    }
    return ray_tracing_dict


def secondary_hyperboloid_root_func(tval, fargs):
    pnt, ray, acoef, fcoef, ccoef, offsets = fargs
    # The offset is a simple displacement of the secondary
    newpnt = (pnt + tval * ray) - offsets
    rad2 = newpnt[0] ** 2 + newpnt[1] ** 2
    pntz = newpnt[2]
    value = fcoef - ccoef + acoef*np.sqrt(1 + rad2/(ccoef**2-acoef**2)) - pntz
    return value

##########################################################
# Actual ray tracing steps in order of light propagation #
##########################################################
def reflect_off_primary(rt_dict, incident_light):
    incident_light = normalize_vector_map(incident_light)
    primary_normals = rt_dict['pr_norm']
    light = np.zeros_like(primary_normals)
    light[:] = incident_light
    reflection = reflect_light(light, primary_normals)
    rt_dict['pr_ref'] = reflection
    rt_dict['light'] = light
    return rt_dict


def reflect_off_analytical_secondary(rt_dict, telescope_pars, offset=np.array((0, 0, 0))):
    pr_points = rt_dict['pr_pnt']
    pr_refle = rt_dict['pr_ref']

    # this is simply 1D
    distance_to_secondary = np.empty_like(pr_points[:, 0])

    fargs = [None, None, telescope_pars['z_intercept'], telescope_pars['focal_length'],
             telescope_pars['foci_half_distance'], offset]

    for ipnt in range(rt_dict['npnt_1d']):
        fargs[0] = pr_points[ipnt]
        fargs[1] = pr_refle[ipnt]
        # Focal length plus the height of departing point (distance from point to primary focus)
        initial_guess = telescope_pars['focal_length'] + pr_points[ipnt][2]
        val, _, ier, _ = fsolve(secondary_hyperboloid_root_func, initial_guess, args=fargs, maxfev=100, full_output=True, xtol=1e-8)
        if ier == 1:
            distance_to_secondary[ipnt] = val
        else:
            distance_to_secondary[ipnt] = np.nan

    secondary_pnt = pr_points + distance_to_secondary[..., np.newaxis] * pr_refle
    # Compute Gradients to compute normals at touched points
    x_grad = np.zeros_like(pr_points)
    y_grad = np.zeros_like(pr_points)
    dcoeff = telescope_pars['foci_half_distance']**2 - telescope_pars['z_intercept']**2
    px, py = secondary_pnt[:, 0], secondary_pnt[:, 1]
    root_term = telescope_pars['z_intercept'] /(dcoeff *  np.sqrt(1 + (px**2 + py**2)/dcoeff))
    x_grad[:, 0] = 1.0
    y_grad[:, 1] = 1.0
    x_grad[:, 2] = px * root_term
    y_grad[:, 2] = py * root_term
    secondary_normals = normalize_vector_map(np.cross(x_grad, y_grad))
    secondary_reflections = reflect_light(pr_refle, secondary_normals)

    rt_dict['dist_pr_sc'] = distance_to_secondary
    rt_dict['sc_pnt'] = secondary_pnt
    rt_dict['sc_norm'] = secondary_normals
    rt_dict['sc_ref'] = secondary_reflections

    return rt_dict


def detect_light(rt_dict, telescope_pars):
    sc_reflec = rt_dict['sc_ref']
    sc_pnt = rt_dict['sc_pnt']
    horn_orientation = np.empty_like(sc_reflec)
    horn_position = np.empty_like(sc_reflec)
    horn_orientation[:] = telescope_pars['horn_orientation']
    horn_position[:] = telescope_pars['horn_position']
    horn_diameter = telescope_pars['horn_diameter']


    distance_secondary_to_horn = (generalized_dot((horn_position - sc_pnt), horn_orientation) /
                                  generalized_dot(sc_reflec, horn_orientation))
    horn_intercept = sc_pnt + distance_secondary_to_horn[..., np.newaxis] * sc_reflec
    distance_to_horn_center = generalized_norm(horn_intercept - horn_position)


    selection = distance_to_horn_center > horn_diameter
    horn_intercept[selection, :] = nanvec3d


    rt_dict['dist_sc_horn'] = distance_secondary_to_horn
    rt_dict['horn_intercept'] = horn_intercept
    return rt_dict


def compute_phase(rt_dict, wavelength, phase_offset):
    incident_light = rt_dict['light']
    pr_pntz = rt_dict['pr_pnt'][:, 2]
    distance_pr_horn = rt_dict['dist_sc_horn'] + rt_dict['dist_pr_sc']
    maxheight = np.max(pr_pntz)
    boresight = np.empty_like(incident_light)
    boresight[:] = [0, 0, -1] # strictly vertical
    cosbeta = generalized_dot(boresight, incident_light)
    path_diff_before_dish = (maxheight-pr_pntz)/cosbeta
    total_path = np.where(np.isnan(rt_dict['horn_intercept'][:, 0]), np.nan, distance_pr_horn + path_diff_before_dish)


    wavenumber = total_path/wavelength
    phase = phase_wrapping(twopi * wavenumber + phase_offset)

    rt_dict['total_path'] = total_path
    rt_dict['phase'] = phase
    return rt_dict

###########################################################
# Plotting routines and plotting aids, such as regridding #
###########################################################
def make_2d(npnt, data, indexes):
    gridded_2d = np.full([npnt, npnt], np.nan)
    npnt_1d = data.shape[0]
    for ipnt in range(npnt_1d):
        ix, iy = indexes[ipnt]
        gridded_2d[ix, iy] = data[ipnt]
    return gridded_2d


def plot_rt_dict(rt_dict, key, telescope_pars, title, filename, coord=None, colormap='viridis', zlim=None):
    if coord is None:
        data = rt_dict[key]
    else:
        data = rt_dict[key][:, coord]
    gridded_array = make_2d(rt_dict['image_size'], data, rt_dict['pr_idx'])
    fig, ax = create_figure_and_axes([10, 8], [1, 1])
    cmap = get_proper_color_map(colormap)
    if zlim is None:
        minmax = [np.nanmin(gridded_array), np.nanmax(gridded_array)]
    else:
        minmax = zlim
    fsize = 10
    ax.set_title(title, size=1.5 * fsize)
    extent = compute_extent(rt_dict['axis'], rt_dict['axis'], margin=0.0)
    im = ax.imshow(gridded_array.T, cmap=cmap, extent=extent, interpolation="nearest", vmin=minmax[0], vmax=minmax[1],
                   origin='lower')
    well_positioned_colorbar(ax, fig, im, "Z Scale")
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xlabel("X axis [m]")
    ax.set_ylabel("Y axis [m]")

    innerring = plt.Circle((0, 0), telescope_pars['inner_radius'], color='black', fill=None)
    outerring = plt.Circle((0, 0), telescope_pars['primary_diameter']/2, color='black', fill=None)
    ax.add_patch(outerring)
    ax.add_patch(innerring)

    close_figure(fig, '', filename, 300, False)


def add_rz_ray_to_plot(ax, origin, destiny, color, ls, label, sign):
    radcoord = [sign * generalized_norm(origin[0:2]), sign * generalized_norm(destiny[0:2])]
    zcoord = [origin[2], destiny[2]]
    ax.plot(radcoord, zcoord, color=color, label=label, ls=ls)


def vla_2d_plot(rt_dict, telescope_pars, nrays=20):
    primary_diameter = telescope_pars['primary_diameter']
    secondary_diameter = telescope_pars['secondary_diameter']
    focal_length = telescope_pars['focal_length']
    foci_half_distance = telescope_pars['foci_half_distance']
    z_intercept = telescope_pars['z_intercept']
    pr_rad = primary_diameter/2
    sc_rad = secondary_diameter/2
    radarr = np.arange(-pr_rad, pr_rad, primary_diameter/1e3)
    primary = radarr**2/4/focal_length
    secondary = focal_length - foci_half_distance + z_intercept*np.sqrt(1+radarr**2/(foci_half_distance**2-z_intercept**2))
    secondary = np.where(np.abs(radarr)<sc_rad, secondary, np.nan)
    fig, ax = create_figure_and_axes([10, 8], [1, 1])
    ax.plot(radarr, primary, color='black', label='Pr mirror')
    ax.plot(radarr, secondary, color='blue', label='Sc mirror')
    ax.scatter([0], [focal_length], color='black', label='Pr focus')
    ax.scatter([0], [focal_length-2*foci_half_distance], color='blue', label='Sc focus')

    pr_pnts = rt_dict['pr_pnt']
    sc_pnts = rt_dict['sc_pnt']
    horn_inters = rt_dict['horn_intercept']
    incomings = rt_dict['light']
    sc_refs = rt_dict['sc_ref']
    pr_refs = rt_dict['pr_ref']

    npnt = pr_pnts.shape[0]
    sign = -1
    inf = 1e3
    for isamp in range(nrays):
        sign *= -1
        ipnt = np.random.randint(0, high=npnt)

        # Data Selection
        sc_pnt = sc_pnts[ipnt]
        pr_pnt = pr_pnts[ipnt]
        pr_ref = pr_refs[ipnt]
        sc_ref = sc_refs[ipnt]
        horn_inter = horn_inters[ipnt]
        incoming = incomings[ipnt]

        # Plot incident light
        origin = pr_pnt -inf*incoming
        add_rz_ray_to_plot(ax, origin, pr_pnt, 'yellow','-',  '$\infty$->Pr', sign)

        # Plot primary reflection
        if np.all(np.isnan(sc_pnt)): # Ray does not touch secondary
            dest = pr_pnt + inf*pr_ref
            add_rz_ray_to_plot(ax, pr_pnt, dest, 'red', '--', 'Pr->$\infty$', sign)
        else:
            add_rz_ray_to_plot(ax, pr_pnt, sc_pnt, 'yellow', '--', 'Pr->Sc', sign)

            # Plot secondary reflection
            if np.all(np.isnan(horn_inter)): # Ray does not touch horn
                dest = sc_pnt + inf*sc_ref
                add_rz_ray_to_plot(ax, sc_pnt, dest, 'red', '-.', 'sc->$\infty$', sign)
            else:
                add_rz_ray_to_plot(ax, sc_pnt, horn_inter,'yellow', '-.', 'Sc->Horn', sign)



    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax.legend(by_label.values(), by_label.keys())
    ax.set_aspect('equal')
    ax.set_xlabel('Radius [m]')
    ax.set_ylabel('Height [m]')
    ax.set_ylim([-0.5, 9.5])
    ax.set_xlim([-13, 13])
    close_figure(fig, '', f'vla-analytical-model.png', 300, False)

######################################
# Master routine for the ray Tracing #
######################################
def vla_ray_tracing_pipeline(telescope_parameters, grid_size, grid_resolution, grid_unit,
                             x_pnt_off, y_pnt_off, pnt_off_unit, x_focus_off, y_focus_off, z_focus_off, focus_off_unit,
                             phase_offset, phase_unit, observing_wavelength, wavelength_unit):

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
    pnt_off = np.sqrt(x_pnt_off**2 + y_pnt_off**2)
    incident_light = np.array([np.sin(x_pnt_off), np.sin(y_pnt_off), -np.cos(pnt_off)])
    print(incident_light, generalized_norm(incident_light))


    # Actual Ray Tracing starts here
    rt_dict = make_gridded_vla_primary(grid_size, grid_resolution, telescope_parameters)
    rt_dict = reflect_off_primary(rt_dict, incident_light)
    rt_dict = reflect_off_analytical_secondary(rt_dict, vla_pars, focus_offset)
    rt_dict = detect_light(rt_dict, vla_pars)
    rt_dict = compute_phase(rt_dict, observing_wavelength*convert_unit(wavelength_unit, 'm', 'length'),
                            phase_offset*convert_unit(phase_unit, 'rad', 'trigonometric'))

    return rt_dict


