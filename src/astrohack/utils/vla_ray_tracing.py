import numpy as np
from scipy.optimize import fsolve, newton, bisect

from astrohack.visualization.plot_tools import get_proper_color_map, create_figure_and_axes, well_positioned_colorbar, \
    close_figure, compute_extent, scatter_plot

vla_pars = {
    'primary_diameter': 25.0,
    'secondary_diameter': 2.5146,
    'focal_length': 9.0,
    'z_intercept':3.140,
    'foci_half_distance':3.662,
    'inner_radius': 2.0
}

vla_secondary = 2.5146

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


def normalize_vector_map(vector_map):
    normalization = np.linalg.norm(vector_map, axis=-1)
    return vector_map / normalization[..., np.newaxis]


def reflect_light(light, normals):
    return light - 2 * np.sum(light * normals, axis=-1)[..., np.newaxis] * normals


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


def make_2d(npnt, data, indexes):
    gridded_2d = np.full([npnt, npnt], np.nan)
    npnt_1d = data.shape[0]
    for ipnt in range(npnt_1d):
        ix, iy = indexes[ipnt]
        gridded_2d[ix, iy] = data[ipnt]
    return gridded_2d


def plot_rt_dict(rt_dict, key, title, filename, coord=None, colormap='viridis'):
    if coord is None:
        data = rt_dict[key]
    else:
        data = rt_dict[key][:, coord]
    gridded_array = make_2d(rt_dict['image_size'], data, rt_dict['pr_idx'])
    fig, ax = create_figure_and_axes([10, 8], [1, 1])
    cmap = get_proper_color_map(colormap)
    minmax = [np.nanmin(gridded_array), np.nanmax(gridded_array)]
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

    close_figure(fig, '', filename, 300, False)


def reflect_off_primary(rt_dict, incident_light):
    incident_light = normalize_vector_map(incident_light)
    primary_normals = rt_dict['pr_norm']
    light = np.zeros_like(primary_normals)
    light[:] = incident_light
    reflection = reflect_light(light, primary_normals)
    rt_dict['pr_ref'] = reflection
    return rt_dict


def secondary_hyperboloid_root_func(tval, fargs):
    pnt, ray, acoef, fcoef, ccoef = fargs
    newpnt = pnt + tval * ray
    rad2 = newpnt[0] ** 2 + newpnt[1] ** 2
    pntz = newpnt[2]
    value = fcoef - ccoef + acoef*np.sqrt(1 + rad2/(ccoef**2-acoef**2)) - pntz
    return value


def vla_2d_plot(pntzs, x_axis, y_axis, rays, primary_diameter=25, secondary_diameter=2.5146, focal_length=9.0, z_intercept=3.140, foci_half_distance=3.662, nrays=20):
    pr_rad = primary_diameter/2
    sc_rad = secondary_diameter/2
    radarr = np.arange(-pr_rad, pr_rad, primary_diameter/1e3)
    primary = radarr**2/4/focal_length
    secondary = focal_length - foci_half_distance + z_intercept*np.sqrt(1+radarr**2/(foci_half_distance**2-z_intercept**2))
    secondary = np.where(np.abs(radarr)<sc_rad, secondary, np.nan)
    fig, ax = create_figure_and_axes([10, 8], [1, 1])
    ax.plot(radarr, primary, color='red', label='primary')
    ax.plot(radarr, secondary, color='blue', label='secondary')
    ax.scatter([0], [focal_length], color='black', label='Primary focus')
    ax.scatter([0], [focal_length-2*foci_half_distance], color='red', label='Secondary focus')


    tparr = np.arange(0, 20, 0.1)
    npnt = pntzs.shape[0]
    ipnt = 0
    iy = npnt//2
    while ipnt < nrays:
        ix = np.random.randint(0, high=npnt)
        if np.isnan(pntzs[ix, iy]):
            ipnt -= 1
            continue
        else:
            point = [x_axis[ix], pntzs[ix, iy]]
            ray = [rays[ix, iy, 0], rays[ix, iy, 2]]
            tracer = point + tparr[:, np.newaxis]*ray
            ax.plot(tracer[:,0], tracer[:,1], color='green', label='Rays')
            ipnt += 1

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    #ax.legend()
    # scatter_plot(ax, radarr, 'Radius', primary, 'Height', model=secondary, data_label='Primary', model_label='Secondary', plot_residuals=False)
    ax.set_aspect('equal')
    close_figure(fig, '', f'vla-analytical-model.png', 300, False)


def reflect_off_analytical_secondary(rt_dict, telescope_pars):
    pr_points = rt_dict['pr_pnt']
    pr_refle = rt_dict['pr_ref']

    # this is simply 1D
    distance_to_secondary = np.empty_like(pr_points[:, 0])

    fargs = [None, None, telescope_pars['z_intercept'], telescope_pars['focal_length'],
             telescope_pars['foci_half_distance']]

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
    secondary_normals = np.cross(x_grad, y_grad)
    secondary_reflections = reflect_light(pr_refle, secondary_normals)

    rt_dict['dist_pr_sc'] = distance_to_secondary
    rt_dict['sc_pnt'] = secondary_pnt
    rt_dict['sc_norm'] = secondary_normals
    rt_dict['sc_ref'] = secondary_reflections

    return rt_dict




def solve_bisection(func, minbound, maxbound, args, tol=1e-8, maxit=100):
    searching = True
    vminbound = func(minbound, args)
    vmaxbound = func(maxbound, args)
    nit = 0
    # solution is within bounds
    if vmaxbound*vminbound < 0:
        while searching:
            newguess = (maxbound + minbound) / 2
            func_val = func(newguess, args)
            if abs(func_val) < tol:
                #print(newguess)
                return newguess
            elif vmaxbound*func_val < 0:
                minbound = newguess
            else:
                maxbound = newguess
            nit += 1
            if nit > maxit:
                return np.nan
    else:
        return np.nan










