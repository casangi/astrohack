import numpy as np
from scipy.optimize import fsolve

from astrohack.visualization.plot_tools import get_proper_color_map, create_figure_and_axes, well_positioned_colorbar, \
    close_figure, compute_extent, scatter_plot


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


def create_radial_mask(radius, inner_rad, outer_rad):
    mask = np.full_like(radius, 1.0)
    mask = np.where(radius > outer_rad, np.nan, mask)
    mask = np.where(radius < inner_rad, np.nan, mask)
    return mask


def make_gridded_vla_primary(grid_size, resolution, antena_radius=12.5, inner_radius=2, focal_length=9.0):
    grid_minmax = [-grid_size, grid_size]
    x_axis = simple_axis(grid_minmax, resolution, margin=0.0)
    y_axis = simple_axis(grid_minmax, resolution, margin=0.0)
    npnt = x_axis.shape[0]
    vec_shape = [npnt, npnt, 3]

    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
    img_radius = np.sqrt(x_mesh**2+y_mesh**2)
    radial_mask = create_radial_mask(img_radius, inner_radius, antena_radius)
    img_radius *= radial_mask

    # Parabola formula = (x**2 + y**2)/4/focal_length
    gridded_primary = img_radius**2/4/focal_length
    x_grad = np.zeros(vec_shape)
    y_grad = np.zeros(vec_shape)
    x_grad[:, :, 0] = 1.0
    x_grad[:, :, 2] = 2 * x_mesh / 4 / focal_length
    y_grad[:, :, 1] = 1.0
    y_grad[:, :, 2] = 2 * y_mesh / 4 / focal_length

    primary_normals = np.cross(x_grad, y_grad)
    primary_normals *= radial_mask[..., np.newaxis]
    primary_normals = normalize_vector_map(primary_normals)

    return gridded_primary, primary_normals, x_axis, y_axis


def simple_im_plot(title, gridded_array, x_axis, y_axis, filename, colormap='viridis'):
    fig, ax = create_figure_and_axes([10, 8], [1, 1])
    cmap = get_proper_color_map(colormap)
    minmax = [np.nanmin(gridded_array), np.nanmax(gridded_array)]
    fsize = 10
    ax.set_title(title, size=1.5 * fsize)
    extent = compute_extent(x_axis, y_axis, margin=0.0)
    im = ax.imshow(gridded_array, cmap=cmap, extent=extent, interpolation="nearest", vmin=minmax[0], vmax=minmax[1],
                   origin='lower')
    well_positioned_colorbar(ax, fig, im, "Z Scale")
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xlabel("X axis [m]")
    ax.set_ylabel("Y axis [m]")

    close_figure(fig, '', filename, 300, False)


def reflect_off_primary(primary_normals, incident_light):
    incident_light = normalize_vector_map(incident_light)
    light = np.zeros_like(primary_normals)
    light[:] = np.array(incident_light)
    # np.sum in axis 2 is a another way of expressing the dot product
    reflection = light - 2 * np.sum(light * primary_normals, axis=2)[..., np.newaxis] * primary_normals
    return reflection


def reflect_off_analytical_secondary(primary_grided, x_axis, y_axis, primary_reflections, offset, focal_length=9.0,
                                     z_intercept=3.140, foci_half_distance=3.662):

    def fit_func(tval, fargs):
        pnt, ref, acoef, fcoef, ccoeff = fargs
        newpnt = pnt+tval*ref
        value = fcoef - ccoeff + acoef*np.sqrt(1 + (newpnt[0]**2+newpnt[1]**2)/(ccoeff**2-acoef**2)) - newpnt[2]
        return value

    solved_t = np.empty_like(primary_grided)
    npnt = primary_grided.shape[0]
    for ix in range(npnt):
        px = x_axis[ix]
        for iy in range(npnt):
            py = y_axis[iy]
            pz = primary_grided[ix, iy]
            if np.isnan(pz):
                solved_t[ix, iy] = np.nan
            else:
                args = [[px, py, pz], primary_reflections[ix, iy], z_intercept, focal_length, foci_half_distance]
                # using scipy fsolve
                # val, _, ier, _ = fsolve(fit_func, 1, args=args, maxfev=100, full_output=True, xtol=1e-6)
                # if ier == 1:
                #     solved_t[ix, iy] = val
                # else:
                #     solved_t[ix, iy] = np.nan
                solved_t[ix, iy] = solve_bisection(fit_func, 0, 1e3, args, tol=1e-6, maxit=100)

    return solved_t


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










