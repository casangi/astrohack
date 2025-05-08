import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import xarray as xr
import toolviper.utils.logger as logger

from astrohack import overwrite_file
from astrohack.utils import (
    data_statistics,
    clight,
    statistics_to_text,
    create_aperture_mask,
)
from astrohack.utils.constants import twopi
from astrohack.utils.conversion import convert_unit
from astrohack.utils.algorithms import phase_wrapping, create_coordinate_images
from astrohack.utils.ray_tracing_general import (
    generalized_dot,
    generalized_norm,
    normalize_vector_map,
    reflect_light,
    simple_axis,
)
from astrohack.visualization.plot_tools import (
    get_proper_color_map,
    create_figure_and_axes,
    well_positioned_colorbar,
    close_figure,
    compute_extent,
)
from astrohack.visualization.textual_data import create_pretty_table
from astrohack.utils.text import format_value_error, format_label

nanvec3d = np.full([3], np.nan)


####################
# Data IO routines #
####################
def open_rt_zarr(zarr_filename):
    """
    Open a Zarr file containing the results of a run of the Ray Tracing pipeline
    Args:
        zarr_filename: Name of the Zarr file containing the RT data

    Returns:
        The Xarray dataset containing the results of a run of the Ray Tracing pipeline
    """
    try:
        rt_xds = xr.open_zarr(zarr_filename)
        return rt_xds
    except Exception as error:
        logger.error(f"There was an exception opening {zarr_filename}: {error}")


def write_rt_xds_to_zarr(rt_xds, zarr_filename, overwrite):
    """
    Write a Xarray dataset containing the results of a run of the Ray Tracing pipeline to disk
    Args:
        rt_xds: Xarray dataset containing the results of a run of the Ray Tracing pipeline
        zarr_filename: Name of the Zarr file containing the RT data
        overwrite: Overwrite file if it already exists?
    """
    overwrite_file(zarr_filename, overwrite)
    rt_xds.to_zarr(zarr_filename, mode="w", compute=True, consolidated=True)


######################################################################
# Setup routines and Mathematical description of the secondary shape #
######################################################################
def make_gridded_cassegrain_primary(grid_size, resolution, telescope_pars):
    """
    Create a 1D representation of the primary and the normals to its surface based on a radial mask
    Args:
        grid_size: The span of the grid to used
        resolution: The spacing between points in the grid
        telescope_pars: The optical parameters of the telescope in question

    Returns:
        An Xarray dataset containing the basics for regridding the 1D data, the primary points and normals plus
        the x and y axes
    """
    grid_minmax = [-grid_size / 2, grid_size / 2]
    axis = simple_axis(grid_minmax, resolution, margin=0.0)
    image_size = axis.shape[0]
    axis_idx = np.arange(image_size, dtype=int)

    # It is imperative to put indexing='ij' so that the x and Y axes are not flipped in this step.
    x_mesh, y_mesh, img_radius, _ = create_coordinate_images(
        axis, axis, create_polar_coordinates=True
    )
    x_idx_mesh, y_idx_mesh = np.meshgrid(axis_idx, axis_idx, indexing="ij")
    radial_mask = create_aperture_mask(
        axis,
        axis,
        telescope_pars["inner_radius"],
        telescope_pars["primary_diameter"] / 2,
    )
    img_radius = img_radius[radial_mask]
    npnt_1d = img_radius.shape[0]
    idx_1d = np.empty([npnt_1d, 2], dtype=int)
    idx_1d[:, 0] = x_idx_mesh[radial_mask]
    idx_1d[:, 1] = y_idx_mesh[radial_mask]
    x_mesh_1d = x_mesh[radial_mask]
    y_mesh_1d = y_mesh[radial_mask]

    vec_shape = [npnt_1d, 3]
    focal_length = telescope_pars["focal_length"]
    # Parabola formula = (x**2 + y**2)/4/focal_length
    gridded_primary = img_radius**2 / 4 / focal_length
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

    rt_xds.attrs["image_size"] = image_size
    rt_xds.attrs["npnt_1d"] = npnt_1d
    rt_xds.attrs["telescope_parameters"] = telescope_pars

    rt_xds["primary_points"] = xr.DataArray(primary_points, dims=["points", "xyz"])
    rt_xds["primary_normals"] = xr.DataArray(primary_normals, dims=["points", "vxyz"])
    rt_xds["image_indexes"] = xr.DataArray(idx_1d, dims=["points", "idx"])
    rt_xds["x_axis"] = xr.DataArray(axis, dims=["x"])
    rt_xds["y_axis"] = xr.DataArray(axis, dims=["y"])

    return rt_xds


def _secondary_hyperboloid_root_func(tval, fargs):
    """
    Simple function whose root is the point at which a ray and a hyperboloid cross each other
    Args:
        tval: Distance along ray
        fargs: optical parameters and ray origin and direction

    Returns:
        The value of the function hyperboloid-pnt
    """
    pnt, ray, acoef, fcoef, ccoef, offsets = fargs
    # The offset is a simple displacement of the secondary
    newpnt = (pnt + tval * ray) - offsets
    rad2 = newpnt[0] ** 2 + newpnt[1] ** 2
    pntz = newpnt[2]
    value = fcoef - ccoef + acoef * np.sqrt(1 + rad2 / (ccoef**2 - acoef**2)) - pntz
    return value


##########################################################
# Actual ray tracing steps in order of light propagation #
##########################################################
def reflect_off_primary(rt_xds, incident_light):
    """
    Reflect incident light onto primary surface
    Args:
        rt_xds: Ray tracing Xarray dataset with primary normals and points
        incident_light: 3D vector with the direction of incident light

    Returns:
        Ray tracing XDS enriched with the incident light and the reflections of the primary mirror
    """
    incident_light = normalize_vector_map(incident_light)
    primary_normals = rt_xds["primary_normals"].values
    light = np.zeros_like(primary_normals)
    light[:] = incident_light
    reflection = reflect_light(light, primary_normals)
    rt_xds["primary_reflections"] = xr.DataArray(reflection, dims=["points", "vxyz"])
    rt_xds["light"] = xr.DataArray(light, dims=["points", "vxyz"])
    return rt_xds


def reflect_off_analytical_secondary(rt_xds, offset=np.array((0, 0, 0))):
    """
    Compute reflections off of the secundary using an analutical description of the secondary
    Args:
        rt_xds: Ray tracing XDS containing primary points and their reflections
        offset: An offset in meters to be applied to the position of the secondary mirror

    Returns:
        Ray tracing XDS enriched with the distance along the rays to the secondary, the points touched in the secondary,
         the normals at these points and the reflections at each of these points.
    """
    primary_points = rt_xds["primary_points"].values
    primary_reflections = rt_xds["primary_reflections"].values
    telescope_pars = rt_xds.attrs["telescope_parameters"]

    # this is simply 1D
    distance_to_secondary = np.empty_like(primary_points[:, 0])

    fargs = [
        None,
        None,
        telescope_pars["z_intercept"],
        telescope_pars["focal_length"],
        telescope_pars["foci_half_distance"],
        offset,
    ]

    for ipnt in range(rt_xds.attrs["npnt_1d"]):
        fargs[0] = primary_points[ipnt]
        fargs[1] = primary_reflections[ipnt]
        # Focal length plus the height of departing point (distance from point to primary focus)
        initial_guess = telescope_pars["focal_length"] + primary_points[ipnt][2]
        val, _, ier, _ = fsolve(
            _secondary_hyperboloid_root_func,
            initial_guess,
            args=fargs,
            maxfev=100,
            full_output=True,
            xtol=1e-8,
        )
        if ier == 1:
            distance_to_secondary[ipnt] = val
        else:
            distance_to_secondary[ipnt] = np.nan

    secondary_points = (
        primary_points + distance_to_secondary[..., np.newaxis] * primary_reflections
    )
    # Compute Gradients to compute normals at touched points
    x_grad = np.zeros_like(primary_points)
    y_grad = np.zeros_like(primary_points)
    dcoeff = (
        telescope_pars["foci_half_distance"] ** 2 - telescope_pars["z_intercept"] ** 2
    )
    px, py = secondary_points[:, 0], secondary_points[:, 1]
    root_term = telescope_pars["z_intercept"] / (
        dcoeff * np.sqrt(1 + (px**2 + py**2) / dcoeff)
    )
    x_grad[:, 0] = 1.0
    y_grad[:, 1] = 1.0
    x_grad[:, 2] = px * root_term
    y_grad[:, 2] = py * root_term
    secondary_normals = normalize_vector_map(np.cross(x_grad, y_grad))
    secondary_reflections = reflect_light(primary_reflections, secondary_normals)

    rt_xds["distance_primary_to_secondary"] = xr.DataArray(
        distance_to_secondary, dims=["points"]
    )
    rt_xds["secondary_points"] = xr.DataArray(secondary_points, dims=["points", "xyz"])
    rt_xds["secondary_normals"] = xr.DataArray(
        secondary_normals, dims=["points", "vxyz"]
    )
    rt_xds["secondary_reflections"] = xr.DataArray(
        secondary_reflections, dims=["points", "vxyz"]
    )

    return rt_xds


def detect_light(rt_xds):
    """
    Determines which rays touch the mouth of the horn
    Args:
        rt_xds: The ray tracing XDS containing the description of the rays from the primary up to their reflection
         from the secondary

    Returns:
        Ray tracing XDS enriched with the distance along the rays to the horn and the point at which they intercept
        the horn mouth
    """
    secondary_reflections = rt_xds["secondary_reflections"].values
    secondary_points = rt_xds["secondary_points"].values
    telescope_pars = rt_xds.attrs["telescope_parameters"]

    horn_orientation = np.empty_like(secondary_reflections)
    horn_position = np.empty_like(secondary_reflections)
    horn_orientation[:] = telescope_pars["horn_orientation"]
    horn_position[:] = telescope_pars["horn_position"]
    horn_diameter = telescope_pars["horn_diameter"]

    distance_secondary_to_horn = generalized_dot(
        (horn_position - secondary_points), horn_orientation
    ) / generalized_dot(secondary_reflections, horn_orientation)
    horn_intercept = (
        secondary_points
        + distance_secondary_to_horn[..., np.newaxis] * secondary_reflections
    )
    distance_to_horn_center = generalized_norm(horn_intercept - horn_position)

    selection = distance_to_horn_center > horn_diameter
    horn_intercept[selection, :] = nanvec3d

    rt_xds["distance_secondary_to_horn"] = xr.DataArray(
        distance_secondary_to_horn, dims=["points"]
    )
    rt_xds["horn_intercept"] = xr.DataArray(horn_intercept, dims=["points", "xyz"])
    return rt_xds


def compute_phase(rt_xds, wavelength, phase_offset):
    """
    Uses the distances along the ray from the rim of the primary all the way to the horn to compute the phase of
    each ray
    Args:
        rt_xds: Ray tracing XDS with the distances from the primary to the esconday and secondary to horn.
        wavelength: The light wavelength
        phase_offset: A phase offset to be added to the phases (i.e. light may not arrive with phase=0)

    Returns:
        RT XDS enriched with the rays' total_path and their phases
    """
    incident_light = rt_xds["light"]
    primary_points_z = rt_xds["primary_points"].values[:, 2]
    distance_pr_horn = (
        rt_xds["distance_secondary_to_horn"].values
        + rt_xds["distance_primary_to_secondary"].values
    )

    maxheight = np.max(primary_points_z)
    boresight = np.empty_like(incident_light)
    boresight[:] = [0, 0, -1]  # strictly vertical
    cosbeta = generalized_dot(boresight, incident_light)
    path_diff_before_dish = (maxheight - primary_points_z) / cosbeta
    total_path = np.where(
        np.isnan(rt_xds["horn_intercept"].values[:, 0]),
        np.nan,
        distance_pr_horn + path_diff_before_dish,
    )

    wavenumber = total_path / wavelength
    phase = phase_wrapping(twopi * wavenumber + phase_offset)

    rt_xds["total_path"] = xr.DataArray(total_path, dims=["points"])
    rt_xds["phase"] = xr.DataArray(phase, dims=["points"])
    return rt_xds


###########################################################
# Plotting routines and plotting aids, such as regridding #
###########################################################
def regrid_data_onto_2d_grid(npnt, data, indexes):
    """
    Use index information to get 1D data back onto a 2D grid.
    Args:
        npnt: Number of points on the 2d grid (assumed to be a square)
        data: 1D data to be regridded
        indexes: 1D array of 2D indexes

    Returns:
        Data regridded onto a 2D array
    """
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
    """
    Create a string with all the descriptive user inputs for the results of the ray tracing.
    Args:
        inpt_dict: Dinctionary of user inputs given by the user to the RT pipeline

    Returns:
        A string formatted to display the information in a matplotlib plot.
    """
    title = ""
    title += (
        f'Pointing offset = ({inpt_dict["x_pointing_offset"]}, {inpt_dict["y_pointing_offset"]}) '
        f'[{inpt_dict["pointing_offset_unit"]}], '
    )
    title += (
        f'Focus offset = ({inpt_dict["x_focus_offset"]}, {inpt_dict["y_focus_offset"]}, '
        f'{inpt_dict["z_focus_offset"]}) [{inpt_dict["focus_offset_unit"]}], '
    )
    title += f'Phase offset = {inpt_dict["phase_offset"]} [{inpt_dict["phase_unit"]}], '
    lambda_char = "\u03bb"
    title += f'{lambda_char} = {inpt_dict["observing_wavelength"]} [{inpt_dict["wavelength_unit"]}]'
    return title


def _imshow_2d_map(
    ax,
    fig,
    gridded_array,
    title,
    extent,
    zlabel,
    colormap,
    inner_radius,
    outer_radius,
    zlim,
):
    """
    Simple wrap function around pyplot.imshow which allows for some customization while also making the plots standard.
    Args:
        ax: Axes object onto which to plot
        fig: The figure in which the plot is inbedded
        gridded_array: The 2D representation of the data to be plotted
        title: Title to be shown on top of plot
        extent: The span of the image in the X and Y directions
        zlabel: The label to be attached to the colorbar
        colormap: The colormap to be used in the plot
        inner_radius: The inner radius over witch to draw a circle
        outer_radius: The outer radius over witch to draw a circle
        zlim: Limits of the colorbar

    Returns:
        None
    """
    cmap = get_proper_color_map(colormap)
    if zlim is None:
        minmax = [np.nanmin(gridded_array), np.nanmax(gridded_array)]
    else:
        minmax = zlim
    fsize = 10
    if title is not None:
        ax.set_title(title, size=1.5 * fsize)

    im = ax.imshow(
        gridded_array.T,
        cmap=cmap,
        extent=extent,
        interpolation="nearest",
        vmin=minmax[0],
        vmax=minmax[1],
        origin="lower",
    )

    if zlabel is None:
        well_positioned_colorbar(ax, fig, im, "Z Scale")
    else:
        well_positioned_colorbar(ax, fig, im, zlabel)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xlabel("X axis [m]")
    ax.set_ylabel("Y axis [m]")

    innercircle = plt.Circle((0, 0), inner_radius, color="black", fill=None)
    outercircle = plt.Circle((0, 0), outer_radius, color="black", fill=None)
    ax.add_patch(outercircle)
    ax.add_patch(innercircle)


def plot_2d_map(
    gridded_array,
    axis,
    telescope_parameters,
    suptitle,
    filename,
    zlabel,
    colormap,
    zlim,
    display,
    dpi,
):
    """
    Receive 2D gridded scalar or Vectorial data and plot accordingly
    Args:
        gridded_array: 2D gridded scalar or vectorial data
        axis: An axis that represents both X and Y axes
        telescope_parameters: Optical parameters of the telescope used in the Ray tracing
        suptitle: Overhanging title to be shown on top of figure
        filename: Name for the file containing the plot(s)
        zlabel: Label to be attached to the colorbar
        colormap: The colormap to be used in the plot
        zlim: Limits of the colorbar
        display: Display plots?
        dpi: Plot resolution on png file

    Returns:
        None
    """
    inner_radius = telescope_parameters["inner_radius"]
    outer_radius = telescope_parameters["primary_diameter"] / 2
    axes = ["X", "Y", "Z"]
    extent = compute_extent(axis, axis, margin=0.0)

    if len(gridded_array.shape) == 3:  # XYZ Plots
        fig, ax = create_figure_and_axes([20, 8], [1, 3])
        for iax in range(3):
            _imshow_2d_map(
                ax[iax],
                fig,
                gridded_array[..., iax],
                f"{axes[iax]} Component",
                extent,
                zlabel,
                colormap,
                inner_radius,
                outer_radius,
                zlim,
            )
    else:
        fig, ax = create_figure_and_axes([10, 8], [1, 1])
        _imshow_2d_map(
            ax,
            fig,
            gridded_array,
            None,
            extent,
            zlabel,
            colormap,
            inner_radius,
            outer_radius,
            zlim,
        )

    close_figure(fig, suptitle, filename, dpi, display)


def add_rz_ray_to_plot(ax, origin, destiny, color, ls, label, sign):
    """
    Adds a finite ray to a radial projection plot
    Args:
        ax: Axes object onto which to plot
        origin: Ray origin
        destiny: Ray destiny
        color: Ray color to be plotted
        ls: Ray line style
        label: Ray's label
        sign: is the ray to be shown on the negative or positive side?

    Returns:
        None
    """
    radcoord = [
        sign * generalized_norm(origin[0:2]),
        sign * generalized_norm(destiny[0:2]),
    ]
    zcoord = [origin[2], destiny[2]]
    ax.plot(radcoord, zcoord, color=color, label=label, ls=ls)


def compare_ray_tracing_to_phase_fit_results(
    rt_xds,
    phase_fit_results,
    phase_5d,
    phase_corrected_angle,
    filename,
    phase_unit="deg",
    colormap="viridis",
    display=False,
    dpi=300,
):
    """
    Compare phase fitting results to ray tracing simulation inputs
    Args:
        rt_xds: ray tracing XDS object
        phase_fit_results: Phase fitting array results.
        phase_5d: RT phase simulation inbedded in a 5D array for phase fitting comparison
        phase_corrected_angle: The 5D residuals of the phase fitting
        filename: Name of the png file onto which to save phase comparison plots.
        phase_unit: Unit to use for phase displays
        colormap: The colormap to be used in the plot
        display: Display plots?
        dpi: Plot resolution on png file

    Returns:
        None
    """
    xds_inp = rt_xds.attrs["input_parameters"]
    angle_unit = xds_inp["pointing_offset_unit"]
    length_unit = xds_inp["focus_offset_unit"]
    field_names = ["Parameter", "Value", "Reference", "Difference", "unit"]
    alignment = "c"
    outstr = ""
    wavelength = xds_inp["observing_wavelength"] * convert_unit(
        xds_inp["wavelength_unit"], "m", "length"
    )
    valid_pars = [
        "phase_offset",
        "x_point_offset",
        "y_point_offset",
        "x_focus_offset",
        "y_focus_offset",
        "z_focus_offset",
    ]
    unit_types = [
        "trigonometric",
        "trigonometric",
        "trigonometric",
        "length",
        "length",
        "length",
    ]
    units = ["deg", angle_unit, angle_unit, length_unit, length_unit, length_unit]
    reference_values = [
        0.0,
        xds_inp["x_pointing_offset"],
        xds_inp["y_pointing_offset"],
        xds_inp["x_focus_offset"],
        xds_inp["y_focus_offset"],
        xds_inp["z_focus_offset"],
    ]

    outstr += "Comparison between input and fitted values\n"
    freq = clight / wavelength
    cropped_dict = phase_fit_results["map_0"][freq]["I"]
    table = create_pretty_table(field_names, alignment)
    for ip, par_name in enumerate(valid_pars):
        item = cropped_dict[par_name]
        val = item["value"]
        err = item["error"]
        unitin = item["unit"]
        ref = reference_values[ip]
        fac = convert_unit(unitin, units[ip], unit_types[ip])
        val *= fac
        err *= fac
        diff = val - ref
        row = [
            format_label(par_name),
            format_value_error(val, err, 1.0, 1e-4),
            f"{ref}",
            f"{diff}",
            units[ip],
        ]
        table.add_row(row)

    outstr += table.get_string() + "\n\n"
    print(outstr)

    phase_2d = phase_5d[0, 0, 0]
    residuals_2d = phase_corrected_angle[0, 0, 0]
    correction = residuals_2d - phase_2d

    axis = rt_xds["x_axis"].values
    extent = compute_extent(axis, axis, margin=0.0)
    telescope_parameters = rt_xds.attrs["telescope_parameters"]
    inner_radius = telescope_parameters["inner_radius"]
    outer_radius = telescope_parameters["primary_diameter"] / 2
    fac = convert_unit("rad", phase_unit, "trigonometric")
    zlim = [-np.pi * fac, np.pi * fac]

    fig, ax = create_figure_and_axes([20, 8], [1, 3])
    statkeys = ["mean", "median", "rms"]

    _imshow_2d_map(
        ax[0],
        fig,
        fac * phase_2d,
        f"RT phase model\n{statistics_to_text(data_statistics(fac * phase_2d), statkeys)}",
        extent,
        f"Phase [{phase_unit}]",
        colormap,
        inner_radius,
        outer_radius,
        zlim,
    )
    _imshow_2d_map(
        ax[1],
        fig,
        fac * correction,
        f"Fitted correction",
        extent,
        f"Phase [{phase_unit}]",
        colormap,
        inner_radius,
        outer_radius,
        zlim,
    )
    _imshow_2d_map(
        ax[2],
        fig,
        fac * residuals_2d,
        f"Residuals\n{statistics_to_text(data_statistics(fac * residuals_2d), statkeys)}",
        extent,
        f"Phase [{phase_unit}]",
        colormap,
        inner_radius,
        outer_radius,
        zlim,
    )
    close_figure(
        fig,
        "Cassegrain RT model fitting for \n"
        + title_from_input_parameters(rt_xds.attrs["input_parameters"]),
        filename,
        dpi,
        display,
    )
