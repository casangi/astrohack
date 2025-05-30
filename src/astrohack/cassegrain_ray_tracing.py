import toolviper
import xarray as xr

from astrohack.antenna.telescope import Telescope
from astrohack.utils.validation import custom_unit_checker, custom_plots_checker
from astrohack.core.cassegrain_ray_tracing import *
from astrohack.utils import convert_unit, clight, add_caller_and_version_to_dict
from astrohack.utils.phase_fitting import aips_like_phase_fitting
from astrohack.visualization.plot_tools import create_figure_and_axes, close_figure
from typing import Union


@toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
def create_ray_tracing_telescope_parameter_dict(
    primary_diameter: Union[float, int] = 25,
    secondary_diameter: Union[float, int] = 2.5146,
    focal_length: Union[float, int] = 9.0,
    z_intercept: Union[float, int] = 3.140,
    foci_half_distance: Union[float, int] = 3.662,
    inner_radius: Union[float, int] = 2.0,
    horn_diameter: Union[float, int] = 0.2,
    length_unit: str = "m",
):
    """Create a dictionary with a cassegrain telescope parameters

    :param primary_diameter: Diameter of the primary mirror.
    :type primary_diameter: float, int, optional

    :param secondary_diameter: Diameter of the secondary mirror.
    :type secondary_diameter: float, int, optional

    :param focal_length: Focal length of the primary mirror.
    :type focal_length: float, int, optional

    :param z_intercept: Distance between the Z intercept of the secondary and the mid-point between the primary and \
    secondary focus, usually refered to as 'a'.
    :type z_intercept: float, int, optional

    :param foci_half_distance: Half-distance between the primary and secondary foci, usually refered to as 'c'.
    :type foci_half_distance: float, int, optional

    :param inner_radius: Inner valid surface radius of the primary reflector, usually refered to as Blockage.
    :type inner_radius: float, int, optional

    :param horn_diameter: Diameter of the horn detecting the signals, used to determine if rays are detected or lost.
    :type horn_diameter: float, int, optional

    :param length_unit: Unit for the telescope dimensions, default is "m".
    :type length_unit: str, optional

    :return: A dictionary filled with the user inputs and also the horn position and orientation.
    :rtype: dict

    .. _Description:

        Create a basic description of a Cassegrain radio telescope from user inputs. This function assumes that the\
         horn is positioned at the secondary focus and is pointed directly upwards, i.e. an axi-symmetric design.\
        Default values reflect the values for the VLA available in EVLA memo 211.

    """
    local_vars = locals()

    # Convert dimensions from user unit to meters
    fac = convert_unit(length_unit, "m", "length")
    telescope_parameters = {}
    for key, item in local_vars.items():
        if key != "length_unit":
            telescope_parameters[key] = item * fac

    # Assumed to be at the Secondary focus i.e.: f - 2c
    telescope_parameters["horn_position"] = [
        0,
        0,
        telescope_parameters["focal_length"]
        - 2 * telescope_parameters["foci_half_distance"],
    ]
    # Horn looks straight up
    telescope_parameters["horn_orientation"] = [0, 0, 1]
    return telescope_parameters


@toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
def cassegrain_ray_tracing_pipeline(
    output_xds_filename: str,
    telescope_parameters: dict,
    grid_size: Union[float, int] = 28,
    grid_resolution: Union[float, int] = 0.1,
    grid_unit: str = "m",
    x_pointing_offset: Union[float, int] = 0,
    y_pointing_offset: Union[float, int] = 0,
    pointing_offset_unit: str = "asec",
    x_focus_offset: Union[float, int] = 0,
    y_focus_offset: Union[float, int] = 0,
    z_focus_offset: Union[float, int] = 0,
    focus_offset_unit: str = "mm",
    phase_offset: Union[float, int] = 0,
    phase_unit: str = "deg",
    observing_wavelength: Union[float, int] = 1,
    wavelength_unit: str = "cm",
    overwrite: bool = False,
):
    """Execute the cassegrain ray tracing pipeline to determine phase effects caused by optical mis-alignments.

    :param output_xds_filename: Filename for the output Xarray dataset on disk using a Zarr container.
    :type output_xds_filename: str

    :param telescope_parameters: Dictionary containing the parameters of the cassegrain telescope in use.
    :type telescope_parameters: dict

    :param grid_size: Size of the grid onto which to compute phase effects in grid_unit.
    :type grid_size: float, int, optional

    :param grid_resolution: Resolution of the grid onto which to compute phase effects in grid_unit.
    :type grid_resolution: float, int, optional

    :param grid_unit: Length unit for grid_size and grid_resolution, default is "m".
    :type grid_unit: str, optional

    :param x_pointing_offset: X Pointing offset in pointing_offset_unit.
    :type x_pointing_offset: float, int, optional
    
    :param y_pointing_offset: Y Pointing offset in pointing_offset_unit.
    :type y_pointing_offset: float, int, optional
    
    :param pointing_offset_unit: Angle unit for pointing offsets, default is "asec".
    :type pointing_offset_unit: str, optional
    
    :param x_focus_offset: X offset of the secondary in focus_offset_unit.
    :type x_focus_offset: float, int, optional
    
    :param y_focus_offset: Y offset of the secondary in focus_offset_unit.
    :type y_focus_offset: float, int, optional
    
    :param z_focus_offset: Z offset of the secondary in focus_offset_unit, what is usually refered to as simply focus.
    :type z_focus_offset: float, int, optional
    
    :param focus_offset_unit: Length unit for focus offsets, default is "mm".
    :type focus_offset_unit: str, optional

    :param phase_offset: A phase offset to be applied to the phase image.
    :type phase_offset: float, int, optional

    :param phase_unit: Angle unit for the phase offset, default is "deg".
    :type phase_unit: str, optional

    :param observing_wavelength: Wavelength of the rays to be simulated in wavelength unit.
    :type observing_wavelength: float, int, optional

    :param wavelength_unit: Length unit for the observing wavelength, default is "cm".
    :type wavelength_unit: str, optional

    :param overwrite: Overwrite rt_xds file on disk, default is False.
    :type overwrite: bool, optional

    :return: X array dataset object with the results from the ray tracing.
    :rtype: xr.Dataset

    .. _Description:
        Calculate the total path for rays from the moment they pass the rim of the primary dish until they are detected\
         at the horn and computes the resulting phase.

        .. rubric:: Code Outline
        - A Gridded representation of the primary dish and the normal to its surface are created and stored in an XDS.
        - The reflection between the incident light and the primary mirror is computed for each of the gridded points.
        - The reflected rays from the primary are progated and the intercept between them and the secondary is \
        calculated.
        - Compute the reflection at the secondary for each ray reflected at the primary that touches it.
        - Check which rays from the secondary intercept the mouth of the horn.
        - Compute the total path from the rim of the primary up to horn for detected rays.
        - Compute the phase for each ray based on the total path.

        .. rubric:: Limitations
        - This ray tracing code only aims at estimating aperture phases, not its amplitude.
        - Detection from sidelobes is not estimated.
        - Beam shape estimations cannot be computed from the results as amplitudes are not modeled.
        - This model is axi-symmetric, i.e. it cannot be used to estimate the full range of phase effects present in \
        VLA apertures.
        - If large pointing or focus offsets are chosen the rays may stop intercepting the horn and hence produce \
        partially or fully blank phase images.

    """
    input_pars = locals()
    del input_pars["telescope_parameters"]
    add_caller_and_version_to_dict(input_pars)

    # Convert user units and build proper RT inputs
    grid_fac = convert_unit(grid_unit, "m", "length")
    grid_size *= grid_fac
    grid_resolution *= grid_fac

    focus_fac = convert_unit(focus_offset_unit, "m", "length")
    focus_offset = focus_fac * np.array(
        [x_focus_offset, y_focus_offset, z_focus_offset]
    )

    pnt_fac = convert_unit(pointing_offset_unit, "rad", "trigonometric")
    x_pointing_offset *= pnt_fac
    y_pointing_offset *= pnt_fac
    # Using small angles approximation here
    pnt_off = np.sqrt(x_pointing_offset**2 + y_pointing_offset**2)
    incident_light = np.array(
        [-np.sin(x_pointing_offset), -np.sin(y_pointing_offset), -np.cos(pnt_off)]
    )

    # Actual Ray Tracing starts here
    rt_xds = make_gridded_cassegrain_primary(
        grid_size, grid_resolution, telescope_parameters
    )
    rt_xds = reflect_off_primary(rt_xds, incident_light)
    rt_xds = reflect_off_analytical_secondary(rt_xds, focus_offset)
    rt_xds = detect_light(rt_xds)
    rt_xds = compute_phase(
        rt_xds,
        observing_wavelength * convert_unit(wavelength_unit, "m", "length"),
        phase_offset * convert_unit(phase_unit, "rad", "trigonometric"),
    )

    rt_xds.attrs["input_parameters"] = input_pars

    write_rt_xds_to_zarr(rt_xds, output_xds_filename, overwrite)
    return rt_xds


@toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
def plot_2d_maps_from_rt_xds(
    rt_xds_filename: str,
    keys: Union[str, list],
    rootname: str,
    phase_unit: str = "deg",
    length_unit: str = "m",
    colormap: str = "viridis",
    display: bool = True,
    dpi: int = 300,
):
    """Plot 2D maps of keys in the ray tracing Xarray Dataset

    :param rt_xds_filename: Name on disk of the Xarray dataset containing the results of the Ray tracing pipeline
    :type rt_xds_filename: str

    :param keys: Key or keys in rt_xds to be plotted.
    :type keys: str, list

    :param rootname: Root name for the plots to be created.
    :type rootname: str

    :param phase_unit: Unit for the phase plot, default is "deg".
    :type phase_unit: str, optional

    :param length_unit: Unit for the plots of keys other than phase, default is "m".
    :type length_unit: str, optional

    :param colormap: Colormap to be used for plots, default is "viridis".
    :type colormap: str, optional

    :param display: Display plots inline or suppress, defaults to True
    :type display: bool, optional

    :param dpi: dots per inch to be used in plots, default is 300
    :type dpi: int, optional

    .. _Description:

        Produce plots from the Xarray dataset containing ray tracing results for analysis. All Xarray dataset data \
        variables except for the x and y axes can be plotted.
    """

    rt_xds = open_rt_zarr(rt_xds_filename)

    if isinstance(keys, str):
        keys = [keys]

    suptitle = "Cassegrain Ray tracing model for:\n" + title_from_input_parameters(
        rt_xds.attrs["input_parameters"]
    )
    for key in keys:
        filename = f"{rootname}_{key}.png"

        zlabel = key.capitalize().replace("_", " ")
        if key == "phase":
            fac = convert_unit("rad", phase_unit, "trigonometric")
            zlabel += f" [{phase_unit}]"
            zlim = [fac * -np.pi, fac * np.pi]
        else:
            fac = convert_unit("m", length_unit, "length")
            zlabel += f" [{length_unit}]"
            zlim = None

        gridded_array = fac * regrid_data_onto_2d_grid(
            rt_xds.attrs["image_size"],
            rt_xds[key].values,
            rt_xds["image_indexes"].values,
        )

        plot_2d_map(
            gridded_array,
            rt_xds["x_axis"].values,
            rt_xds.attrs["telescope_parameters"],
            suptitle,
            filename,
            zlabel,
            colormap,
            zlim,
            display=display,
            dpi=dpi,
        )
    return


@toolviper.utils.parameter.validate()
def plot_radial_projection_from_rt_xds(
    rt_xds_filename: str,
    plot_filename: str,
    nrays: int = 20,
    display: bool = True,
    dpi: int = 300,
):
    """Plot a radial projection of some of the rays simulated in the ray tracing Xarray Dataset.

    :param rt_xds_filename: Name on disk of the Xarray dataset containing the results of the Ray tracing pipeline
    :type rt_xds_filename: xr.Dataset

    :param plot_filename: Name of the file to contain the plot.
    :type plot_filename: str

    :param nrays: Number of random rays to be plotted, default is 20.
    :type nrays: int, optional

    :param display: Display plot inline or suppress, default is True
    :type display: bool, optional

    :param dpi: dots per inch to be used in plots, default is 300
    :type dpi: int, optional

    .. _Description:

        Produce a plot of a random selection of nrays that are present on the input Xarray dataset.

    """
    rt_xds = open_rt_zarr(rt_xds_filename)

    telescope_pararameters = rt_xds.attrs["telescope_parameters"]
    primary_diameter = telescope_pararameters["primary_diameter"]
    secondary_diameter = telescope_pararameters["secondary_diameter"]
    focal_length = telescope_pararameters["focal_length"]
    foci_half_distance = telescope_pararameters["foci_half_distance"]
    z_intercept = telescope_pararameters["z_intercept"]

    pr_rad = primary_diameter / 2
    sc_rad = secondary_diameter / 2
    radarr = np.arange(-pr_rad, pr_rad, primary_diameter / 1e3)
    primary = radarr**2 / 4 / focal_length
    secondary = (
        focal_length
        - foci_half_distance
        + z_intercept
        * np.sqrt(1 + radarr**2 / (foci_half_distance**2 - z_intercept**2))
    )
    secondary = np.where(np.abs(radarr) < sc_rad, secondary, np.nan)
    fig, ax = create_figure_and_axes([16, 8], [1, 1])
    ax.plot(radarr, primary, color="black", label="Pr mirror")
    ax.plot(radarr, secondary, color="blue", label="Sc mirror")
    ax.scatter([0], [focal_length], color="black", label="Pr focus")
    ax.scatter(
        [0], [focal_length - 2 * foci_half_distance], color="blue", label="Sc focus"
    )

    primary_points = rt_xds["primary_points"].values
    secondary_points = rt_xds["secondary_points"].values
    horn_intercepts = rt_xds["horn_intercept"].values
    incoming_light = rt_xds["light"].values
    secondary_reflections = rt_xds["secondary_reflections"].values
    primary_reflections = rt_xds["primary_reflections"].values

    npnt = primary_points.shape[0]
    sign = -1
    inf = 1e3
    if nrays > npnt:
        logger.warning(
            "Requested number of plotted rays is larger than the number of available rays."
        )
        nrays = npnt
    elif nrays == 0:
        logger.warning(
            "No rays requested, plot will be a simple Radial projection of the optical system."
        )

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

        inf_str = "\u221e"
        # Plot incident light
        origin = pr_pnt - inf * incoming
        add_rz_ray_to_plot(ax, origin, pr_pnt, "yellow", "-", f"{inf_str}->Pr", sign)

        # Plot primary reflection
        if np.all(np.isnan(sc_pnt)):  # Ray does not touch secondary
            dest = pr_pnt + inf * pr_ref
            add_rz_ray_to_plot(ax, pr_pnt, dest, "red", "--", f"Pr->{inf_str}", sign)
        else:
            add_rz_ray_to_plot(ax, pr_pnt, sc_pnt, "yellow", "--", "Pr->Sc", sign)

            # Plot secondary reflection
            if np.all(np.isnan(horn_inter)):  # Ray does not touch horn
                dest = sc_pnt + inf * sc_ref
                add_rz_ray_to_plot(
                    ax, sc_pnt, dest, "red", "-.", f"Sc->{inf_str}", sign
                )
            else:
                add_rz_ray_to_plot(
                    ax, sc_pnt, horn_inter, "yellow", "-.", "Sc->Horn", sign
                )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax.legend(by_label.values(), by_label.keys())
    ax.set_aspect("equal")
    ax.set_xlabel("Radius [m]")
    ax.set_ylabel("Height [m]")
    ax.set_ylim([-0.5, 9.5])
    ax.set_xlim([-13, 13])
    ax.set_title("Cassegrain Ray tracing 2D Schematic")
    close_figure(
        fig,
        title_from_input_parameters(rt_xds.attrs["input_parameters"]),
        plot_filename,
        dpi,
        display,
    )


@toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
def apply_holog_phase_fitting_to_rt_xds(
    rt_xds_filename: str,
    phase_plot_filename: str,
    fit_pointing_offset: bool = True,
    fit_xy_secondary_offset: bool = True,
    fit_focus_offset: bool = True,
    phase_unit: str = "deg",
    colormap: str = "viridis",
    display: bool = True,
    dpi: int = 300,
):
    """Feed phase image from ray tracing Xarray dataset to Astrohak's default phase fitting tool for VLA data.

    :param rt_xds_filename: Name on disk of the Xarray dataset containing the results of the Ray tracing pipeline
    :type rt_xds_filename: xr.Dataset

    :param phase_plot_filename: filename for the plot containing the RT phase image the fitted phase effects and the \
    residuals
    :type phase_plot_filename: str

    :param fit_pointing_offset: Toggle to determine if pointing offsets are to be fitted, default is True.
    :type fit_pointing_offset: bool, optional

    :param fit_xy_secondary_offset: Toggle to determine if lateral displacements of the secondary are to be fitted,
    default is True.
    :type fit_xy_secondary_offset: bool, optional

    :param fit_focus_offset: Toggle to determine if vertical displacements of the secondary are to be fitted, \
    default is True.
    :type fit_focus_offset: bool, optional

    :param phase_unit: Unit for the phase plot, default is "deg".
    :type phase_unit: str, optional

    :param colormap: Colormap to be used for plots, default is "viridis".
    :type colormap: str, optional

    :param display: Display plots inline or suppress, defaults to True
    :type display: bool, optional

    :param dpi: dots per inch to be used in plots, default is 300
    :type dpi: int, optional

    .. _Description:

        Apply the phase fitting engine used in ``astrohack.holog.holog`` to the phase image computed by the ray \
        tracing pipeline. At the end of the fitting produces a table so that fitting results can be compared to the \
        inputs given for the ray tracing pipeline. Along with the table a plot is produced containing the ray tracing \
        modelled phases, the correction derived from the phase fitting tool and the residuals of the fitting. For \
        easier comparison simple statistics of each image are provided.

    """
    rt_xds = open_rt_zarr(rt_xds_filename)

    ntime = 1
    npol = 1
    nfreq = 1
    # Pull data from rt xds
    npnt = rt_xds.attrs["image_size"]
    telescope_pars = rt_xds.attrs["telescope_parameters"]
    input_pars = rt_xds.attrs["input_parameters"]
    u_axis = rt_xds["x_axis"].values
    v_axis = rt_xds["y_axis"].values
    wavelength = input_pars["observing_wavelength"] * convert_unit(
        input_pars["wavelength_unit"], "m", "length"
    )

    # Create Amplitude and phase images on the shape expected by phase fitting engine.
    shape_5d = [ntime, npol, nfreq, npnt, npnt]
    amplitude_5d = np.empty(shape_5d)
    phase_2d = regrid_data_onto_2d_grid(
        npnt, rt_xds["phase"].values, rt_xds["image_indexes"].values
    )
    phase_5d = np.empty_like(amplitude_5d)
    phase_5d[..., :, :] = phase_2d
    radial_mask, radius, _ = create_aperture_mask(
        u_axis,
        v_axis,
        telescope_pars["inner_radius"],
        telescope_pars["primary_diameter"] / 2,
        return_polar_meshes=True,
    )
    amplitude_5d[..., :, :] = np.where(radial_mask, 1.0, np.nan)

    # Create frequency and polarization axes
    freq_axis = np.array([clight / wavelength])
    pol_axis = np.array(["I"])

    # Misc Parameters
    label = "Cassegrain-RT-Model"  # Relevant only for logger messages
    uv_cell_size = np.array(
        [u_axis[1] - u_axis[0], v_axis[1] - v_axis[0]]
    )  # This should be computed from the axis we are passing the engine...

    # Initiate Control toggles
    phase_fit_control = [
        fit_pointing_offset,  # Pointing Offset (Supported)
        fit_xy_secondary_offset,  # X&Y Focus Offset (Supported)
        fit_focus_offset,  # Z Focus Offset (Supported)
        False,  # Sub-reflector Tilt (not supported)
        False,  # Cassegrain offset (not supported)
    ]

    # Manipulate VLA telescope object so that it has compatible parameters to the ones in the RT model.
    telescope = Telescope("VLA")
    telescope.focus = telescope_pars["focal_length"]
    c_fact = telescope_pars["foci_half_distance"]
    a_fact = telescope_pars["z_intercept"]
    telescope.magnification = (c_fact + a_fact) / (c_fact - a_fact)
    telescope.secondary_dist = c_fact - a_fact
    # Disable secondary slope
    telescope.surp_slope = 0

    phase_corrected_angle, phase_fit_results = aips_like_phase_fitting(
        amplitude_5d,
        phase_5d,
        pol_axis,
        freq_axis,
        telescope,
        u_axis,
        v_axis,
        phase_fit_control,
        label,
    )

    compare_ray_tracing_to_phase_fit_results(
        rt_xds,
        phase_fit_results,
        phase_5d,
        phase_corrected_angle,
        phase_plot_filename,
        phase_unit=phase_unit,
        colormap=colormap,
        display=display,
        dpi=dpi,
    )
