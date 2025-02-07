from astrohack import Telescope
from astrohack.core.vla_ray_tracing import *
from astrohack.core.vla_ray_tracing import  create_radial_mask, \
    compare_ray_tracing_to_phase_fit_results
from astrohack.utils import convert_unit, clight
from astrohack.utils.phase_fitting import execute_phase_fitting
from astrohack.visualization.plot_tools import create_figure_and_axes, close_figure


def plot_2d_maps_from_rt_xds(rt_xds, keys, rootname, phase_unit='deg', length_unit='m', colormap='viridis',
                             display=False, dpi=300):
    suptitle = title_from_input_parameters(rt_xds.attrs['input_parameters'])
    for key in keys:
        filename = f'{rootname}_{key}.png'

        zlabel = key.capitalize()
        if key == 'phase':
            fac = convert_unit('rad', phase_unit, 'trigonometric')
            zlabel += f' [{phase_unit}]'
            zlim = [fac*-np.pi, fac*np.pi]
        else:
            fac = convert_unit('m', length_unit, 'length')
            zlabel += f' [{length_unit}]'
            zlim = None

        gridded_array = fac * regrid_data_onto_2d_grid(rt_xds.attrs['image_size'], rt_xds[key].values,
                                                       rt_xds['image_indexes'].values)

        plot_2d_map(gridded_array, rt_xds["x_axis"].values, rt_xds.attrs['telescope_parameters'], suptitle, filename,
                    zlabel, colormap, zlim, display=display, dpi=dpi)
    return


def plot_radial_projection_from_rt_xds(rt_xds, plot_filename, nrays=20, display=False, dpi=300):
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
    close_figure(fig, title_from_input_parameters(rt_xds.attrs['input_parameters']), plot_filename, dpi, display)


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
    incident_light = np.array([-np.sin(x_pnt_off), -np.sin(y_pnt_off), -np.cos(pnt_off)])

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


def apply_vla_phase_fitting_to_xds(rt_xds, phase_plot_filename, fit_pointing_offset=True, fit_xy_secondary_offset=True,
                                   fit_focus_off=True, phase_unit='deg', colormap='viridis', display=False, dpi=300):
    ntime = 1
    npol = 1
    nfreq = 1
    # Pull data from rt xds
    npnt = rt_xds.attrs['image_size']
    telescope_pars = rt_xds.attrs['telescope_parameters']
    input_pars = rt_xds.attrs['input_parameters']
    u_axis = rt_xds['x_axis'].values
    v_axis = rt_xds['y_axis'].values
    wavelength = input_pars['observing_wavelength']*convert_unit(input_pars['wavelength_unit'], 'm', 'length')

    # Create Amplitude and phase images on the shape expected by phase fitting engine.
    shape_5d = [ntime, npol, nfreq, npnt, npnt]
    amplitude_5d = np.empty(shape_5d)
    phase_2d = regrid_data_onto_2d_grid(npnt, rt_xds['phase'].values, rt_xds['image_indexes'].values)
    phase_5d = np.empty_like(amplitude_5d)
    phase_5d[..., :, :] = phase_2d
    _, _, radius = create_coordinate_images(u_axis, v_axis)
    radial_mask = create_radial_mask(radius, telescope_pars['inner_radius'], telescope_pars['primary_diameter'] / 2)
    amplitude_5d[..., :, :] = np.where(radial_mask, 1.0, np.nan)

    # Create frequency and polarization axes
    freq_axis = np.array([clight/wavelength])
    pol_axis = np.array(['I'])

    # Misc Parameters
    focus_offset = 0.0 # Only relevant for Near Field data
    label = 'VLA-RT-Model' # Relevant only for logger messages
    uv_cell_size = np.array([u_axis[1]-u_axis[0], v_axis[1]-v_axis[0]]) # This should be computed from the axis we are passing the engine...

    # Initiate Control toggles
    is_stokes = True
    is_near_field = False
    phase_fit_parameter = [fit_pointing_offset,  # Pointing Offset (Supported)
                           fit_xy_secondary_offset,  # X&Y Focus Offset (Supported)
                           fit_focus_off,  # Z Focus Offset (Supported)
                           False,  # Sub-reflector Tilt (not supported)
                           False  # Cassegrain offset (not supported)
                           ]

    # Manipulate VLA telescope object so that it has compatible parameters to the ones in the RT model.
    telescope = Telescope('VLA')
    telescope.focus = telescope_pars['focal_length']
    c_fact = telescope_pars['foci_half_distance']
    a_fact = telescope_pars['z_intercept']
    telescope.magnification = (c_fact+a_fact)/(c_fact-a_fact)
    telescope.secondary_dist = c_fact-a_fact
    # Disable secondary slope
    telescope.surp_slope = 0

    phase_corrected_angle, phase_fit_results = execute_phase_fitting(amplitude_5d, phase_5d, pol_axis, freq_axis,
                                                                     telescope, uv_cell_size, phase_fit_parameter,
                                                                     is_stokes, is_near_field, focus_offset, u_axis,
                                                                     v_axis, label)

    compare_ray_tracing_to_phase_fit_results(rt_xds, phase_fit_results, phase_5d, phase_corrected_angle,
                                             phase_plot_filename, phase_unit=phase_unit, colormap=colormap,
                                             display=display, dpi=dpi)
