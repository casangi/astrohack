import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from toolviper.utils import logger as logger

from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.antenna.telescope import Telescope
from astrohack.utils import (
    convert_unit,
    pi,
    clight,
    compute_antenna_relative_off,
    rotate_to_gmt,
    plot_types,
    convert_5d_grid_from_stokes,
    create_dataset_label,
)
from astrohack.utils.constants import fontsize, markersize
from astrohack.utils.text import param_to_list, add_prefix
from astrohack.utils.tools import get_telescope_lat_lon_rad
from astrohack.visualization.plot_tools import (
    create_figure_and_axes,
    close_figure,
    plot_boxes_limits_and_labels,
    scatter_plot,
    simple_imshow_map_plot,
)


def _extract_indices(laxis, maxis, squared_radius):
    indices = []

    assert laxis.shape[0] == maxis.shape[0], "l, m must be same size."

    for i in range(laxis.shape[0]):
        squared_sum = np.power(laxis[i], 2) + np.power(maxis[i], 2)
        if squared_sum <= squared_radius:
            indices.append(i)

    return np.array(indices)


def _matplotlib_calibration_inspection_function(
    data, delta=0.01, pol="RR", width=1000, height=450
):
    pixels = 1 / plt.rcParams["figure.dpi"]
    UNIX_CONVERSION = 3506716800

    radius = np.power(data.grid_params["cell_size"] * delta, 2)
    pol_index = np.squeeze(np.where(data.pol.values == pol))

    laxis = data.DIRECTIONAL_COSINES.values[..., 0]
    maxis = data.DIRECTIONAL_COSINES.values[..., 1]

    assert laxis.shape[0] == maxis.shape[0], "l, m dimensions don't match!"

    indices = _extract_indices(laxis=laxis, maxis=maxis, squared_radius=radius)

    vis = data.isel(time=indices).VIS
    times = Time(vis.time.data - UNIX_CONVERSION, format="unix").iso

    fig, axis = create_figure_and_axes([width * pixels, height * pixels], [2, 1])

    chan = np.arange(0, data.chan.data.shape[0])

    for i in range(times.shape[0]):
        axis[0].plot(chan, vis[i, :, pol_index].real, marker="o", label=times[i])
        axis[0].set_title(
            "Calibration Check: polarization={p}".format(p=data.pol.values[pol_index])
        )
        axis[0].set_ylabel("Visibilities (real)")
        axis[0].set_xlabel("Channel")

        axis[0].legend()

        axis[1].plot(chan, vis[i, :, pol_index].imag, marker="o", label=times[i])
        axis[1].set_ylabel("Visibilities (imag)")
        axis[1].set_xlabel("Channel")

        axis[1].legend()


def calibration_plot_chunk(param_dict):
    data = param_dict["xds_data"]
    delta = param_dict["delta"]
    complex_split = param_dict["complex_split"]
    display = param_dict["display"]
    figuresize = param_dict["figure_size"]
    destination = param_dict["destination"]
    dpi = param_dict["dpi"]
    thisfont = 1.2 * fontsize

    UNIX_CONVERSION = 3506716800

    radius = np.power(data.grid_params["cell_size"] * delta, 2)

    l_axis = data.DIRECTIONAL_COSINES.values[..., 0]
    m_axis = data.DIRECTIONAL_COSINES.values[..., 1]

    assert l_axis.shape[0] == m_axis.shape[0], "l, m dimensions don't match!"

    indices = _extract_indices(laxis=l_axis, maxis=m_axis, squared_radius=radius)

    if complex_split == "cartesian":
        vis_dict = {
            "data": [
                data.isel(time=indices).VIS.real,
                data.isel(time=indices).VIS.imag,
            ],
            "polarization": [0, 3],
            "label": ["REAL", "IMAG"],
        }
    else:
        vis_dict = {
            "data": [
                data.isel(time=indices).apply(np.abs).VIS,
                data.isel(time=indices).apply(np.angle).VIS,
            ],
            "polarization": [0, 3],
            "label": ["AMP", "PHASE"],
        }

    times = np.unique(
        Time(vis_dict["data"][0].time.data - UNIX_CONVERSION, format="unix").iso
    )

    fig, axis = create_figure_and_axes(figuresize, [4, 1], sharex=True)

    chan = np.arange(0, data.chan.data.shape[0])

    length = times.shape[0]

    for i, vis in enumerate(vis_dict["data"]):
        for j, pol in enumerate(vis_dict["polarization"]):
            for time in range(length):
                k = 2 * i + j
                axis[k].plot(
                    chan,
                    vis[time, :, pol],
                    marker="o",
                    label=times[time],
                    markersize=markersize,
                )
                axis[k].set_ylabel(
                    f'Vis ({vis_dict["label"][i]}; {data.pol.values[pol]})',
                    fontsize=thisfont,
                )
                axis[k].tick_params(axis="both", which="major", labelsize=thisfont)

    axis[3].set_xlabel("Channel", fontsize=thisfont)
    axis[0].legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncols=4,
        mode="expand",
        borderaxespad=0.0,
        fontsize=fontsize,
    )

    fig.suptitle(
        f'Data Calibration Check: [{param_dict["this_ddi"]}, {param_dict["this_map"]}, {param_dict["this_ant"]}]',
        ha="center",
        va="center",
        x=0.5,
        y=0.95,
        rotation=0,
        fontsize=1.5 * thisfont,
    )
    plotfile = (
        f'{destination}/holog_diagnostics_{param_dict["this_map"]}_{param_dict["this_ant"]}_'
        f'{param_dict["this_ddi"]}.png'
    )
    close_figure(fig, None, plotfile, dpi, display, tight_layout=False)


def plot_antenna_position(
    outerax, innerax, xpos, ypos, text, box_size, marker="+", color="black"
):
    """
    Plot an antenna to either the inner or outer array boxes
    Args:
        outerax: Plotting axis for the outer array box
        innerax: Plotting axis for the inner array box
        xpos: X antenna position (east-west)
        ypos: Y antenna position (north-south)
        text: Antenna label
        box_size: Size of the inner array box
        marker: Antenna position marker
        color: Color for the antenna position marker
    """
    half_box = box_size / 2
    if abs(xpos) > half_box or abs(ypos) > half_box:
        outerax.plot(xpos, ypos, marker=marker, color=color)
        outerax.text(xpos, ypos, text, fontsize=fontsize, ha="left", va="center")
    else:
        outerax.plot(xpos, ypos, marker=marker, color=color)
        innerax.plot(xpos, ypos, marker=marker, color=color)
        innerax.text(xpos, ypos, text, fontsize=fontsize, ha="left", va="center")


def plot_corrections(
    outerax, innerax, xpos, ypos, xcorr, ycorr, box_size, color="red", linewidth=0.5
):
    """
    Plot an antenna position corrections as a vector to the antenna position
    Args:
        outerax: Plotting axis for the outer array box
        innerax: Plotting axis for the inner array box
        xpos: X antenna position (east-west)
        ypos: Y antenna position (north-south)
        xcorr: X axis correction (horizontal on plot)
        ycorr: Y axis correction (vectical on plot)
        box_size: inner array box size
        color: vector color
        linewidth: vector line width
    """
    half_box = box_size / 2
    head_size = np.sqrt(xcorr**2 + ycorr**2) / 4
    if abs(xpos) > half_box or abs(ypos) > half_box:
        outerax.arrow(
            xpos,
            ypos,
            xcorr,
            ycorr,
            color=color,
            linewidth=linewidth,
            head_width=head_size,
        )
    else:
        outerax.arrow(
            xpos,
            ypos,
            xcorr,
            ycorr,
            color=color,
            linewidth=linewidth,
            head_width=head_size,
        )
        innerax.arrow(
            xpos,
            ypos,
            xcorr,
            ycorr,
            color=color,
            linewidth=linewidth,
            head_width=head_size,
        )


def plot_lm_coverage(param_dict):
    data = param_dict["xds_data"]
    angle_fact = convert_unit("rad", param_dict["angle_unit"], "trigonometric")
    real_lm = data["DIRECTIONAL_COSINES"] * angle_fact
    ideal_lm = data["IDEAL_DIRECTIONAL_COSINES"] * angle_fact
    time = data.time.values
    time -= time[0]
    time *= convert_unit("sec", param_dict["time_unit"], "time")
    param_dict["l_label"] = f'L [{param_dict["angle_unit"]}]'
    param_dict["m_label"] = f'M [{param_dict["angle_unit"]}]'
    param_dict["time_label"] = (
        f'Time from observation start [{param_dict["time_unit"]}]'
    )

    param_dict["marker"] = "."
    param_dict["linestyle"] = "-"
    param_dict["color"] = "blue"

    _plot_lm_coverage_sub(time, real_lm, ideal_lm, param_dict)

    if (
        param_dict["plot_correlation"] is None
        or param_dict["plot_correlation"] == "None"
    ):
        pass
    else:
        param_dict["linestyle"] = ""
        visi = np.average(data["VIS"].values, axis=1)
        weights = np.average(data["WEIGHT"].values, axis=1)
        pol_axis = data.pol.values
        if isinstance(param_dict["plot_correlation"], (list, tuple)):
            for correlation in param_dict["plot_correlation"]:
                plot_correlation(
                    visi, weights, correlation, pol_axis, time, real_lm, param_dict
                )
        else:
            if param_dict["plot_correlation"] == "all":
                for correlation in pol_axis:
                    plot_correlation(
                        visi, weights, correlation, pol_axis, time, real_lm, param_dict
                    )
            else:
                plot_correlation(
                    visi,
                    weights,
                    param_dict["plot_correlation"],
                    pol_axis,
                    time,
                    real_lm,
                    param_dict,
                )


def _plot_lm_coverage_sub(time, real_lm, ideal_lm, param_dict):
    fig, ax = create_figure_and_axes(param_dict["figure_size"], [2, 2])
    scatter_plot(
        ax[0, 0],
        time,
        param_dict["time_label"],
        real_lm[:, 0],
        param_dict["l_label"],
        "Time vs Real L",
        data_marker=param_dict["marker"],
        data_linestyle=param_dict["linestyle"],
        data_color=param_dict["color"],
    )
    scatter_plot(
        ax[0, 1],
        time,
        param_dict["time_label"],
        real_lm[:, 1],
        param_dict["m_label"],
        "Time vs Real M",
        data_marker=param_dict["marker"],
        data_linestyle=param_dict["linestyle"],
        data_color=param_dict["color"],
    )
    scatter_plot(
        ax[1, 0],
        real_lm[:, 0],
        param_dict["l_label"],
        real_lm[:, 1],
        param_dict["m_label"],
        "Real L and M",
        data_marker=param_dict["marker"],
        data_linestyle=param_dict["linestyle"],
        data_color=param_dict["color"],
    )
    scatter_plot(
        ax[1, 1],
        ideal_lm[:, 0],
        param_dict["l_label"],
        ideal_lm[:, 1],
        param_dict["m_label"],
        "Ideal L and M",
        data_marker=param_dict["marker"],
        data_linestyle=param_dict["linestyle"],
        data_color=param_dict["color"],
    )
    plotfile = (
        f'{param_dict["destination"]}/holog_directional_cosines_{param_dict["this_map"]}_'
        f'{param_dict["this_ant"]}_{param_dict["this_ddi"]}.png'
    )
    close_figure(
        fig, "Directional Cosines", plotfile, param_dict["dpi"], param_dict["display"]
    )


def plot_correlation(visi, weights, correlation, pol_axis, time, lm, param_dict):
    if correlation in pol_axis:
        ipol = pol_axis == correlation
        loc_vis = visi[:, ipol]
        loc_wei = weights[:, ipol]
        if param_dict["complex_split"] == "polar":
            y_data = [np.absolute(loc_vis)]
            y_label = [f"{correlation} Amplitude [arb. units]"]
            title = ["Amplitude"]
            y_data.append(
                np.angle(loc_vis)
                * convert_unit("rad", param_dict["phase_unit"], "trigonometric")
            )
            y_label.append(f'{correlation} Phase [{param_dict["phase_unit"]}]')
            title.append("Phase")
        else:
            y_data = [loc_vis.real]
            y_label = [f"Real {correlation} [arb. units]"]
            title = ["real part"]
            y_data.append(loc_vis.imag)
            y_label.append(f"Imaginary {correlation} [arb. units]")
            title.append("imaginary part")

        y_data.append(loc_wei)
        y_label.append(f"{correlation} weights [arb. units]")
        title.append("weights")

        fig, ax = create_figure_and_axes(param_dict["figure_size"], [3, 3])
        for isplit in range(3):
            scatter_plot(
                ax[isplit, 0],
                time,
                param_dict["time_label"],
                y_data[isplit],
                y_label[isplit],
                f"Time vs {correlation} {title[isplit]}",
                data_marker=param_dict["marker"],
                data_linestyle=param_dict["linestyle"],
                data_color=param_dict["color"],
            )
            scatter_plot(
                ax[isplit, 1],
                lm[:, 0],
                param_dict["l_label"],
                y_data[isplit],
                y_label[isplit],
                f"L vs {correlation} {title[isplit]}",
                data_marker=param_dict["marker"],
                data_linestyle=param_dict["linestyle"],
                data_color=param_dict["color"],
            )
            scatter_plot(
                ax[isplit, 2],
                lm[:, 1],
                param_dict["m_label"],
                y_data[isplit],
                y_label[isplit],
                f"M vs {correlation} {title[isplit]}",
                data_marker=param_dict["marker"],
                data_linestyle=param_dict["linestyle"],
                data_color=param_dict["color"],
            )

        plotfile = (
            f'{param_dict["destination"]}/holog_directional_cosines_{correlation}_{param_dict["this_map"]}_'
            f'{param_dict["this_ant"]}_{param_dict["this_ddi"]}.png'
        )
        close_figure(
            fig,
            f"Channel averaged {correlation} vs Directional Cosines",
            plotfile,
            param_dict["dpi"],
            param_dict["display"],
        )
    else:

        logger.warning(
            f'Correlation {correlation} is not present for {param_dict["this_ant"]} {param_dict["this_ddi"]} '
            f'{param_dict["this_map"]}, skipping...'
        )
    return


def plot_sky_coverage_chunk(parm_dict):
    """
    Plot the sky coverage for a XDS
    Args:
        parm_dict: Parameter dictionary from the caller function enriched with the XDS data

    Returns:
    PNG file with the sky coverage
    """

    combined = parm_dict["combined"]
    antenna = parm_dict["this_ant"]
    destination = parm_dict["destination"]

    if combined:
        export_name = f"{destination}/position_sky_coverage_{antenna}.png"
        suptitle = f'Sky coverage for antenna {antenna.split("_")[1]}'
    else:
        ddi = parm_dict["this_ddi"]
        export_name = f"{destination}/position_sky_coverage_{antenna}_{ddi}.png"
        suptitle = (
            f'Sky coverage for antenna {antenna.split("_")[1]}, DDI {ddi.split("_")[1]}'
        )

    xds = parm_dict["xds_data"]
    figuresize = parm_dict["figure_size"]
    angle_unit = parm_dict["angle_unit"]
    time_unit = parm_dict["time_unit"]
    display = parm_dict["display"]
    dpi = parm_dict["dpi"]
    antenna_info = xds.attrs["antenna_info"]

    time = xds.time.values * convert_unit("day", time_unit, "time")
    angle_fact = convert_unit("rad", angle_unit, "trigonometric")
    ha = xds["HOUR_ANGLE"] * angle_fact
    dec = xds["DECLINATION"] * angle_fact
    ele = xds["ELEVATION"] * angle_fact

    fig, axes = create_figure_and_axes(figuresize, [2, 2])

    elelim, elelines, declim, declines, halim = compute_plot_borders(
        angle_fact, antenna_info["latitude"], xds.attrs["elevation_limit"]
    )
    timelabel = f"Time from observation start [{time_unit}]"
    halabel = f"Hour Angle [{angle_unit}]"
    declabel = f"Declination [{angle_unit}]"
    scatter_plot(
        axes[0, 0],
        time,
        timelabel,
        ele,
        f"Elevation [{angle_unit}]",
        "Time vs Elevation",
        ylim=elelim,
        hlines=elelines,
    )
    scatter_plot(
        axes[0, 1], time, timelabel, ha, halabel, "Time vs Hour angle", ylim=halim
    )
    scatter_plot(
        axes[1, 0],
        time,
        timelabel,
        dec,
        declabel,
        "Time vs Declination",
        ylim=declim,
        hlines=declines,
    )
    scatter_plot(
        axes[1, 1],
        ha,
        halabel,
        dec,
        declabel,
        "Hour angle vs Declination",
        ylim=declim,
        xlim=halim,
        hlines=declines,
    )

    close_figure(fig, suptitle, export_name, dpi, display)
    return


def plot_delays_chunk(parm_dict):
    """
    Plot the delays and optionally the delay model for a XDS
    Args:
        parm_dict: Parameter dictionary from the caller function enriched with the XDS data

    Returns:
    PNG file with the delay plots
    """
    combined = parm_dict["combined"]
    plot_model = parm_dict["plot_model"]
    antenna = parm_dict["this_ant"]
    destination = parm_dict["destination"]
    if combined:
        export_name = f'{destination}/position_delays_{antenna}_combined_{parm_dict["comb_type"]}.png'
        suptitle = f'Delays for antenna {antenna.split("_")[1]}'
    else:
        ddi = parm_dict["this_ddi"]
        export_name = f"{destination}/position_delays_{antenna}_separated_{ddi}.png"
        suptitle = (
            f'Delays for antenna {antenna.split("_")[1]}, DDI {ddi.split("_")[1]}'
        )

    xds = parm_dict["xds_data"]
    figuresize = parm_dict["figure_size"]
    angle_unit = parm_dict["angle_unit"]
    time_unit = parm_dict["time_unit"]
    delay_unit = parm_dict["delay_unit"]
    display = parm_dict["display"]
    dpi = parm_dict["dpi"]
    antenna_info = xds.attrs["antenna_info"]

    time = xds.time.values * convert_unit("day", time_unit, "time")
    angle_fact = convert_unit("rad", angle_unit, "trigonometric")
    delay_fact = convert_unit("sec", delay_unit, kind="time")
    ha = xds["HOUR_ANGLE"] * angle_fact
    dec = xds["DECLINATION"] * angle_fact
    ele = xds["ELEVATION"] * angle_fact
    delays = xds["DELAYS"].values * delay_fact

    elelim, elelines, declim, declines, halim = compute_plot_borders(
        angle_fact, antenna_info["latitude"], xds.attrs["elevation_limit"]
    )
    delay_minmax = [np.min(delays), np.max(delays)]
    delay_border = 0.05 * (delay_minmax[1] - delay_minmax[0])
    delaylim = [delay_minmax[0] - delay_border, delay_minmax[1] + delay_border]

    fig, axes = create_figure_and_axes(figuresize, [2, 2])

    ylabel = f"Delays [{delay_unit}]"
    if plot_model:
        model = xds["MODEL"].values * delay_fact
    else:
        model = None
    scatter_plot(
        axes[0, 0],
        time,
        f"Time from observation start [{time_unit}]",
        delays,
        ylabel,
        "Time vs Delays",
        ylim=delaylim,
        model=model,
    )
    scatter_plot(
        axes[0, 1],
        ele,
        f"Elevation [{angle_unit}]",
        delays,
        ylabel,
        "Elevation vs Delays",
        xlim=elelim,
        vlines=elelines,
        ylim=delaylim,
        model=model,
    )
    scatter_plot(
        axes[1, 0],
        ha,
        f"Hour Angle [{angle_unit}]",
        delays,
        ylabel,
        "Hour Angle vs Delays",
        xlim=halim,
        ylim=delaylim,
        model=model,
    )
    scatter_plot(
        axes[1, 1],
        dec,
        f"Declination [{angle_unit}]",
        delays,
        ylabel,
        "Declination vs Delays",
        xlim=declim,
        vlines=declines,
        ylim=delaylim,
        model=model,
    )

    close_figure(fig, suptitle, export_name, dpi, display)
    return


def compute_plot_borders(angle_fact, latitude, elevation_limit):
    """
    Compute plot limits and position of lines to be added to the plots
    Args:
        angle_fact: Angle scaling unit factor
        latitude: Antenna latitude
        elevation_limit: The elevation limit in the data set

    Returns:
    Elevation limits, elevation lines, declination limits, declination lines and hour angle limits
    """
    latitude *= angle_fact
    elevation_limit *= angle_fact
    right_angle = pi / 2 * angle_fact
    border = 0.05 * right_angle
    elelim = [-border, right_angle + border]
    border *= 2
    declim = [-border - right_angle + latitude, right_angle + border]
    border *= 2
    halim = [-border, 4 * right_angle + border]
    elelines = [0, elevation_limit]  # lines at zero and elevation limit
    declines = [latitude - right_angle, latitude + right_angle]
    return elelim, elelines, declim, declines, halim


def plot_position_corrections(parm_dict, data_dict):
    """
    Plot the position corrections on top of an array configuration plot
    Args:
        parm_dict: Calling function parameter dictionary
        data_dict: The MDS contents

    Returns:
    PNG file(s) with the correction plots
    """
    telescope = Telescope(data_dict._meta_data["telescope_name"])
    destination = parm_dict["destination"]
    ref_ant = data_dict._meta_data["reference_antenna"]
    combined = parm_dict["combined"]

    ant_list = param_to_list(parm_dict["ant"], data_dict, "ant")
    if combined:
        filename = f'{destination}/position_corrections_combined_{data_dict._meta_data["combine_ddis"]}.png'
        attribute_list = []
        for ant in ant_list:
            attribute_list.append(data_dict[ant].attrs)
        plot_corrections_sub(attribute_list, filename, telescope, ref_ant, parm_dict)

    else:
        ddi_list = []
        if parm_dict["ddi"] == "all":
            for ant in ant_list:
                ddi_list.extend(data_dict[ant].keys())
            ddi_list = np.unique(ddi_list)
        else:
            ddi_list = parm_dict["ddi"]
            for i_ddi in range(len(ddi_list)):
                ddi_list[i_ddi] = "ddi_" + ddi_list[i_ddi]
        for ddi in ddi_list:
            filename = f"{destination}/position_corrections_separated_{ddi}.png"
            attribute_list = []
            for ant in ant_list:
                if ddi in data_dict[ant].keys():
                    attribute_list.append(data_dict[ant][ddi].attrs)
            plot_corrections_sub(
                attribute_list, filename, telescope, ref_ant, parm_dict
            )


def plot_corrections_sub(attributes_list, filename, telescope, ref_ant, parm_dict):
    """
    Does the actual individual position correction plots
    Args:
        attributes_list: List of XDS attributes
        filename: Name of the PNG file to be created
        telescope: Telescope object used in observations
        ref_ant: Reference antenna in the data set
        parm_dict: Parameter dictionary of the caller's caller

    Returns:
    PNG file with the position corrections plot
    """
    tel_lon, tel_lat, tel_rad = get_telescope_lat_lon_rad(telescope)
    length_unit = parm_dict["unit"]
    scaling = parm_dict["scaling"]
    len_fac = convert_unit("m", length_unit, "length")
    corr_fac = clight * scaling
    figure_size = parm_dict["figure_size"]
    box_size = parm_dict["box_size"]
    dpi = parm_dict["dpi"]
    display = parm_dict["display"]

    xlabel = f"East [{length_unit}]"
    ylabel = f"North [{length_unit}]"

    fig, axes = create_figure_and_axes(figure_size, [2, 2], default_figsize=[8, 8])
    xy_whole = axes[0, 0]
    xy_inner = axes[0, 1]
    z_whole = axes[1, 0]
    z_inner = axes[1, 1]

    for attributes in attributes_list:
        antenna = attributes["antenna_info"]
        ew_off, ns_off, _, _ = compute_antenna_relative_off(
            antenna, tel_lon, tel_lat, tel_rad, len_fac
        )
        corrections, _ = rotate_to_gmt(
            np.copy(attributes["position_fit"]),
            attributes["position_error"],
            antenna["longitude"],
        )
        corrections = np.array(corrections) * corr_fac
        text = "  " + antenna["name"]
        if antenna["name"] == ref_ant:
            text += "*"
        plot_antenna_position(
            xy_whole, xy_inner, ew_off, ns_off, text, box_size, marker="+"
        )
        plot_corrections(
            xy_whole, xy_inner, ew_off, ns_off, corrections[0], corrections[1], box_size
        )
        plot_antenna_position(
            z_whole, z_inner, ew_off, ns_off, text, box_size, marker="+"
        )
        plot_corrections(z_whole, z_inner, ew_off, ns_off, 0, corrections[2], box_size)

    plot_boxes_limits_and_labels(
        xy_whole,
        xy_inner,
        xlabel,
        ylabel,
        box_size,
        "X & Y, outer array",
        "X & Y, inner array",
    )
    plot_boxes_limits_and_labels(
        z_whole, z_inner, xlabel, ylabel, box_size, "Z, outer array", "Z, inner array"
    )
    close_figure(fig, "Position corrections", filename, dpi, display)


def plot_antenna_chunk(parm_dict):
    """
    Chunk function for the user facing function plot_antenna
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    destination = parm_dict["destination"]
    plot_type = parm_dict["plot_type"]
    basename = f"{destination}/{antenna}_{ddi}"
    xds = parm_dict["xds_data"]
    telescope = Telescope(xds.attrs["telescope_name"])
    surface = AntennaSurface(xds, telescope, reread=True)
    if plot_type == plot_types[0]:  # deviation plot
        surface.plot_deviation(basename, "panel", parm_dict)
    elif plot_type == plot_types[1]:  # phase plot
        surface.plot_phase(basename, "panel", parm_dict)
    elif plot_type == plot_types[2]:  # Ancillary plot
        surface.plot_mask(basename, "panel", parm_dict)
        surface.plot_amplitude(basename, "panel", parm_dict)
    else:  # all plots
        surface.plot_deviation(basename, "panel", parm_dict)
        surface.plot_phase(basename, "panel", parm_dict)
        surface.plot_mask(basename, "panel", parm_dict)
        surface.plot_amplitude(basename, "panel", parm_dict)


def plot_aperture_chunk(parm_dict):
    """
    Chunk function for the user facing function plot_apertures
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    destination = parm_dict["destination"]
    input_xds = parm_dict["xds_data"]
    input_xds.attrs["AIPS"] = False
    telescope = Telescope.from_xds(input_xds)

    asked_pol_states = parm_dict["polarization_state"]
    avail_pol_states = input_xds.pol.values
    if asked_pol_states == "all":
        plot_pol_states = avail_pol_states
    elif type(asked_pol_states) is str:
        plot_pol_states = [asked_pol_states]
    elif type(asked_pol_states) is list:
        plot_pol_states = asked_pol_states
    else:
        msg = f"Uncomprehensible polarization state: {asked_pol_states}"
        logger.error(msg)
        raise Exception(msg)

    for pol_state in plot_pol_states:
        if pol_state in avail_pol_states:
            surface = AntennaSurface(
                input_xds,
                telescope,
                nan_out_of_bounds=False,
                pol_state=str(pol_state),
                clip_type="absolute",
                clip_level=0,
            )
            basename = f"{destination}/{antenna}_{ddi}_pol_{pol_state}"
            surface.plot_phase(basename, "image_aperture", parm_dict)
            surface.plot_deviation(basename, "image_aperture", parm_dict)
            surface.plot_amplitude(basename, "image_aperture", parm_dict)
        else:
            logger.warning(f"Polarization state {pol_state} not available in data")


def plot_beam_chunk(parm_dict):
    """
    Chunk function for the user facing function plot_beams
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    destination = parm_dict["destination"]
    input_xds = parm_dict["xds_data"]
    laxis = input_xds.l.values * convert_unit(
        "rad", parm_dict["angle_unit"], "trigonometric"
    )
    maxis = input_xds.m.values * convert_unit(
        "rad", parm_dict["angle_unit"], "trigonometric"
    )
    if input_xds.sizes["chan"] != 1:
        raise Exception("Only single channel holographies supported")

    if input_xds.sizes["time"] != 1:
        raise Exception("Only single mapping holographies supported")

    full_beam = input_xds.BEAM.isel(time=0, chan=0).values
    pol_axis = input_xds.pol.values

    for i_pol, pol in enumerate(pol_axis):
        basename = f"{destination}/{antenna}_{ddi}_pol_{pol}"
        plot_beam_by_pol(laxis, maxis, pol, full_beam[i_pol, ...], basename, parm_dict)


def plot_beam_by_pol(laxis, maxis, pol, beam_image, basename, parm_dict):
    """
    Plot a beam
    Args:
        laxis: L axis
        maxis: M axis
        pol: Polarization state
        beam_image: Beam data
        basename: Basename for output file
        parm_dict: dictionary with general and plotting parameters
    """

    fig, axes = create_figure_and_axes(parm_dict["figure_size"], [1, 2])
    norm_z_label = f"Z Scale [Normalized]"
    x_label = f'L axis [{parm_dict["angle_unit"]}]'
    y_label = f'M axis [{parm_dict["angle_unit"]}]'

    if parm_dict["complex_split"] == "cartesian":
        vmin = np.min([np.nanmin(beam_image.real), np.nanmin(beam_image.imag)])
        vmax = np.max([np.nanmax(beam_image.real), np.nanmax(beam_image.imag)])
        simple_imshow_map_plot(
            axes[0],
            fig,
            laxis,
            maxis,
            beam_image.real,
            "Real part",
            parm_dict["colormap"],
            [vmin, vmax],
            x_label=x_label,
            y_label=y_label,
            z_label=norm_z_label,
        )
        simple_imshow_map_plot(
            axes[1],
            fig,
            laxis,
            maxis,
            beam_image.imag,
            "Imaginary part",
            parm_dict["colormap"],
            [vmin, vmax],
            x_label=x_label,
            y_label=y_label,
            z_label=norm_z_label,
        )
    else:
        scale = convert_unit("rad", parm_dict["phase_unit"], "trigonometric")
        amplitude = np.absolute(beam_image)
        phase = np.angle(beam_image) * scale

        simple_imshow_map_plot(
            axes[0],
            fig,
            laxis,
            maxis,
            amplitude,
            "Amplitude",
            parm_dict["colormap"],
            [np.nanmin(amplitude[amplitude > 1e-8]), np.nanmax(amplitude)],
            x_label=x_label,
            y_label=y_label,
            z_label=norm_z_label,
        )
        simple_imshow_map_plot(
            axes[1],
            fig,
            laxis,
            maxis,
            phase,
            "Phase",
            parm_dict["colormap"],
            [-np.pi * scale, np.pi * scale],
            x_label=x_label,
            y_label=y_label,
            z_label=f"Phase [{parm_dict['phase_unit']}]",
        )

    plot_name = add_prefix(
        add_prefix(basename, parm_dict["complex_split"]), "image_beam"
    )
    suptitle = (
        f'Beam for Antenna: {parm_dict["this_ant"].split("_")[1]}, DDI: {parm_dict["this_ddi"].split("_")[1]}, '
        f"pol. State: {pol}"
    )
    close_figure(fig, suptitle, plot_name, parm_dict["dpi"], parm_dict["display"])
    return


def plot_zernike_model_chunk(parm_dict):
    """
    Chunk function for the user facing function plot_zernike_model
    Args:
        parm_dict: the parameter dict containing the parameters for the plot and the data.

    Returns:
        Plots of the Zernike models along with residuals in png files inside destination.
    """
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    destination = parm_dict["destination"]
    input_xds = parm_dict["xds_data"]

    if input_xds.sizes["chan"] != 1:
        raise Exception("Only single channel holographies supported")

    if input_xds.sizes["time"] != 1:
        raise Exception("Only single mapping holographies supported")

    # Data retrieval
    u_axis = input_xds.u.values
    v_axis = input_xds.v.values
    pol_axis = input_xds.pol.values
    corr_axis = input_xds.orig_pol.values
    zernike_model = input_xds.ZERNIKE_MODEL.isel(time=0, chan=0).values
    aperture = input_xds.APERTURE.values
    zernike_n_order = input_xds.attrs["zernike_N_order"]
    corr_aperture = convert_5d_grid_from_stokes(aperture, pol_axis, corr_axis)[
        0, 0, :, :, :
    ]
    suptitle = (
        f"Zernike model with N<={zernike_n_order} for {create_dataset_label(antenna, ddi, ',')} "
        f"correlation: "
    )

    for icorr, corr in enumerate(corr_axis):
        filename = f"{destination}/image_zernike_model_{antenna}_{ddi}_corr_{corr}.png"
        _plot_zernike_aperture_model(
            suptitle + f"{corr}",
            corr_aperture[icorr],
            u_axis,
            v_axis,
            zernike_model[icorr],
            filename,
            parm_dict,
        )

    return


def _plot_cartesian_component(
    ax, fig, aperture, model, u_axis, v_axis, colormap, comp_label
):
    maxabs = np.nanmax(np.abs(aperture))
    zlim = [-maxabs, maxabs]
    residuals = aperture - model
    nvalid = np.sum(np.isfinite(model))
    rms = np.sqrt(np.nansum(residuals**2)) / nvalid
    simple_imshow_map_plot(
        ax[0],
        fig,
        u_axis,
        v_axis,
        aperture,
        f"Aperture {comp_label} part",
        colormap,
        zlim,
        z_label="EM intensity",
    )
    simple_imshow_map_plot(
        ax[1],
        fig,
        u_axis,
        v_axis,
        model,
        f"Model {comp_label} part",
        colormap,
        zlim,
        z_label="EM intensity",
    )
    simple_imshow_map_plot(
        ax[2],
        fig,
        u_axis,
        v_axis,
        residuals,
        f"Residuals {comp_label} part, RMS={rms:.5f}",
        colormap,
        zlim,
        z_label="EM intensity",
    )


def _plot_zernike_aperture_model(
    suptitle, aperture, u_axis, v_axis, model_aperture, filename, parm_dict
):
    fig, ax = create_figure_and_axes(parm_dict["figure_size"], [2, 3])
    _plot_cartesian_component(
        ax[0],
        fig,
        aperture.real,
        model_aperture.real,
        u_axis,
        v_axis,
        parm_dict["colormap"],
        "real",
    )
    _plot_cartesian_component(
        ax[1],
        fig,
        aperture.imag,
        model_aperture.imag,
        u_axis,
        v_axis,
        parm_dict["colormap"],
        "imaginary",
    )
    close_figure(
        fig,
        suptitle,
        filename,
        parm_dict["dpi"],
        parm_dict["display"],
        tight_layout=True,
    )
