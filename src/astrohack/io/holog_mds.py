from copy import deepcopy
from datetime import date

import numpy as np
import toolviper.utils.parameter
from astropy.time import Time
from toolviper.utils import logger as logger

from astrohack.io.base_mds import AstrohackBaseFile
from astrohack.utils.algorithms import compute_average_stokes_visibilities
from astrohack.utils.constants import fontsize, markersize
from astrohack.utils.conversion import convert_unit
from astrohack.utils.graph import create_and_execute_graphs_for_outputs
from astrohack.utils.text import create_informative_label_from_summary, lnbr
from astrohack.utils.validation import custom_plots_checker, custom_unit_checker
from astrohack.visualization.observation_summary import (
    generate_observation_summary,
)
from astrohack.visualization.plot_tools import (
    close_figure,
    create_figure_and_axes,
    scatter_plot,
)


class AstrohackHologFile(AstrohackBaseFile):
    """Data class for holog data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackHologFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackHologFile object
        :rtype: AstrohackHologFile
        """
        super().__init__(file=file)

    @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    def plot_diagnostics(
        self,
        destination: str,
        delta: float = 0.01,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        map_id: int | list[int] | str = "all",
        complex_split: str = "polar",
        display: bool = False,
        figure_size: tuple | list[float] | np.ndarray | None = None,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """ Plot diagnostic calibration plots from the holography data file.

        :param destination: Name of the destination folder to contain diagnostic plots
        :type destination: str

        :param delta: Defines a fraction of cell_size around which to look for peaks., defaults to 0.01
        :type delta: float, optional

        :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param map_id: map ID to use in subselection. This relates to which antenna are in the mapping vs. scanning \
        configuration,  defaults to "all" when None, ex. 0
        :type map_id: list or int, optional

        :param complex_split: How to split complex data, cartesian (real + imaginary) or polar (amplitude + phase), \
        default is polar
        :type complex_split: str, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        :param parallel: Run in parallel, defaults to False
        :type parallel: bool, optional

        **Additional Information**
        The visibilities extracted by extract_holog are complex due to the nature of interferometric measurements. To
        ease the visualization of the complex data it can be split into real and imaginary parts (cartesian) or in
        amplitude and phase (polar).

        .. rubric:: Available complex splitting possibilities:
        - *cartesian*: Split is done to a real part and an imaginary part in the plots
        - *polar*:     Split is done to an amplitude and a phase in the plots

        """

        param_dict = locals()
        param_dict["map"] = map_id

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_diagnostics_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi", "map"],
        )

    @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    def plot_lm_sky_coverage(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        map_id: int | list[int] | str = "all",
        angle_unit: str = "deg",
        time_unit: str = "hour",
        plot_correlation: str | list[str] | None = None,
        complex_split: str = "polar",
        phase_unit: str = "deg",
        display: bool = False,
        figure_size: tuple | list[float] | np.ndarray | None = None,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """ Plot directional cosine coverage.

        :param destination: Name of the destination folder to contain plots
        :type destination: str

        :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param map_id: map ID to use in subselection. This relates to which antenna are in the mapping vs. scanning \
        configuration,  defaults to "all" when None, ex. 0
        :type map_id: list or int, optional

        :param angle_unit: Unit for L and M axes in plots, default is 'deg'.
        :type angle_unit: str, optional

        :param time_unit: Unit for time axis in plots, default is 'hour'.
        :type time_unit: str, optional

        :param plot_correlation: Which correlation to plot against L and M, default is None (no correlation plots).
        :type plot_correlation: str, list, optional

        :param complex_split: How to split complex data, cartesian (real + imaginary) or polar (amplitude + phase), \
        default is polar
        :type complex_split: str, optional

        :param phase_unit: Unit for phase in 'polar' plots, default is 'deg'.
        :type phase_unit: str

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        :param parallel: Run in parallel, defaults to False
        :type parallel: bool, optional

        **Additional Information**
        The visibilities extracted by extract_holog are complex due to the nature of interferometric measurements. To
        ease the visualization of the complex data it can be split into real and imaginary parts (cartesian) or in
        amplitude and phase (polar).

        .. rubric:: Available complex splitting possibilities:
        - *cartesian*: Split is done to a real part and an imaginary part in the plots
        - *polar*:     Split is done to an amplitude and a phase in the plots

        .. rubric:: Plotting correlations:
        - *RR, RL, LR, LL*: Are available for circular systems
        - *XX, XY, YX, YY*: Are available for linear systems
        - *all*: Plot all correlations in dataset

        """

        param_dict = locals()
        param_dict["map"] = map_id

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_lm_sky_coverage_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi", "map"],
        )
        return

    @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    def export_to_aips(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        map_id: int | list[int] | str = "all",
        parallel: bool = False,
    ) -> None:
        """ Export data compatible to AIPS's HOLOG task

        :param destination: Name of the destination folder to contain ASCII files
        :type destination: str

        :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param map_id: map ID to use in subselection. This relates to which antenna are in the mapping vs. scanning \
        configuration,  defaults to "all" when None, ex. 0
        :type map_id: list or int, optional

        :param parallel: Run in parallel, defaults to False
        :type parallel: bool, optional

        **Additional Information**

        This method converts the data for an Antenna mapping to the ASCII format used by AIPS's HOLOG task.
        Currently only stokes I is supported.
        """
        param_dict = locals()
        param_dict["map"] = map_id

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_export_to_aips_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi", "map"],
        )
        return

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def observation_summary(
        self,
        summary_file: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        map_id: int | list[int] | str = "all",
        az_el_key: str = "center",
        phase_center_unit: str = "radec",
        az_el_unit: str = "deg",
        time_format: str = "%d %h %Y, %H:%M:%S",
        tab_size: int = 3,
        print_summary: bool = True,
        parallel: bool = False,
    ) -> None:
        """ Create a Summary of observation information

        :param summary_file: Text file to put the observation summary
        :type summary_file: str

        :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param map_id: map ID to use in subselection. This relates to which antenna are in the mapping vs. scanning \
        configuration,  defaults to "all" when None, ex. 0
        :type map_id: list or int, optional

        :param az_el_key: What type of Azimuth & Elevation information to print, 'mean', 'median' or 'center', default\
        is 'center'
        :type az_el_key: str, optional

        :param phase_center_unit: What unit to display phase center coordinates, 'radec' and angle units supported, \
        default is 'radec'
        :type phase_center_unit: str, optional

        :param az_el_unit: Angle unit used to display Azimuth & Elevation information, default is 'deg'
        :type az_el_unit: str, optional

        :param time_format: datetime time format for the start and end dates of observation, default is \
        "%d %h %Y, %H:%M:%S"
        :type time_format: str, optional

        :param tab_size: Number of spaces in the tab levels, default is 3
        :type tab_size: int, optional

        :param print_summary: Print the summary at the end of execution, default is True
        :type print_summary: bool, optional

        :param parallel: Run in parallel, defaults to False
        :type parallel: bool, optional

        **Additional Information**

        This method produces a summary of the data in the AstrohackHologFile displaying general information,
        spectral information and suggested beam image characteristics.
        """
        param_dict = locals()
        param_dict["map"] = map_id
        generate_observation_summary(
            mds_object=self,
            param_dict=param_dict,
            key_order=["ant", "ddi", "map"],
            summary_type="holog",
        )


def _extract_indices(laxis, maxis, squared_radius):
    indices = []

    assert laxis.shape[0] == maxis.shape[0], "l, m must be same size."

    for i in range(laxis.shape[0]):
        squared_sum = np.power(laxis[i], 2) + np.power(maxis[i], 2)
        if squared_sum <= squared_radius:
            indices.append(i)

    return np.array(indices)


def _plot_diagnostics_chunk(param_dict):
    xds_data = param_dict["xdt_data"].dataset
    delta = param_dict["delta"]
    complex_split = param_dict["complex_split"]
    display = param_dict["display"]
    figuresize = param_dict["figure_size"]
    destination = param_dict["destination"]
    dpi = param_dict["dpi"]
    thisfont = 1.2 * fontsize

    unix_conversion = 3506716800

    radius = np.power(xds_data.attrs["summary"]["beam"]["cell size"] * delta, 2)

    l_axis = xds_data.DIRECTIONAL_COSINES.values[..., 0]
    m_axis = xds_data.DIRECTIONAL_COSINES.values[..., 1]

    assert l_axis.shape[0] == m_axis.shape[0], "l, m dimensions don't match!"

    indices = _extract_indices(laxis=l_axis, maxis=m_axis, squared_radius=radius)

    if complex_split == "cartesian":
        vis_dict = {
            "data": [
                xds_data.isel(time=indices).VIS.real,
                xds_data.isel(time=indices).VIS.imag,
            ],
            "polarization": [0, 3],
            "label": ["REAL", "IMAG"],
        }
    else:
        vis_dict = {
            "data": [
                xds_data.isel(time=indices).map(np.abs).VIS,
                xds_data.isel(time=indices).map(np.angle).VIS,
            ],
            "polarization": [0, 3],
            "label": ["AMP", "PHASE"],
        }

    times = np.unique(
        Time(vis_dict["data"][0].time.data - unix_conversion, format="unix").iso
    )

    fig, axis = create_figure_and_axes(figuresize, [4, 1], sharex=True)

    chan = np.arange(0, xds_data.chan.data.shape[0])

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
                    f"Vis ({vis_dict['label'][i]}; {xds_data.pol.values[pol]})",
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

    plotfile = (
        f"{destination}/holog_diagnostics_{param_dict['this_ant']}_"
        f"{param_dict['this_ddi']}_{param_dict['this_map']}.png"
    )
    title = "Cal. check for: " + create_informative_label_from_summary(
        xds_data.attrs["summary"], "deg", add_date=False
    )
    close_figure(fig, title, plotfile, dpi, display, tight_layout=False)


def _plot_lm_sky_coverage_chunk(param_dict):
    xdt_data = param_dict["xdt_data"]
    angle_fact = convert_unit("rad", param_dict["angle_unit"], "trigonometric")
    real_lm = xdt_data["DIRECTIONAL_COSINES"] * angle_fact
    ideal_lm = xdt_data["IDEAL_DIRECTIONAL_COSINES"] * angle_fact
    time = deepcopy(xdt_data.time.values)
    time -= time[0]
    time *= convert_unit("sec", param_dict["time_unit"], "time")
    param_dict["l_label"] = f"L [{param_dict['angle_unit']}]"
    param_dict["m_label"] = f"M [{param_dict['angle_unit']}]"
    param_dict["time_label"] = (
        f"Time from observation start [{param_dict['time_unit']}]"
    )

    param_dict["marker"] = "."
    param_dict["linestyle"] = "-"
    param_dict["color"] = "blue"

    _plot_lm_coverage_sub(
        time, real_lm, ideal_lm, param_dict, xdt_data.attrs["summary"]
    )

    if (
        param_dict["plot_correlation"] is None
        or param_dict["plot_correlation"] == "None"
    ):
        pass
    else:
        param_dict["linestyle"] = ""
        visi = np.average(xdt_data["VIS"].values, axis=1)
        weights = np.average(xdt_data["WEIGHT"].values, axis=1)
        pol_axis = xdt_data.pol.values
        if isinstance(param_dict["plot_correlation"], (list, tuple)):
            for correlation in param_dict["plot_correlation"]:
                _plot_correlation_sub(
                    visi,
                    weights,
                    correlation,
                    pol_axis,
                    time,
                    real_lm,
                    param_dict,
                    xdt_data.attrs["summary"],
                )
        else:
            if param_dict["plot_correlation"] == "all":
                for correlation in pol_axis:
                    _plot_correlation_sub(
                        visi,
                        weights,
                        correlation,
                        pol_axis,
                        time,
                        real_lm,
                        param_dict,
                        xdt_data.attrs["summary"],
                    )
            else:
                _plot_correlation_sub(
                    visi,
                    weights,
                    param_dict["plot_correlation"],
                    pol_axis,
                    time,
                    real_lm,
                    param_dict,
                    xdt_data.attrs["summary"],
                )


def _plot_lm_coverage_sub(time, real_lm, ideal_lm, param_dict, summary):
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
        add_legend=False,
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
        add_legend=False,
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
        add_legend=False,
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
        add_legend=False,
    )
    plotfile = (
        f"{param_dict['destination']}/holog_directional_cosines_"
        f"{param_dict['this_ant']}_{param_dict['this_ddi']}_{param_dict['this_map']}.png"
    )
    title = "L&M for: " + create_informative_label_from_summary(summary, "deg")
    close_figure(fig, title, plotfile, param_dict["dpi"], param_dict["display"])


def _plot_correlation_sub(
    visi, weights, correlation, pol_axis, time, lm, param_dict, summary
):
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
            y_label.append(f"{correlation} Phase [{param_dict['phase_unit']}]")
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
                add_legend=False,
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
                add_legend=False,
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
                add_legend=False,
            )

        plotfile = (
            f"{param_dict['destination']}/holog_directional_cosines_{correlation}_"
            f"{param_dict['this_ant']}_{param_dict['this_ddi']}_{param_dict['this_map']}.png"
        )
        title = f"{correlation} vs L&M for: " + create_informative_label_from_summary(
            summary, "deg"
        )
        close_figure(
            fig,
            title,
            plotfile,
            param_dict["dpi"],
            param_dict["display"],
        )
    else:
        logger.warning(
            f"Correlation {correlation} is not present for {param_dict['this_ant']} {param_dict['this_ddi']} "
            f"{param_dict['this_map']}, skipping..."
        )
    return


def _export_to_aips_chunk(param_dict):
    xds_data = param_dict["xdt_data"]
    stokes = "I"
    stokes_vis = compute_average_stokes_visibilities(xds_data, stokes)
    filename = (
        f"{param_dict['destination']}/holog_visibilities_{param_dict['this_ant']}_"
        f"{param_dict['this_ddi']}_{param_dict['this_map']}.txt"
    )
    ant_num = xds_data.attrs["summary"]["general"]["antenna name"].split("a")[1]
    cmt = "#! "

    today = date.today().strftime("%y%m%d")
    outstr = (
        cmt
        + f"RefAnt = ** Antenna = {ant_num} Stokes = '{stokes}_' Freq =  {stokes_vis.attrs['frequency']:.9f}"
        f" DATE-OBS = '{today}'{lnbr}"
    )
    outstr += cmt + "MINsamp =   0  Npoint =   1\n"
    outstr += cmt + "IFnumber =   2   Channel =    32.0\n"
    outstr += cmt + "TimeRange = -99,  0,  0,  0,  999,  0,  0,  0\n"
    outstr += cmt + "Averaged Ref-Ants = 10, 15,\n"
    outstr += cmt + "DOCAL = T  DOPOL =-1\n"
    outstr += cmt + "BCHAN=     4 ECHAN=    60 CHINC=  1 averaged\n"
    outstr += (
        cmt
        + "   LL             MM             AMPLITUDE      PHASE         SIGMA(AMP)   SIGMA(PHASE)\n"
    )
    lm = xds_data["DIRECTIONAL_COSINES"].values
    amp = stokes_vis["AMPLITUDE"].values
    pha = stokes_vis["PHASE"].values
    sigma_amp = stokes_vis["SIGMA_AMP"]
    sigma_pha = stokes_vis["SIGMA_PHA"]
    for i_time in range(len(xds_data.time)):
        if np.isfinite(sigma_amp[i_time]):
            outstr += (
                f"{lm[i_time, 0]:15.7f}{lm[i_time, 1]:15.7f}{amp[i_time]:15.7f}{pha[i_time]:15.7f}"
                f"{sigma_amp[i_time]:15.7f}{sigma_pha[i_time]:15.7f}\n"
            )
    outstr += f"{cmt}Average number samples per point =   1.000"

    with open(filename, "w") as outfile:
        outfile.write(outstr)

    return
