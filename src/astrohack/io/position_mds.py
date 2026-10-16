import pathlib

import numpy as np
import toolviper.utils.logger as logger
import toolviper.utils.parameter

from astrohack.antenna.telescope import get_proper_telescope
from astrohack.io.locit_mds import AstrohackLocitFile
from astrohack.utils.algorithms import (
    compute_antenna_relative_off,
    data_statistics,
    rotate_to_gmt,
)
from astrohack.utils.constants import (
    clight,
    notavail,
    pi,
    twopi,
)
from astrohack.utils.conversion import convert_unit
from astrohack.utils.graph import create_and_execute_graphs_for_outputs
from astrohack.utils.text import (
    add_prefix,
    create_pretty_table,
    fixed_format_error,
    lnbr,
    param_to_list,
    string_to_ascii_file,
    undscr,
)
from astrohack.utils.tools import get_telescope_lat_lon_rad
from astrohack.utils.validation import custom_unit_checker
from astrohack.visualization.array_cfg_plot import (
    define_inner_box_size,
    plot_one_antenna_position,
)
from astrohack.visualization.plot_tools import (
    close_figure,
    create_figure_and_axes,
    plot_boxes_limits_and_labels,
    scatter_plot,
)


class AstrohackPositionFile(AstrohackLocitFile):
    """Data class for position data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackPositionFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPositionFile object
        :rtype: AstrohackPositionFile
        """
        super().__init__(file=file)

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def export_locit_fit_results(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        position_unit: str = "m",
        time_unit: str = "hour",
        delay_unit: str = "nsec",
        phase_unit: str = "deg",
    ) -> None:
        """Export antenna position fit results to a text file.

        :param destination: Name of the destination folder to contain exported fit results
        :type destination: str

        :param ant: List of antennas/antenna to be exported, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be exported, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param position_unit: Unit to list position fit results, defaults to 'm'
        :type position_unit: str, optional

        :param time_unit: Unit for time in position fit results, defaults to 'hour'
        :type time_unit: str, optional

        :param delay_unit: Unit for delays, defaults to 'nsec'
        :type delay_unit: str, optional

        :param phase_unit: Unit for phasess, defaults to 'deg'
        :type phase_unit: str, optional

        .. _Description:

        Produce a text file with the fit results from astrohack.locit for better determination of antenna locations.
        """

        param_dict = locals()
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)

        len_fact = convert_unit("m", position_unit, "length")
        del_fact = convert_unit("sec", delay_unit, kind="time")
        pha_fact = convert_unit("rad", phase_unit, kind="trigonometric")
        pos_fact = len_fact * clight
        combined = self.root.attrs["combined"]
        input_pars = self.root.attrs["input_parameters"]

        if combined:
            field_names = [
                "Antenna",
                "Station",
                f"RMS [{delay_unit}]",
                f"RMS [{phase_unit}]",
                f"F. delay [{delay_unit}]",
                f"X offset [{position_unit}]",
                f"Y offset [{position_unit}]",
                f"Z offset [{position_unit}]",
            ]
            specifier = f"combined_{self.root.attrs['combine_specifier']}"

        else:
            field_names = [
                "Antenna",
                "Station",
                "DDI",
                f"RMS [{delay_unit}]",
                f"RMS [{phase_unit}]",
                f"F. delay [{delay_unit}]",
                f"X offset [{position_unit}]",
                f"Y offset [{position_unit}]",
                f"Z offset [{position_unit}]",
            ]
            specifier = "separated_ddis"
        kterm_present = input_pars["fit_kterm"]
        rate_present = input_pars["fit_delay_rate"]
        if kterm_present:
            field_names.extend([f"K offset [{position_unit}]"])
        if rate_present:
            slo_unit = f"{delay_unit}/{time_unit}"
            slo_fact = del_fact / convert_unit("day", time_unit, "time")
            field_names.extend([f"Rate [{slo_unit}]"])
        else:
            slo_unit = notavail
            slo_fact = 1.0

        table = create_pretty_table(field_names)
        telescope = get_proper_telescope(self.root.attrs["telescope_name"])
        full_antenna_list = telescope.antenna_list
        selected_antenna_list = param_to_list(ant, self, "ant")

        for ant_name in full_antenna_list:
            ant_key = add_prefix(ant_name, "ant")
            if ant_name == self.root.attrs["reference_antenna"]:
                ant_name += " (ref)"

            if ant_key in selected_antenna_list:
                if ant_key in self.keys():
                    antenna = self[ant_key]
                    if combined:
                        row = [ant_name, antenna.attrs["antenna_info"]["station"]]
                        table.add_row(
                            _export_position_xds_to_table_row(
                                row,
                                antenna.attrs,
                                del_fact,
                                pha_fact,
                                pos_fact,
                                slo_fact,
                                position_unit,
                                delay_unit,
                                kterm_present,
                                rate_present,
                            )
                        )
                    else:
                        ddi_list = param_to_list(ddi, self[ant_key], "ddi")
                        for ddi_key in ddi_list:
                            row = [
                                ant_name,
                                antenna[ddi_key].attrs["antenna_info"]["station"],
                                ddi_key.split(undscr)[1],
                            ]
                            table.add_row(
                                _export_position_xds_to_table_row(
                                    row,
                                    antenna[ddi_key].attrs,
                                    del_fact,
                                    pha_fact,
                                    pos_fact,
                                    slo_fact,
                                    position_unit,
                                    delay_unit,
                                    kterm_present,
                                    rate_present,
                                )
                            )

        print(table.get_string())
        string_to_ascii_file(
            table.get_string(),
            f"{destination}/position_{specifier}_fit_results.txt",
        )

    @toolviper.utils.parameter.validate()
    def export_results_to_parminator(
        self,
        filename: str,
        ant: str | list[str] = "all",
        ddi: int | None = None,
        correction_threshold: float | int = 0.01,
    ) -> None:
        """Export antenna position fit results to a VLA parminator file.

        :param filename: Name of the parminator file to be created
        :type filename: str

        :param ant: List of antennas/antenna to be exported, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be exported, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param correction_threshold: Correction threshold in meters to include an antenna position correction in output.
        :type correction_threshold: float, optional

        .. _Description:

        Produce a VLA parminator compatible text file with the fit results from astrohack.locit.
        """
        param_dict = locals()
        combined = self.root.attrs["combined"]
        input_pars = self.root.attrs["input_parameters"]

        if (not combined) and (not isinstance(ddi, int)):
            msg = "If position file contains multiple DDIs one must be specified."
            logger.error(msg)
            raise ValueError(msg)

        kterm_present = input_pars["fit_kterm"]

        telescope = get_proper_telescope(self.root.attrs["telescope_name"])
        full_antenna_list = telescope.antenna_list
        selected_antenna_list = param_to_list(ant, self, "ant")
        threshold = correction_threshold

        parmstr = ""
        for ant_name in full_antenna_list:
            ant_key = add_prefix(ant_name, "ant")

            if ant_key in selected_antenna_list:
                if ant_key in self.keys():
                    if combined:
                        position_xds = self[ant_key]
                    else:
                        position_xds = self[ant_key][f"ddi_{ddi}"]

                    parmstr += _export_position_xds_to_parminator(
                        position_xds.attrs, threshold, kterm_present
                    )

        string_to_ascii_file(parmstr, filename)

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def plot_sky_coverage(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        time_unit: str = "hour",
        angle_unit: str = "deg",
        display: bool = False,
        figure_size: tuple | list[float | int] | np.ndarray | None = None,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """Plot the sky coverage of the data used for antenna position fitting

        :param destination: Name of the destination folder to contain the plots
        :type destination: str

        :param ant: List of antennas/antenna to be plotted, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param angle_unit: Unit for angle in plots, defaults to 'deg'
        :type angle_unit: str, optional

        :param time_unit: Unit for time in plots, defaults to 'hour'
        :type time_unit: str, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot size in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: plot resolution in pixels per inch, default is 300
        :type dpi: int, optional

        :param parallel: If True will use an existing astrohack client to produce plots in parallel, default is False
        :type parallel: bool, optional

        .. _Description:

        This method produces 4 plots for each selected antenna and DDI. These plots are:
        1) Time vs Elevation
        2) Time vs Hour Angle
        3) Time vs Declination
        4) Hour Angle vs Declination

        These plots are intended to display the coverage of the sky of the fitted data

        """

        param_dict = locals()
        param_dict["combined"] = self.root.attrs["combined"]

        if self.root.attrs["combined"]:
            key_order = ["ant"]
        else:
            key_order = ["ant", "ddi"]
        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_sky_coverage_chunk,
            param_dict=param_dict,
            key_order=key_order,
        )

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def plot_delays(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        time_unit: str = "hour",
        angle_unit: str = "deg",
        delay_unit: str = "nsec",
        delay_limits: tuple | list[float | int] | np.ndarray | None = None,
        plot_model: bool = True,
        display: bool = False,
        figure_size: tuple | list[float | int] | np.ndarray | None = None,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """Plot the delays used for antenna position fitting and optionally the resulting fit.

        :param destination: Name of the destination folder to contain the plots
        :type destination: str

        :param ant: List of antennas/antenna to be plotted, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param angle_unit: Unit for angle in plots, defaults to 'deg'
        :type angle_unit: str, optional

        :param time_unit: Unit for time in plots, defaults to 'hour'
        :type time_unit: str, optional

        :param delay_unit: Unit for delay in plots, defaults to 'nsec'
        :type delay_unit: str, optional

        :param delay_limits: Y Limits for the delay plots, default is None (i.e. guess from data)
        :type delay_limits: numpy.ndarray, list, tuple, optional

        :param plot_model: Plot the fitted model results alongside the data.
        :type plot_model: bool, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot size in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: plot resolution in pixels per inch, default is 300
        :type dpi: int, optional

        :param parallel: If True will use an existing astrohack client to produce plots in parallel, default is False
        :type parallel: bool, optional

        .. _Description:

        This method produces 4 plots for each selected antenna and DDI. These plots are:
        1) Time vs Delays
        2) Elevation vs Delays
        3) Hour Angle vs Delays
        4) Declination vs Delays

        These plots are intended to display the gain variation with the 4 relevant parameters for the fitting and also
        asses the quality of the position fit.

        """

        param_dict = locals()

        param_dict["combined"] = self.root.attrs["combined"]
        param_dict["comb_type"] = self.root.attrs["combine_specifier"]
        if self.root.attrs["combined"]:
            key_order = ["ant"]
        else:
            key_order = ["ant", "ddi"]
        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_delays_chunk,
            param_dict=param_dict,
            key_order=key_order,
        )

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def plot_position_corrections(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        unit: str = "km",
        box_size: int | float = None,
        scaling: int | float = 250,
        figure_size: tuple | list[float | int] | np.ndarray | None = None,
        display: bool = False,
        dpi: int = 300,
    ) -> None:
        """Plot Antenna position corrections on an array configuration plot

        :param destination: Name of the destination folder to contain plot
        :type destination: str

        :param ant: Select which antennas are to be plotted, defaults to all when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param unit: Unit for the plot, valid values are length units, default is km
        :type unit: str, optional

        :param box_size: Size of the box for plotting the inner part of the array in unit, default is 5 km
        :type box_size: int, float, optional

        :param scaling: scaling factor to plotting the corrections, default is 250
        :type scaling: int, float, optional

        :param display: Display plots inline or suppress, defaults to False
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        .. _Description:

        Plot the position corrections computed by locit on top of an array configuration plot.
        The corrections are too small to be visualized on the array plot since they are of the order of mm and the array
        is usually spread over km, or at least hundreds of meters.
        The scaling factor is used to bring the corrections to a scale discernible on the plot, this plot should not be
        used to estimate correction values, for that purpose use export_locit_fit_results instead.

        """

        param_dict = locals()
        pathlib.Path(param_dict["destination"]).mkdir(exist_ok=True)

        combined = self.root.attrs["combined"]
        telescope = get_proper_telescope(
            self.root.attrs["telescope_name"],
        )
        ref_ant = self.root.attrs["reference_antenna"]

        ant_list = param_to_list(ant, self, "ant")
        if combined:
            filename = (
                f"{destination}/position_corrections_combined_"
                + f"{self.root.attrs['combine_specifier']}.png"
            )
            attribute_list = []
            for ant in ant_list:
                attribute_list.append(self[ant].attrs)
            _plot_antenna_position_corrections_sub(
                attribute_list, filename, telescope, ref_ant, param_dict
            )

        else:
            ddi_list = []
            if ddi == "all":
                for ant in ant_list:
                    ddi_list.extend(self[ant].keys())
                ddi_list = np.unique(ddi_list)
            elif isinstance(ddi, list):
                ddi_list = ddi
                for i_ddi in range(len(ddi_list)):
                    ddi_list[i_ddi] = f"ddi_{ddi_list[i_ddi]}"
            else:
                ddi_list = [f"ddi_{ddi}"]
            for ddi in ddi_list:
                filename = f"{destination}/position_corrections_separated_{ddi}.png"
                attribute_list = []
                for ant in ant_list:
                    if ddi in self[ant].keys():
                        attribute_list.append(self[ant][ddi].attrs)
                _plot_antenna_position_corrections_sub(
                    attribute_list, filename, telescope, ref_ant, param_dict
                )


def _plot_sky_coverage_chunk(parm_dict):
    """
    Plot the sky coverage for a XDS
    Args:
        parm_dict: Parameter dictionary from the caller function enriched with the XDS data

    Returns:
    PNG file with the sky coverage
    """

    ant_xdt = parm_dict["xdt_data"]
    combined = parm_dict["combined"]
    antenna = parm_dict["this_ant"]
    destination = parm_dict["destination"]

    if combined:
        export_name = f"{destination}/position_sky_coverage_{antenna}.png"
        suptitle = f"Sky coverage for antenna {antenna.split(undscr)[1]}"
    else:
        ddi = parm_dict["this_ddi"]
        export_name = f"{destination}/position_sky_coverage_{antenna}_{ddi}.png"
        suptitle = f"Sky coverage for antenna {antenna.split(undscr)[1]}, DDI {ddi.split(undscr)[1]}"

    figuresize = parm_dict["figure_size"]
    angle_unit = parm_dict["angle_unit"]
    time_unit = parm_dict["time_unit"]
    display = parm_dict["display"]
    dpi = parm_dict["dpi"]
    antenna_info = ant_xdt.attrs["antenna_info"]

    time = ant_xdt.time.values * convert_unit("day", time_unit, "time")
    angle_fact = convert_unit("rad", angle_unit, "trigonometric")
    ha = ant_xdt["HOUR_ANGLE"] * angle_fact
    dec = ant_xdt["DECLINATION"] * angle_fact
    ele = ant_xdt["ELEVATION"] * angle_fact

    fig, axes = create_figure_and_axes(figuresize, [2, 2])

    elelim, elelines, declim, declines, halim = _compute_plot_borders(
        angle_fact, antenna_info["latitude"], ant_xdt.attrs["elevation_limit"]
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
        add_legend=False,
    )
    scatter_plot(
        axes[0, 1],
        time,
        timelabel,
        ha,
        halabel,
        "Time vs Hour angle",
        ylim=halim,
        add_legend=False,
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
        add_legend=False,
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
        add_legend=False,
    )

    close_figure(fig, suptitle, export_name, dpi, display)
    return


def _add_scans_to_plot(ax, x_axis, scans, residuals, sigma_clip=3):
    """
    Add Scan information to an axes object that has time as the X axis.
    Args:
        ax: axes object to add Scan information to
        x_axis: X axis values for scans
        scans: Scan IDs
        residuals: residual values for times and scans
        sigma_clip: Level for sigma clipping, default is 3

    Returns:
        None
    """
    residual_stats = data_statistics(residuals)
    bad_scan_selection = np.abs(residuals) > sigma_clip * residual_stats["rms"]
    tick_positions = []
    tick_labels = []
    for i_x, x_point in enumerate(x_axis):
        if bad_scan_selection[i_x]:
            tick_positions.append(x_point)
            tick_labels.append(str(scans[i_x]))

    if len(tick_positions) > 0:
        bad_scan_fontsize = 8
        sec_x_ax = ax.secondary_xaxis("top")
        sec_x_ax.set_xticks(tick_positions)
        sec_x_ax.set_xticklabels(tick_labels, fontsize=bad_scan_fontsize)
        sec_x_ax.set_xlabel("Bad Scans", fontsize=bad_scan_fontsize)
    return


def _plot_delays_chunk(parm_dict):
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
        export_name = f"{destination}/position_delays_{antenna}_combined_{parm_dict['comb_type']}.png"
        suptitle = f"Delays for antenna {antenna.split(undscr)[1]}"
    else:
        ddi = parm_dict["this_ddi"]
        export_name = f"{destination}/position_delays_{antenna}_separated_{ddi}.png"
        suptitle = (
            f"Delays for antenna {antenna.split(undscr)[1]}, DDI {ddi.split(undscr)[1]}"
        )

    ant_xdt = parm_dict["xdt_data"]
    figuresize = parm_dict["figure_size"]
    angle_unit = parm_dict["angle_unit"]
    time_unit = parm_dict["time_unit"]
    delay_unit = parm_dict["delay_unit"]
    display = parm_dict["display"]
    dpi = parm_dict["dpi"]
    antenna_info = ant_xdt.attrs["antenna_info"]

    time = ant_xdt.time.values * convert_unit("day", time_unit, "time")
    angle_fact = convert_unit("rad", angle_unit, "trigonometric")
    delay_fact = convert_unit("sec", delay_unit, kind="time")
    ha = ant_xdt["HOUR_ANGLE"] * angle_fact
    dec = ant_xdt["DECLINATION"] * angle_fact
    ele = ant_xdt["ELEVATION"] * angle_fact
    delays = ant_xdt["DELAYS"].values * delay_fact
    scans = ant_xdt["SCANS"].values

    elelim, elelines, declim, declines, halim = _compute_plot_borders(
        angle_fact, antenna_info["latitude"], ant_xdt.attrs["elevation_limit"]
    )
    if parm_dict["delay_limits"] is None:
        delay_minmax = [np.min(delays), np.max(delays)]
        delay_border = 0.05 * (delay_minmax[1] - delay_minmax[0])
        delaylim = [delay_minmax[0] - delay_border, delay_minmax[1] + delay_border]
    else:
        delaylim = parm_dict["delay_limits"]

    fig, axes = create_figure_and_axes(figuresize, [2, 2])

    ylabel = f"Delays [{delay_unit}]"
    if plot_model:
        model = ant_xdt["MODEL"].values * delay_fact
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
    _add_scans_to_plot(axes[0, 0], time, scans, delays - model)

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


def _export_position_xds_to_table_row(
    row,
    attributes,
    del_fact,
    pha_fact,
    pos_fact,
    slo_fact,
    pos_unit,
    del_unit,
    kterm_present,
    rate_present,
):
    """
    Export the data from a single X array DataSet attributes to a table row (a list)
    Args:
        row: row onto which the data results are to be added
        attributes: The XDS attributes dictionary
        del_fact: Delay unit scaling factor
        pos_fact: Position unit scaling factor
        slo_fact: Delay rate unit scaling factor
        kterm_present: Is the elevation axis offset term present?
        rate_present: Is the delay rate term present?
        pha_fact: phase unit scaling factor
        pos_unit: Position unit
        del_unit: Delay unit

    Returns:
    The filled table row
    """

    delay_rms = np.sqrt(attributes["chi_squared"])
    mean_freq = np.nanmean(attributes["frequency"])
    phase_rms = twopi * mean_freq * delay_rms
    row.append(f"{delay_rms * del_fact:4.2e}")
    row.append(f"{phase_rms * pha_fact:5.1f}")

    sig_scale_pos = convert_unit("mm", pos_unit, "length")
    sig_scale_del = 1e-3 * convert_unit("nsec", del_unit, "time")

    row.append(
        fixed_format_error(
            attributes["fixed_delay_fit"],
            attributes["fixed_delay_error"],
            del_fact,
            sig_scale_del,
        )
    )
    position, poserr = rotate_to_gmt(
        np.copy(attributes["position_fit"]),
        attributes["position_error"],
        attributes["antenna_info"]["longitude"],
    )

    for i_pos in range(3):
        row.append(
            fixed_format_error(position[i_pos], poserr[i_pos], pos_fact, sig_scale_pos)
        )
    if kterm_present:
        row.append(
            fixed_format_error(
                attributes["koff_fit"],
                attributes["koff_error"],
                pos_fact,
                sig_scale_pos,
            )
        )
    if rate_present:
        row.append(
            fixed_format_error(
                attributes["rate_fit"],
                attributes["rate_error"],
                slo_fact,
                sig_scale_del,
            )
        )
    return row


def _export_position_xds_to_parminator(attributes, threshold, kterm_present):
    """
    Export a position xds attributes to a string ingestible by VLA's parminator
    :param attributes: xds attributes
    :param threshold: threshold of valid corrections in meters
    :param kterm_present: include K term in the parminator output
    :return: string Formated for parminator output
    """
    axes = ["X", "Y", "Z"]
    delays, _ = rotate_to_gmt(
        np.copy(attributes["position_fit"]),
        attributes["position_error"],
        attributes["antenna_info"]["longitude"],
    )
    station = attributes["antenna_info"]["station"]

    outstr = ""
    for iaxis, delay in enumerate(delays):
        correction = delay * clight
        if np.abs(correction) > threshold:
            outstr += f"{station}, ,{axes[iaxis]},${correction: .4f}{lnbr}"

    if kterm_present:
        correction = attributes["koff_fit"] * clight
        if np.abs(correction) > threshold:
            outstr += f"{station}, ,K,${correction: .4f}{lnbr}"
    return outstr


def _plot_antenna_position_corrections_sub(
    attributes_list, filename, telescope, ref_ant, parm_dict
):
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
    dpi = parm_dict["dpi"]
    display = parm_dict["display"]

    xlabel = f"East [{length_unit}]"
    ylabel = f"North [{length_unit}]"

    fig, axes = create_figure_and_axes(figure_size, [2, 2], default_figsize=[8, 8])
    xy_whole = axes[0, 0]
    xy_inner = axes[0, 1]
    z_whole = axes[1, 0]
    z_inner = axes[1, 1]

    antenna_offsets = []
    antenna_corrections = []
    antenna_texts = []

    for attributes in attributes_list:
        antenna = attributes["antenna_info"]
        ew_off, ns_off, _, _ = compute_antenna_relative_off(
            antenna, tel_lon, tel_lat, tel_rad, len_fac
        )
        this_corrections, _ = rotate_to_gmt(
            np.copy(attributes["position_fit"]),
            attributes["position_error"],
            antenna["longitude"],
        )
        this_corrections = np.array(this_corrections) * corr_fac
        this_text = "  " + antenna["name"]
        if antenna["name"] == ref_ant:
            this_text += "*"
        antenna_offsets.append([ew_off, ns_off])
        antenna_texts.append(this_text)
        antenna_corrections.append(this_corrections)

    box_size = define_inner_box_size(parm_dict["box_size"], np.array(antenna_offsets))
    for i_ant, this_corrections in enumerate(antenna_corrections):
        this_text = antenna_texts[i_ant]
        ew_off, ns_off = antenna_offsets[i_ant]
        plot_one_antenna_position(
            xy_whole, xy_inner, ew_off, ns_off, this_text, box_size, marker="+"
        )
        _add_antenna_position_corrections_to_plot(
            xy_whole,
            xy_inner,
            ew_off,
            ns_off,
            this_corrections[0],
            this_corrections[1],
            box_size,
        )
        plot_one_antenna_position(
            z_whole, z_inner, ew_off, ns_off, this_text, box_size, marker="+"
        )
        _add_antenna_position_corrections_to_plot(
            z_whole, z_inner, ew_off, ns_off, 0, this_corrections[2], box_size
        )

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


def _add_antenna_position_corrections_to_plot(
    outerax, innerax, xpos, ypos, xcorr, ycorr, box_size, color="red", linewidth=0.5
):
    """
    Plot an antenna position corrections as a vector from the antenna position
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


def _compute_plot_borders(angle_fact, latitude, elevation_limit):
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
