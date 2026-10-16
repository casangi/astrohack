import numpy as np
from toolviper.utils.parameter import validate

from astrohack.utils.conversion import convert_unit, to_antenna_tapering
from astrohack.utils.graph import create_and_execute_graphs_for_outputs
from astrohack.utils.text import (
    create_informative_label_from_summary,
    create_pretty_table,
    format_value_unit,
    lnbr,
    spc,
)
from astrohack.utils.validation import custom_plots_checker, custom_unit_checker
from astrohack.visualization.observation_summary import (
    generate_observation_summary,
)
from astrohack.visualization.plot_tools import (
    close_figure,
    create_figure_and_axes,
    scatter_plot,
    set_y_axis_lims_from_default,
)

from .base_mds import AstrohackBaseFile


class AstrohackBeamcutFile(AstrohackBaseFile):
    """Data class for beam cut data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackBeamcutFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackBeamcutFile object
        :rtype: AstrohackBeamcutFile
        """
        super().__init__(file=file)

    @validate(custom_checker=custom_unit_checker)
    def observation_summary(
        self,
        summary_file: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        az_el_key: str = "center",
        phase_center_unit: str = "radec",
        az_el_unit: str = "deg",
        time_format: str = "%d %h %Y, %H:%M:%S",
        tab_size: int = 3,
        print_summary: bool = True,
        parallel: bool = False,
    ) -> None:
        """
        Create a Summary of observation information

        :param summary_file: Text file to put the observation summary
        :type summary_file: str

        :param ant: antenna ID to use in subselection, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: data description ID to use in subselection, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

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

        :return: None
        :rtype: NoneType

        **Additional Information**

        This method produces a summary of the data in the AstrohackBeamcutFile displaying general information,
        spectral information, beam image characteristics and aperture image characteristics.
        """
        generate_observation_summary(
            mds_object=self,
            param_dict=locals(),
            key_order=["ant", "ddi"],
            summary_type="beamcut",
        )

    @validate(custom_checker=custom_plots_checker)
    def plot_in_amplitude(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        lm_unit: str = "amin",
        azel_unit: str = "deg",
        y_scale: list[float | int] | None = None,
        display: bool = False,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """
        Plot beamcuts contained in the beamcut_mds in amplitude

        :param destination: Directory into which to save plots.
        :type destination: str

        :param ant: Antenna ID to use in subselection, e.g. ea25, defaults to "all".
        :type ant: list or str, optional

        :param ddi: Data description ID to use in subselection, e.g. 0, defaults to "all".
        :type ddi: list or int, optional

        :param lm_unit: Unit for L/M offsets, default is "amin".
        :type lm_unit: str, optional

        :param azel_unit: Unit for Az/El information, default is "deg".
        :type azel_unit: str, optional

        :param y_scale: Set the y scale for the plots.
        :type y_scale: str, optional

        :param display: Display plots during execution, default is False.
        :type display: bool, optional

        :param dpi: Pixel resolution for plots, default is 300.
        :type dpi: int, optional

        :param parallel: Run in parallel, defaults to False.
        :type parallel: bool, optional

        :return: None
        :rtype: NoneType
        """
        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_beamcut_in_amplitude_chunk,
            param_dict=locals(),
            key_order=["ant", "ddi"],
        )
        return

    @validate(custom_checker=custom_plots_checker)
    def plot_in_tapering(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        lm_unit: str = "amin",
        azel_unit: str = "deg",
        y_scale: list[float | int] | tuple[float | int] | None = None,
        display: bool = False,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """
        Plot beamcuts contained in the beamcut_mds in tapering

        :param destination: Directory into which to save plots.
        :type destination: str

        :param ant: Antenna ID to use in subselection, e.g. ea25, defaults to "all".
        :type ant: list or str, optional

        :param ddi: Data description ID to use in subselection, e.g. 0, defaults to "all".
        :type ddi: list or int, optional

        :param lm_unit: Unit for L/M offsets, default is "amin".
        :type lm_unit: str, optional

        :param azel_unit: Unit for Az/El information, default is "deg".
        :type azel_unit: str, optional

        :param y_scale: Set the y scale for the plots.
        :type y_scale: str, optional

        :param display: Display plots during execution, default is False.
        :type display: bool, optional

        :param dpi: Pixel resolution for plots, default is 300.
        :type dpi: int, optional

        :param parallel: Run in parallel, defaults to False.
        :type parallel: bool, optional

        :return: None
        :rtype: NoneType
        """

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_beamcut_in_tapering_chunk,
            param_dict=locals(),
            key_order=["ant", "ddi"],
        )
        return

    @validate(custom_checker=custom_plots_checker)
    def plot_lm_offsets(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        lm_unit: str = "amin",
        azel_unit: str = "deg",
        display: bool = False,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """
        Plot beamcuts contained in the beamcut_mds over the sky

        :param destination: Directory into which to save plots.
        :type destination: str

        :param ant: Antenna ID to use in subselection, e.g. ea25, defaults to "all".
        :type ant: list or str, optional

        :param ddi: Data description ID to use in subselection, e.g. 0, defaults to "all".
        :type ddi: list or int, optional

        :param lm_unit: Unit for L/M offsets, default is "amin".
        :type lm_unit: str, optional

        :param azel_unit: Unit for Az/El information, default is "deg".
        :type azel_unit: str, optional

        :param display: Display plots during execution, default is False.
        :type display: bool, optional

        :param dpi: Pixel resolution for plots, default is 300.
        :type dpi: int, optional

        :param parallel: Run in parallel, defaults to False.
        :type parallel: bool, optional

        :return: None
        :rtype: NoneType
        """

        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_cuts_in_lm_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )
        return

    @validate(custom_checker=custom_plots_checker)
    def plot_in_phase(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        lm_unit: str = "amin",
        azel_unit: str = "deg",
        phase_unit: str = "deg",
        phase_scale: list[float] | tuple[float] | np.ndarray | None = None,
        display: bool = False,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """
        Plot beamcuts contained in the beamcut_mds in phase

        :param destination: Directory into which to save plots.
        :type destination: str

        :param ant: Antenna ID to use in subselection, e.g. ea25, defaults to "all".
        :type ant: list or str, optional

        :param ddi: Data description ID to use in subselection, e.g. 0, defaults to "all".
        :type ddi: list or int, optional

        :param lm_unit: Unit for L/M offsets, default is "amin".
        :type lm_unit: str, optional

        :param azel_unit: Unit for Az/El information, default is "deg".
        :type azel_unit: str, optional

        :param phase_unit: Unit for the phase plots, default is "deg".
        :type phase_unit: str, optional

        :param phase_scale: Scale for the phase plots, in phase_unit, default is None, meaning 1 full cycle.
        :type phase_scale: Union[List[float], Tuple[float], np.array], optional

        :param display: Display plots during execution, default is False.
        :type display: bool, optional

        :param dpi: Pixel resolution for plots, default is 300.
        :type dpi: int, optional

        :param parallel: Run in parallel, defaults to False.
        :type parallel: bool, optional

        :return: None
        :rtype: NoneType
        """

        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_beamcut_in_phase_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )
        return

    @validate(custom_checker=custom_plots_checker)
    def export_report(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        lm_unit: str = "amin",
        azel_unit: str = "deg",
        parallel: bool = False,
    ) -> None:
        """
        Create reports on the parameters of the gaussians fitted to the beamcut.

        :param destination: Directory into which to save the reports.
        :type destination: str

        :param ant: Antenna ID to use in subselection, e.g. ea25, defaults to "all".
        :type ant: list or str, optional

        :param ddi: Data description ID to use in subselection, e.g. 0, defaults to "all".
        :type ddi: list or int, optional

        :param lm_unit: Unit for L/M offsets, default is "amin".
        :type lm_unit: str, optional

        :param azel_unit: Unit for Az/El information, default is "deg".
        :type azel_unit: str, optional

        :param parallel: run in parallel, defaults to False.
        :type parallel: bool, optional

        :return: None
        :rtype: NoneType
        """

        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_create_report_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )
        return


def _plot_beamcut_in_amplitude_chunk(par_dict):
    """
    Produce Amplitude beam cut plots from a xdtree containing beam cuts.

    :param par_dict: Paremeter dictionary controlling plot aspects
    :type par_dict: dict

    :return: None
    :rtype: NoneType
    """
    cut_xdtree = par_dict["xdt_data"]
    n_cuts = len(cut_xdtree.children.values())
    # Loop over cuts
    fig, axes = create_figure_and_axes(
        [12, 1 + n_cuts * 4], [n_cuts, 2], force_2d_axes_array=True
    )
    for icut, cut_xds in enumerate(cut_xdtree.children.values()):
        _plot_single_cut_in_amplitude(cut_xds, axes[icut, :], par_dict)

    # Header creation
    summary = cut_xdtree.attrs["summary"]
    title = _create_beamcut_header(summary, par_dict)

    filename = _file_name_factory("amplitude", par_dict)
    close_figure(fig, title, filename, par_dict["dpi"], par_dict["display"])


def _plot_beamcut_in_tapering_chunk(par_dict):
    """
    Produce tapering beam cut plots from a xdtree containing beam cuts.

    :param par_dict: Paremeter dictionary controlling plot aspects
    :type par_dict: dict


    :return: None
    :rtype: NoneType
    """
    cut_xdtree = par_dict["xdt_data"]
    n_cuts = len(cut_xdtree.children.values())
    # Loop over cuts
    fig, axes = create_figure_and_axes(
        [6, 1 + n_cuts * 4], [n_cuts, 1], force_2d_axes_array=True
    )
    for icut, cut_xds in enumerate(cut_xdtree.children.values()):
        _plot_single_cut_in_tapering(cut_xds, axes[icut, 0], par_dict)

    # Header creation
    summary = cut_xdtree.attrs["summary"]
    title = _create_beamcut_header(summary, par_dict)

    filename = _file_name_factory("db", par_dict)
    close_figure(fig, title, filename, par_dict["dpi"], par_dict["display"])


def _plot_cuts_in_lm_chunk(par_dict):
    """
    Produce plot of LM offsets in all cuts for antenna and ddi xd tree.

    :param par_dict: Paremeter dictionary controlling plot aspects
    :type par_dict: dict


    :return: None
    :rtype: NoneType
    """
    cut_xdtree = par_dict["xdt_data"]
    _plot_cuts_in_lm_sub(cut_xdtree, par_dict)


def _plot_beamcut_in_phase_chunk(par_dict):
    """
    Produce phase beam cut plots from a xdtree containing beam cuts.

    :param par_dict: Paremeter dictionary controlling plot aspects
    :type par_dict: dict

    :return: None
    :rtype: NoneType
    """
    cut_xdtree = par_dict["xdt_data"]
    n_cuts = len(cut_xdtree.children.values())
    # Loop over cuts
    fig, axes = create_figure_and_axes(
        [12, 1 + n_cuts * 4], [n_cuts, 2], force_2d_axes_array=True
    )
    for icut, cut_xds in enumerate(cut_xdtree.children.values()):
        _plot_single_cut_in_phase(cut_xds, axes[icut, :], par_dict)

    # Header creation
    summary = cut_xdtree.attrs["summary"]
    title = _create_beamcut_header(summary, par_dict)

    filename = _file_name_factory("phase", par_dict)
    close_figure(fig, title, filename, par_dict["dpi"], par_dict["display"])
    return


def _create_report_chunk(par_dict, spacing=2, item_marker="-", precision=3):
    """
    Produce a report on beamcut fit results from a xdtree containing beam cuts.

    :param par_dict: Paremeter dictionary controlling report aspects
    :type par_dict: dict

    :param spacing: Identation
    :type spacing: int

    :param item_marker: Character to denote a different item in a list
    :type item_marker: str

    :param precision: Number of decimal places to include in table results
    :type precision: int

    :return: None
    :rtype: NoneType
    """
    cut_xdtree = par_dict["xdt_data"]
    outstr = f"{item_marker}{spc}"
    lm_unit = par_dict["lm_unit"]
    lm_fac = convert_unit("rad", lm_unit, "trigonometric")
    summary = cut_xdtree.attrs["summary"]

    items = [
        "Id",
        f"Center [{lm_unit}]",
        "Amplitude [ ]",
        f"FWHM [{lm_unit}]",
        "Tapering [dB]",
    ]
    outstr += "Beam cut for " + _create_beamcut_header(summary, par_dict) + 2 * lnbr
    for icut, cut_xds in enumerate(cut_xdtree.children.values()):
        sub_title = _make_parallel_hand_sub_title(cut_xds.attrs, icut, lm_unit)
        for i_corr, parallel_hand in enumerate(cut_xds.attrs["available_corrs"]):
            outstr += f"{spacing * spc}{item_marker}{spc}{parallel_hand} {sub_title}, Beam fit results:{lnbr}"
            table = create_pretty_table(items, "c")
            fit_pars = cut_xds.attrs[f"{parallel_hand}_amp_fit_pars"]
            centers = fit_pars[0::3]
            amps = fit_pars[1::3]
            fwhms = fit_pars[2::3]
            max_amp = np.max(cut_xds[f"{parallel_hand}_amplitude"].values)

            for i_peak in range(cut_xds.attrs[f"{parallel_hand}_n_peaks"]):
                table.add_row(
                    [
                        f"{i_peak + 1})",  # Id
                        f"{lm_fac * centers[i_peak]:.{precision}f}",  # center
                        f"{amps[i_peak]:.{precision}f}",  # Amp
                        f"{lm_fac * fwhms[i_peak]:.{precision}f}",  # FWHM
                        f"{to_antenna_tapering(amps[i_peak] / max_amp):.{precision}f}",  # Tapering
                    ]
                )
            for line in table.get_string().splitlines():
                outstr += 2 * spacing * spc + line + lnbr
            outstr += lnbr

    with open(_file_name_factory("report", par_dict), "w") as outfile:
        outfile.write(outstr)


def _file_name_factory(file_type, par_dict):
    """
    Generate filenames from file type and execution parameters

    :param file_type: File type description
    :type file_type: str

    :param par_dict: Paremeter dictionary containing destination, antenna and ddi parameters
    :type par_dict: dict

    :return: Filename
    :rtype: str
    """
    destination = par_dict["destination"]
    antenna = par_dict["this_ant"]
    ddi = par_dict["this_ddi"]
    if file_type in ["db", "amplitude", "lm_offsets", "phase"]:
        ext = "png"
    elif file_type == "report":
        ext = "txt"
    else:
        raise ValueError("Invalid file type")
    return f"{destination}/beamcut_{file_type}_{antenna}_{ddi}.{ext}"


def _add_secondary_beam_hpbw_x_axis_to_plot(pb_fwhm, ax):
    """
    Add a secondary X axis on top of the figure representing the LM distances in primary beam FWHMs.

    :param pb_fwhm: Primary beam FWHM
    :type pb_fwhm: float

    :param ax: Matplotlib Axes object
    :type ax: matplotlib.axes.Axes

    :return: None
    :rtype: NoneType
    """
    if np.isnan(pb_fwhm):
        return
    sec_x_axis = ax.secondary_xaxis(
        "top", functions=(lambda x: x * 1.0, lambda xb: 1 * xb)
    )
    sec_x_axis.set_xlabel(f"Offset in Primary Beam HPBWs{lnbr}")
    sec_x_axis.set_xticks([])
    y_min, y_max = ax.get_ylim()
    x_lims = np.array(ax.get_xlim())
    pb_min, pb_max = np.ceil(x_lims / pb_fwhm)
    beam_offsets = np.arange(pb_min, pb_max, 1, dtype=int)

    for itk in beam_offsets:
        ax.axvline(itk * pb_fwhm, color="k", linestyle="--", linewidth=0.5)
        ax.text(itk * pb_fwhm, y_max, f"{itk:d}", va="bottom", ha="center")


def _add_lobe_identification_to_plot(ax, centers, peaks, y_off, y_lims):
    """
    Add gaussians identification to plot

    :param ax: Matplotlib Axes object
    :type ax: matplotlib.axes.Axes

    :param centers: Gaussian centers
    :type centers: list, numpy.array

    :param peaks: Gaussian peaks
    :type peaks: list, numpy.array

    :param y_off: Y offset to add peak Ids
    :type y_off: float

    :return: None
    :rtype: NoneType
    """
    for i_peak, peak in enumerate(peaks):
        text_y_coor = peak + y_off
        if text_y_coor < y_lims[1] - y_off:
            text_y_coor = y_lims[1] - y_off
        ax.text(
            centers[i_peak], text_y_coor, f"{i_peak + 1})", ha="center", va="bottom"
        )


def _add_beam_parameters_box(
    ax,
    pb_center,
    pb_fwhm,
    sidelobe_ratio,
    lm_unit,
    alpha=0.8,
    x_pos=0.05,
    y_pos=0.95,
    tapering_plot=False,
):
    """
    Add text bos with beam parameters

    :param ax: Matplotlib Axes object
    :type ax: matplotlib.axes.Axes

    :param pb_center: Primary beam center offset
    :type pb_center: float

    :param pb_fwhm: Primary beam FWHM
    :type pb_fwhm: float

    :param sidelobe_ratio: First side lobe ratio
    :type sidelobe_ratio: float

    :param lm_unit: L/M axis unit
    :type lm_unit: str

    :param alpha: Opacity of text box
    :type alpha: float

    :param x_pos: Relative x position of the text box
    :type x_pos: float

    :param y_pos: Relative y position of the text box
    :type y_pos: float

    :param tapering_plot: Is this a tapering plot?
    :type tapering_plot: bool

    :return: None
    :rtype: NoneType
    """
    if tapering_plot:
        head = "avg "
    else:
        head = ""
    pars_str = f"{head}PB off. = {format_value_unit(pb_center, lm_unit, 3)}{lnbr}"
    pars_str += f"{head}PB FWHM = {format_value_unit(pb_fwhm, lm_unit, 3)}{lnbr}"
    pars_str += f"{head}FSLR = {format_value_unit(to_antenna_tapering(sidelobe_ratio), 'dB', 2)}"
    bounds_box = dict(boxstyle="square", facecolor="white", alpha=alpha)
    ax.text(
        x_pos,
        y_pos,
        pars_str,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=bounds_box,
    )


def _plot_single_cut_in_amplitude(cut_xds, axes, par_dict):
    """
    Plot a single beam cut in amplitude with each correlation in a different panel

    :param cut_xds: xarray dataset containing the beamcut
    :type cut_xds: xarray.Dataset

    :param axes: numpy array with the Matplotlib Axes objects for the different panels
    :type axes: numpy.array([Matplotlib.axes.Axes])

    :param par_dict: Parameter dictionary containing plot configuration
    :type par_dict: dict

    :return: None
    :rtype: NoneType
    """
    # Init
    lm_unit = par_dict["lm_unit"]
    i_cut = int(cut_xds.name.split("_")[1])
    sub_title = _make_parallel_hand_sub_title(cut_xds.attrs, i_cut, lm_unit)
    max_amp = cut_xds.attrs["all_corr_ymax"]
    if max_amp < 1e-2:
        max_amp = 1.0
    y_off = 0.05 * max_amp
    lm_fac = convert_unit("rad", lm_unit, "trigonometric")

    # Loop over correlations
    for i_corr, parallel_hand in enumerate(cut_xds.attrs["available_corrs"]):
        # Init labels
        this_ax = axes[i_corr]
        x_data = lm_fac * cut_xds["lm_dist"].values
        y_data = cut_xds[f"{parallel_hand}_amplitude"].values
        fit_data = cut_xds[f"{parallel_hand}_amp_fit"].values
        xlabel = f"{cut_xds.attrs['xlabel']} [{lm_unit}]"
        ylabel = f"{parallel_hand} Amplitude [ ]"

        # Call plotting tool
        if cut_xds.attrs[f"{parallel_hand}_fit_succeeded"]:
            scatter_plot(
                this_ax,
                x_data,
                xlabel,
                y_data,
                ylabel,
                model=fit_data,
                model_marker="",
                title=sub_title,
                data_marker="+",
                residuals_marker=".",
                model_linestyle="-",
                data_label=f"{parallel_hand} data",
                model_label=f"{parallel_hand} fit",
                data_color="red",
                model_color="blue",
                residuals_color="black",
                legend_location="upper right",
            )

            # Add fit peak identifiers
            centers = lm_fac * np.array(
                cut_xds.attrs[f"{parallel_hand}_amp_fit_pars"][0::3]
            )
            amps = np.array(cut_xds.attrs[f"{parallel_hand}_amp_fit_pars"][1::3])

            _add_lobe_identification_to_plot(
                this_ax, centers, amps, y_off, (-y_off, max_amp + 3 * y_off)
            )
        else:
            scatter_plot(
                this_ax,
                x_data,
                xlabel,
                y_data,
                ylabel,
                title=sub_title,
                data_marker="+",
                data_label=f"{parallel_hand} data",
                data_color="red",
                legend_location="upper right",
            )

        # equalize Y scale between correlations
        set_y_axis_lims_from_default(
            this_ax, par_dict["y_scale"], (-y_off, max_amp + 3 * y_off)
        )

        _add_secondary_beam_hpbw_x_axis_to_plot(
            cut_xds.attrs[f"{parallel_hand}_pb_fwhm"] * lm_fac, this_ax
        )

        # Add bounded box with Beam parameters
        _add_beam_parameters_box(
            this_ax,
            cut_xds.attrs[f"{parallel_hand}_pb_center"] * lm_fac,
            cut_xds.attrs[f"{parallel_hand}_pb_fwhm"] * lm_fac,
            cut_xds.attrs[f"{parallel_hand}_first_side_lobe_ratio"],
            lm_unit,
        )


def _plot_single_cut_in_phase(cut_xds, axes, par_dict):
    """
    Plot a single beam cut in phase with each correlation in a different panel

    :param cut_xds: xarray dataset containing the beamcut
    :type cut_xds: xarray.Dataset

    :param axes: numpy array with the Matplotlib Axes objects for the different panels
    :type axes: numpy.array([Matplotlib.axes.Axes])

    :param par_dict: Parameter dictionary containing plot configuration
    :type par_dict: dict

    :return: None
    :rtype: NoneType
    """
    # Init
    lm_unit = par_dict["lm_unit"]
    i_cut = int(cut_xds.name.split("_")[1])
    sub_title = _make_parallel_hand_sub_title(cut_xds.attrs, i_cut, lm_unit)
    phase_unit = par_dict["phase_unit"]
    phase_fac = convert_unit("rad", phase_unit, "trigonometric")
    lm_fac = convert_unit("rad", lm_unit, "trigonometric")

    phase_scale = par_dict["phase_scale"]
    if phase_scale is None:
        phase_scale = phase_fac * np.array([-np.pi, np.pi])

    y_range = phase_scale[1] - phase_scale[0]
    y_off = 0.05 * y_range
    fit_scale = y_range / 2
    fit_offset = phase_scale[0] + y_range / 4

    # Loop over correlations
    for i_corr, parallel_hand in enumerate(cut_xds.attrs["available_corrs"]):
        # Init labels
        this_ax = axes[i_corr]
        x_data = lm_fac * cut_xds["lm_dist"].values
        y_data = phase_fac * cut_xds[f"{parallel_hand}_phase"].values
        raw_fit_data = cut_xds[f"{parallel_hand}_amp_fit"].values
        max_fit_data = np.max(raw_fit_data)
        fit_data = (fit_scale * raw_fit_data / max_fit_data) + fit_offset
        xlabel = f"{cut_xds.attrs['xlabel']} [{lm_unit}]"
        ylabel = f"{parallel_hand} Phase [{phase_unit}]"

        # Call plotting tool
        if cut_xds.attrs[f"{parallel_hand}_fit_succeeded"]:
            scatter_plot(
                this_ax,
                x_data,
                xlabel,
                y_data,
                ylabel,
                model=fit_data,
                model_marker="",
                title=sub_title,
                data_marker="+",
                model_linestyle="-",
                data_label=f"{parallel_hand} phase",
                model_label=f"{parallel_hand} amp. fit",
                hlines=[fit_offset],
                hv_linestyle="--",
                hv_color="black",
                data_color="red",
                model_color="blue",
                plot_residuals=False,
                legend_location="upper right",
            )

            # Add fit peak identifiers
            centers = lm_fac * np.array(
                cut_xds.attrs[f"{parallel_hand}_amp_fit_pars"][0::3]
            )
            amps = np.array(cut_xds.attrs[f"{parallel_hand}_amp_fit_pars"][1::3])

            _add_lobe_identification_to_plot(
                this_ax,
                centers,
                amps,
                y_off,
                phase_scale,
            )
        else:
            scatter_plot(
                this_ax,
                x_data,
                xlabel,
                y_data,
                ylabel,
                title=sub_title,
                data_marker="+",
                data_label=f"{parallel_hand} phase",
                data_color="red",
                legend_location="upper right",
            )

        this_ax.set_ylim(phase_scale)

        _add_secondary_beam_hpbw_x_axis_to_plot(
            cut_xds.attrs[f"{parallel_hand}_pb_fwhm"] * lm_fac, this_ax
        )

        # Add bounded box with Beam parameters
        _add_beam_parameters_box(
            this_ax,
            cut_xds.attrs[f"{parallel_hand}_pb_center"] * lm_fac,
            cut_xds.attrs[f"{parallel_hand}_pb_fwhm"] * lm_fac,
            cut_xds.attrs[f"{parallel_hand}_first_side_lobe_ratio"],
            lm_unit,
        )


def _plot_single_cut_in_tapering(cut_xds, ax, par_dict):
    """
    Plot a single beam cut in tapering with superposed correlations

    :param cut_xds: xarray dataset containing the beamcut
    :type cut_xds: xarray.Dataset

    :param ax: Matplotlib Axes object
    :type ax: Matplotlib.axes.Axes

    :param par_dict: Parameter dictionary containing plot configuration
    :type par_dict: dict

    :return: None
    :rtype: NoneType
    """
    i_cut = int(cut_xds.name.split("_")[1])
    lm_unit = par_dict["lm_unit"]
    sub_title = _make_parallel_hand_sub_title(cut_xds.attrs, i_cut, lm_unit)
    lm_fac = convert_unit("rad", lm_unit, "trigonometric")
    corr_colors = ["blue", "red"]

    min_tapering = 1e34
    pb_center = 0.0
    pb_fwhm = 0.0
    fsl_ratio = 0.0
    xlabel = f"{cut_xds.attrs['xlabel']} [{lm_unit}]"
    ylabel = "Tapering [dB]"

    # Loop over correlations
    n_data = 0
    for i_corr, parallel_hand in enumerate(cut_xds.attrs["available_corrs"]):
        # Init labels
        x_data = lm_fac * cut_xds["lm_dist"].values
        amps = cut_xds[f"{parallel_hand}_amplitude"].values
        max_amp = np.max(amps)
        if max_amp == 0:
            y_data = np.full_like(x_data, -1)
        else:
            y_data = to_antenna_tapering(amps / max_amp)

        y_min = np.nanmin(y_data)
        if y_min < min_tapering:
            min_tapering = y_min

        ax.plot(
            x_data,
            y_data,
            label=parallel_hand,
            color=corr_colors[i_corr],
            marker=".",
            ls="",
        )

        if not np.isnan(pb_center):
            pb_center += cut_xds.attrs[f"{parallel_hand}_pb_center"]
            pb_fwhm += cut_xds.attrs[f"{parallel_hand}_pb_fwhm"]
            fsl_ratio += cut_xds.attrs[f"{parallel_hand}_first_side_lobe_ratio"]
            n_data += 1

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(sub_title, fontsize=10)
    ax.legend(loc="upper right")

    pb_center /= n_data
    pb_fwhm /= n_data
    fsl_ratio /= n_data
    # equalize Y scale between correlations
    y_off = 0.1 * np.abs(min_tapering)
    set_y_axis_lims_from_default(ax, par_dict["y_scale"], (min_tapering - y_off, y_off))

    # Add fit peak identifiers
    first_corr = cut_xds.attrs["available_corrs"][0]

    _add_secondary_beam_hpbw_x_axis_to_plot(
        cut_xds.attrs[f"{first_corr}_pb_fwhm"] * lm_fac, ax
    )

    # Add bounded box with Beam parameters
    _add_beam_parameters_box(
        ax,
        pb_center * lm_fac,
        pb_fwhm * lm_fac,
        fsl_ratio,
        lm_unit,
        tapering_plot=True,
    )
    return


def _plot_cuts_in_lm_sub(cut_xdtree, par_dict):
    """
    Produce plot of LM offsets in all cuts for antenna and ddi xd tree.

    :param par_dict: Paremeter dictionary controlling plot aspects
    :type par_dict: dict

    :param cut_xdtree: Way to deliver a xdtree when not present in par_dict
    :type cut_xdtree: xr.DataTree

    :return: None
    :rtype: NoneType
    """
    colors = [
        "blue",
        "red",
        "green",
        "black",
        "orange",
        "grey",
        "purple",
        "brown",
        "pink",
        "olive",
        "cyan",
        "gold",
        "navy",
        "darkorange",
        "crimson",
        "forestgreen",
    ]
    lm_unit = par_dict["lm_unit"]
    lm_fac = convert_unit("rad", lm_unit, "trigonometric")

    fig, ax = create_figure_and_axes(None, [1, 1])
    for icut, cut_xds in enumerate(cut_xdtree.children.values()):
        lm_offsets = lm_fac * cut_xds["lm_offsets"].values
        ax.plot(
            lm_offsets[:, 0],
            lm_offsets[:, 1],
            label=f"cut {icut}, {cut_xds.attrs['direction']}",
            marker=".",
            ls="",
            color=colors[icut],
        )

    ax.legend(loc="best")
    ax.set_xlabel(f"L offset [{lm_unit}]")
    ax.set_ylabel(f"M offset [{lm_unit}]")
    ax.set_aspect("equal", adjustable="datalim")

    # Header creation
    summary = cut_xdtree.attrs["summary"]
    title = _create_beamcut_header(summary, par_dict)

    filename = _file_name_factory("lm_offsets", par_dict)
    close_figure(fig, title, filename, par_dict["dpi"], par_dict["display"])


def _make_parallel_hand_sub_title(attributes, i_cut, lm_unit):
    """
    Make subtitle for data based on XDS attributes.

    :param attributes: beamcut xds attributes
    :type attributes: dict

    :param i_cut: Cut number for subtitle
    :type i_cut: int

    :return: Subtitle string
    :rtype: str
    """
    direction = attributes["direction"]
    time_string = attributes["time_string"]
    exp_pb_fwhm = attributes["expected_pb_fwhm"] * convert_unit(
        "rad", lm_unit, "trigonometric"
    )
    return f"{direction}, {time_string} UTC, cut {i_cut}, exp. PB = {exp_pb_fwhm:.2f} [{lm_unit}]"


def _create_beamcut_header(summary, par_dict):
    """
    Create a data labeling header for plots and/or reports.

    :param summary: Data summary from xds attributes
    :type summary: dict

    :param par_dict: Parameter dictionary containing configuration parameters
    :type par_dict: dict

    :return: Data labeling header for plots and/or reports.
    :rtype: str
    """

    return create_informative_label_from_summary(
        summary, par_dict["azel_unit"], add_date=False, add_source=True
    )
