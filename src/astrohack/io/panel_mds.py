import numpy as np
import toolviper.utils.logger as logger
import toolviper.utils.parameter

from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.io.base_mds import AstrohackBaseFile
from astrohack.utils.constants import clight, plot_types
from astrohack.utils.conversion import convert_unit
from astrohack.utils.graph import create_and_execute_graphs_for_outputs
from astrohack.utils.text import (
    create_informative_label_from_summary,
    create_pretty_table,
    format_frequency,
    format_value_unit,
    format_wavelength,
    lnbr,
    string_to_ascii_file,
    undscr,
)
from astrohack.utils.validation import custom_plots_checker, custom_unit_checker
from astrohack.visualization.observation_summary import (
    generate_observation_summary,
)


class AstrohackPanelFile(AstrohackBaseFile):
    """Data class for panel data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackPanelFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPanelFile object
        :rtype: AstrohackPanelFile
        """
        super().__init__(file=file)

    @toolviper.utils.parameter.validate()
    def get_antenna(self, ant: str, ddi: int) -> AntennaSurface:
        """Retrieve an AntennaSurface object for interaction

        :param ant: Antenna to be retrieved, ex. ea25.
        :type ant: str

        :param ddi: DDI to be retrieved for ant_id, ex. 0
        :type ddi: int

        :return: AntennaSurface object describing for further interaction
        :rtype: AntennaSurface
        """
        ant = "ant_" + ant
        ddi = f"ddi_{ddi}"
        xds = self[ant][ddi].dataset
        return AntennaSurface(xds, reread=True)

    @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    def export_screws(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        unit: str = "mm",
        threshold: float | int | None = None,
        panel_labels: bool = True,
        display: bool = False,
        colormap: str = "RdBu_r",
        figure_size: tuple | list[float | int] | np.ndarray | None = None,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """ Export screw adjustments to text files and optionally plots.

        :param destination: Name of the destination folder to contain exported screw adjustments
        :type destination: str

        :param ant: List of antennas/antenna to be exported, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be exported, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param unit: Unit for screws adjustments, most length units supported, defaults to "mm"
        :type unit: str, optional

        :param threshold: Threshold below which data is considered negligible, value is assumed to be in the same unit\
         as the plot, if not given defaults to 10% of the maximal deviation
        :type threshold: float, optional

        :param panel_labels: Add panel labels to antenna surface plots, default is True
        :type panel_labels: bool, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param colormap: Colormap for screw adjustment map, default is RdBu_r
        :type colormap: str, optional

        :param figure_size: 2 element array/list/tuple with the screw adjustment map size in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: Screw adjustment map resolution in pixels per inch, default is 300
        :type dpi: int, optional

        :param parallel: Produce screw maps and screw adjustments in parallel
        :type parallel: bool, optional

        .. _Description:

        Produce the screw adjustments from ``astrohack.panel`` results to be used at the antenna site to improve \
        the antenna surface

        """
        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_export_screws_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )

    @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    def plot_antennas(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        plot_type: str = "deviation",
        plot_screws: bool = False,
        amplitude_limits: tuple | list[float] | np.ndarray | None = None,
        phase_unit: str = "deg",
        phase_limits: tuple | list[float] | np.ndarray | None = None,
        deviation_unit: str = "mm",
        deviation_limits: tuple | list[float] | np.ndarray | None = None,
        panel_labels: bool = False,
        display: bool = False,
        colormap: str = "viridis",
        figure_size: tuple | list[float] | np.ndarray = (8.0, 6.4),
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """ Create diagnostic plots of antenna surfaces from panel data file.

        :param destination: Name of the destination folder to contain plots
        :type destination: str

        :param ant: List of antennas/antenna to be plotted, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param plot_type: type of plot to be produced, deviation, phase, ancillary or all, default is deviation
        :type plot_type: str, optional

        :param plot_screws: Add screw positions to plot
        :type plot_screws: bool, optional

        :param amplitude_limits: Lower than Upper limit for amplitude in volts default is None (Guess from data)
        :type amplitude_limits: numpy.ndarray, list, tuple, optional

        :param phase_unit: Unit for phase plots, defaults is 'deg'
        :type phase_unit: str, optional

        :param phase_limits: Lower than Upper limit for phase, value in phase_unit, default is None (Guess from data)
        :type phase_limits: numpy.ndarray, list, tuple, optional

        :param deviation_unit: Unit for deviation plots, defaults is 'mm'
        :type deviation_unit: str, optional

        :param deviation_limits: Lower than Upper limit for deviation, value in deviation_unit, default is None (Guess \
        from data)
        :type deviation_limits: numpy.ndarray, list, tuple, optional

        :param panel_labels: Add panel labels to antenna surface plots, default is False
        :type panel_labels: bool, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param colormap: Colormap for plots, default is viridis
        :type colormap: str, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        :param parallel: If True will use an existing astrohack client to produce plots in parallel, default is False
        :type parallel: bool, optional

        .. _Description:

        Produce plots from ``astrohack.panel`` results to be analyzed to judge the quality of the results

        **Additional Information**
        .. rubric:: Available plot types:
        - *deviation*: Surface deviation estimated from phase and wavelength, three plots are produced for each antenna \
                       and ddi combination, surface before correction, the corrections applied and the corrected \
                       surface, most length units available
        - *phase*: Phase deviations over the surface, three plots are produced for each antenna and ddi combination, \
                   phase before correction, the corrections applied and the corrected phase, deg and rad available as \
                   units
        - *ancillary*: Two ancillary plots with useful information: The mask used to select data to be fitted, the \
                       amplitude data used to derive the mask, units are irrelevant for these plots
        - *all*: All the plots listed above. In this case the unit parameter is taken to mean the deviation unit, the \
                 phase unit is set to degrees
        """

        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_antenna_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )

    @toolviper.utils.parameter.validate()
    def export_to_fits(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        parallel: bool = False,
    ) -> None:
        """Export contents of an Astrohack MDS file to several FITS files in the destination folder

        :param destination: Name of the destination folder to contain plots
        :type destination: str

        :param ant: List of antennas/antenna to be plotted, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param parallel: If True will use an existing astrohack client to export FITS in parallel, default is False
        :type parallel: bool, optional

        .. _Description:
        Export the products from the panel mds onto FITS files to be read by other software packages

        **Additional Information**

        The FITS fils produced by this method have been tested and are known to work with CARTA and DS9
        """

        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_export_to_fits_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def export_gain_tables(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        wavelengths: float | int | list[float | int] | np.ndarray | None = None,
        wavelength_unit: str = "cm",
        frequencies: float | int | list[float | int] | np.ndarray | None = None,
        frequency_unit: str = "GHz",
        rms_unit: str = "mm",
        parallel: bool = False,
    ) -> None:
        """ Compute estimated antenna gains in dB and saves them to ASCII files.

        :param destination: Name of the destination folder to contain ASCII files
        :type destination: str

        :param ant: List of antennas/antenna to be exported, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be exported, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param wavelengths: List of wavelengths at which to compute the gains.
        :type wavelengths: list or float, optional

        :param wavelength_unit: Unit for the wavelengths being used, default is cm.
        :type wavelength_unit: str, optional

        :param frequencies: List of frequencies at which to compute the gains.
        :type frequencies: list or float, optional

        :param frequency_unit: Unit for the frequencies being used, default is GHz.
        :type frequency_unit: str, optional

        :param rms_unit: Unit for the Antenna surface RMS, default is mm.
        :type rms_unit: str, optional

        :param parallel: If True will use an existing astrohack client to produce ASCII files in parallel, default is False
        :type parallel: bool, optional

        .. _Description:

        Export antenna gains in dB from ``astrohack.panel`` for analysis.

        **Additional Information**

        .. rubric:: Selecting frequencies and wavelengths:

        If neither a frequency list nor a wavelength list is provided, ``export_gains_table`` will try to use a\
        predefined list set for the telescope associated with the dataset. If both are provided, ``export_gains_table``\
        will combine both lists.
        """

        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_export_gain_tables_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
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
        """ Create a Summary of observation information

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

        **Additional Information**

        This method produces a summary of the data in the AstrohackPanelFile displaying general information,
        spectral information, beam image characteristics and aperture image characteristics.
        """

        generate_observation_summary(
            mds_object=self,
            param_dict=locals(),
            key_order=["ant", "ddi"],
            summary_type="panel",
        )


def _export_screws_chunk(parm_dict):
    """
    Chunk function for the user facing function export_screws
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    export_name = parm_dict["destination"] + f"/panel_screws_{antenna}_{ddi}."
    xds = parm_dict["xdt_data"]
    surface = AntennaSurface(xds, reread=True)
    surface.export_screws(export_name + "txt", unit=parm_dict["unit"])
    surface.plot_screw_adjustments(export_name + "png", parm_dict)


def _plot_antenna_chunk(parm_dict):
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
    xds = parm_dict["xdt_data"]
    surface = AntennaSurface(xds, reread=True)
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


def _export_to_fits_chunk(parm_dict):
    """
    Panel side chunk function for the user facing function export_to_fits
    Args:
        parm_dict: parameter dictionary
    """

    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    destination = parm_dict["destination"]
    logger.info(
        f"Exporting panel contents of {antenna} {ddi} to FITS files in {destination}"
    )
    xds = parm_dict["xdt_data"]
    surface = AntennaSurface(xds, reread=True)
    basename = f"{destination}/{antenna}_{ddi}"
    surface.export_to_fits(basename)
    return


def _export_gain_tables_chunk(parm_dict):
    in_waves = parm_dict["wavelengths"]
    in_freqs = parm_dict["frequencies"]
    ant = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    xds = parm_dict["xdt_data"]
    antenna = AntennaSurface(xds, reread=True)
    frequency = clight / antenna.wavelength

    if in_waves is None and in_freqs is None:
        try:
            wavelengths = antenna.telescope.gain_wavelengths
        except AttributeError:
            msg = f"Telescope {antenna.telescope.name} has no predefined list of wavelengths to compute gains"
            logger.error(msg)
            logger.info("Please provide one in the arguments")
            raise NotImplementedError(msg)
    else:
        wave_fac = convert_unit(parm_dict["wavelength_unit"], "m", "length")
        freq_fac = convert_unit(parm_dict["frequency_unit"], "Hz", "frequency")
        wavelengths = []
        if in_waves is not None:
            if isinstance(in_waves, float) or isinstance(in_waves, int):
                in_waves = [in_waves]
            for in_wave in in_waves:
                wavelengths.append(wave_fac * in_wave)
        if in_freqs is not None:
            if isinstance(in_freqs, float) or isinstance(in_freqs, int):
                in_freqs = [in_freqs]
            for in_freq in in_freqs:
                wavelengths.append(clight / freq_fac / in_freq)

    db = "dB"
    rmsunit = parm_dict["rms_unit"]
    rmses = antenna.get_rms(rmsunit)

    field_names = [
        "Frequency",
        "Wavelength",
        "Before panel",
        "After panel",
        "Theoretical Max.",
    ]
    table = create_pretty_table(field_names)

    date_str = create_informative_label_from_summary(xds.attrs["summary"], "deg").split(
        ","
    )[-1]
    outstr = f"# Gain estimates for {antenna.telescope.name} antenna {ant.split(undscr)[1]}{lnbr}"
    outstr += f"# Based on a measurement at {format_frequency(frequency)}, {format_wavelength(antenna.wavelength)}{lnbr}"
    outstr += f"# Observation date:{date_str}{lnbr}"
    outstr += f"# Antenna surface RMS before adjustment: {format_value_unit(rmses[0], rmsunit)}{lnbr}"
    outstr += f"# Antenna surface RMS after adjustment: {format_value_unit(rmses[1], rmsunit)}{lnbr}"
    outstr += lnbr

    for wavelength in wavelengths:
        prior, theo = antenna.gain_at_wavelength(False, wavelength)
        after, _ = antenna.gain_at_wavelength(True, wavelength)
        row = [
            format_frequency(clight / wavelength),
            format_wavelength(wavelength),
            format_value_unit(prior, db),
            format_value_unit(after, db),
            format_value_unit(theo, db),
        ]
        table.add_row(row)

    outstr += table.get_string()
    string_to_ascii_file(
        outstr, parm_dict["destination"] + f"/panel_gains_{ant}_{ddi}.txt"
    )
