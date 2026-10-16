import numpy as np
import toolviper.utils.logger as logger
import toolviper.utils.parameter

from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.io.base_mds import AstrohackBaseFile
from astrohack.utils.constants import clight, length_units, trigo_units
from astrohack.utils.conversion import convert_5d_grid_from_stokes, convert_unit
from astrohack.utils.fits import (
    add_prefix,
    put_axis_in_fits_header,
    put_resolution_in_fits_header,
    put_stokes_axis_in_fits_header,
    write_fits,
)
from astrohack.utils.graph import create_and_execute_graphs_for_outputs
from astrohack.utils.phase_fitting import aips_par_names
from astrohack.utils.text import (
    create_dataset_label,
    create_informative_label_from_summary,
    create_pretty_table,
    format_label,
    format_value_error,
    lnbr,
    spc,
    string_to_ascii_file,
    undscr,
)
from astrohack.utils.validation import (
    custom_plots_checker,
    custom_split_checker,
    custom_unit_checker,
)
from astrohack.visualization.observation_summary import (
    generate_observation_summary,
)
from astrohack.visualization.plot_tools import (
    close_figure,
    create_figure_and_axes,
    simple_imshow_map_plot,
)


class AstrohackImageFile(AstrohackBaseFile):
    """Data class for image data.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackImageFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackImageFile object
        :rtype: AstrohackImageFile
        """
        super().__init__(file=file)

    @toolviper.utils.parameter.validate(custom_checker=custom_split_checker)
    def export_to_fits(
        self,
        destination: str,
        complex_split: str = "cartesian",
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        parallel: bool = False,
    ) -> None:
        """Export contents of an AstrohackImageFile object to several FITS files in the destination folder

        :param destination: Name of the destination folder to contain plots
        :type destination: str

        :param complex_split: How to split complex data, cartesian (real + imag, default) or polar (amplitude + phase)
        :type complex_split: str, optional

        :param ant: List of antennas/antenna to be plotted, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param parallel: If True will use an existing astrohack client to export FITS in parallel, default is False
        :type parallel: bool, optional

        .. _Description:
        Export the products from the holog mds onto FITS files to be read by other software packages

        **Additional Information**
        The image products of holog are complex images due to the nature of interferometric measurements and Fourier
        transforms, currently complex128 FITS files are not supported by astropy, hence the need to split complex images
        onto two real image products, we present the user with two options to carry out this split.

        .. rubric:: Available complex splitting possibilities:
        - *cartesian*: Split is done to a real part and an imaginary part FITS files
        - *polar*:     Split is done to an amplitude and a phase FITS files


        The FITS files produced by this function have been tested and are known to work with CARTA and DS9
        """

        param_dict = locals()

        param_dict["input_params"] = self.root.attrs["input_parameters"]
        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_export_to_fits_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )

    @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    def plot_apertures(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        polarization_state: str | list[str] = "I",
        plot_screws: bool = False,
        amplitude_limits: list[float | int] | tuple | np.ndarray | None = None,
        phase_unit: str = "deg",
        phase_limits: list[float | int] | tuple | np.ndarray | None = None,
        deviation_unit: str = "mm",
        deviation_limits: list[float | int] | tuple | np.ndarray | None = None,
        panel_labels: bool = False,
        display: bool = False,
        colormap: str = "viridis",
        figure_size: tuple | list[float | int] | np.ndarray | None = None,
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """ Aperture amplitude and phase plots from the data in an AstrohackImageFIle object.

        :param destination: Name of the destination folder to contain plots
        :type destination: str

        :param ant: List of antennas/antenna to be plotted, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param polarization_state: List of polarization states/ polarization state to be plotted, defaults to "I"
        :type polarization_state: list or str, optional

        :param plot_screws: Add screw positions to plot, default is False
        :type plot_screws: bool, optional

        :param amplitude_limits: Lower than Upper limit for amplitude in volts default is None (Guess from data)
        :type amplitude_limits: numpy.ndarray, list, tuple, optional

        :param phase_unit: Unit for phase plots, defaults is 'deg'
        :type phase_unit: str, optional

        :param phase_limits: Lower than Upper limit for phase, value in phase_unit, default is None (Guess from data)
        :type phase_limits: numpy.ndarray, list, tuple, optional

        :param deviation_unit: Unit for deviation plots, defaults is 'mm'
        :type deviation_unit: str, optional

        :param deviation_limits: Lower than Upper limit for deviation, value in deviation_unit, default is None (Guess\
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

        Produce plots from ``astrohack.holog`` results for analysis
        """
        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_aperture_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )

    @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    def plot_beams(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        complex_split: str = "polar",
        angle_unit: str = "deg",
        phase_unit: str = "deg",
        display: bool = False,
        colormap: str = "viridis",
        figure_size: tuple | list[float] | np.ndarray = (8, 3.5),
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """ Beam plots from the data in an AstrohackImageFIle object.

        :param destination: Name of the destination folder to contain plots
        :type destination: str

        :param ant: List of antennas/antenna to be plotted, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param angle_unit: Unit for L and M axes in plots, default is 'deg'.
        :type angle_unit: str, optional

        :param complex_split: How to split complex beam data, cartesian (real + imag) or polar (amplitude + phase, \
        default)
        :type complex_split: str, optional

        :param phase_unit: Unit for phase in 'polar' plots, default is 'deg'.
        :type phase_unit: str

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

        Produce plots from ``astrohack.holog`` results for analysis
        """
        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_beam_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )

    @toolviper.utils.parameter.validate(custom_checker=custom_unit_checker)
    def export_phase_fit_results(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        angle_unit: str = "deg",
        length_unit: str = "mm",
        parallel: bool = False,
    ) -> None:
        """Export perturbations phase fit results from the data in an AstrohackImageFIle object to ASCII files.

        :param destination: Name of the destination folder to contain ASCII files
        :type destination: str

        :param ant: List of antennas/antenna to be exported, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be exported, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param angle_unit: Unit for results that are angles.
        :type angle_unit: str, optional

        :param length_unit: Unit for results that are displacements.
        :type length_unit: str, optional

        :param parallel: If True will use an existing astrohack client to produce ASCII files in parallel, default is False
        :type parallel: bool, optional

        .. _Description:

        Export the results of the phase fitting process in ``astrohack.holog`` for analysis
        """
        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_export_phase_fit_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )

    @toolviper.utils.parameter.validate()
    def export_zernike_fit_results(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        parallel: bool = False,
    ) -> None:
        """Export Zernike coefficients from the data in an AstrohackImageFIle object to ASCII files.

        :param destination: Name of the destination folder to contain ASCII files
        :type destination: str

        :param ant: List of antennas/antenna to be exported, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be exported, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

        :param parallel: If True will use an existing astrohack client to produce ASCII files in parallel, default is False
        :type parallel: bool, optional

        .. _Description:

        Export Zernike coefficients from the AstrohackImageFile object obtained during processing in \
        ``astrohack.holog`` for analysis.
        """
        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_export_zernike_fit_chunk,
            param_dict=param_dict,
            key_order=["ant", "ddi"],
        )

    @toolviper.utils.parameter.validate(custom_checker=custom_plots_checker)
    def plot_zernike_model(
        self,
        destination: str,
        ant: str | list[str] = "all",
        ddi: str | int | list[int] = "all",
        display: bool = False,
        colormap: str = "viridis",
        figure_size: tuple | list[float] | np.ndarray = (16, 9),
        dpi: int = 300,
        parallel: bool = False,
    ) -> None:
        """Plot Zernike models from the data in an AstrohackImageFile object.

        :param destination: Name of the destination folder to contain the model plots
        :type destination: str

        :param ant: List of antennas/antenna to be exported, defaults to "all" when None, ex. ea25
        :type ant: list or str, optional

        :param ddi: List of ddis/ddi to be exported, defaults to "all" when None, ex. 0
        :type ddi: list or int, optional

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

        Export Zernike coefficients from the AstrohackImageFile object obtained during processing in \
        ``astrohack.holog`` for analysis.
        """
        param_dict = locals()

        create_and_execute_graphs_for_outputs(
            mds_object=self,
            chunk_function=_plot_zernike_model_chunk,
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

        This method produces a summary of the data in the AstrohackImageFile displaying general information,
        spectral information, beam image characteristics and aperture image characteristics.
        """
        generate_observation_summary(
            mds_object=self,
            param_dict=locals(),
            summary_type="image",
            key_order=["ant", "ddi"],
        )


def _export_to_fits_chunk(param_dict):
    """
    Holog side chunk function for the user facing function export_to_fits
    Args:
        param_dict: parameter dictionary
    """
    input_xds = param_dict["xdt_data"]
    input_params = param_dict["input_params"]
    antenna = param_dict["this_ant"]
    ddi = param_dict["this_ddi"]
    destination = param_dict["destination"]
    basename = f"{destination}/{antenna}_{ddi}"

    logger.info(
        f"Exporting image contents of {antenna} {ddi} to FITS files in {destination}"
    )

    aperture_resolution = input_xds.attrs["summary"]["aperture"]["cell size"]

    nchan = len(input_xds.chan)

    if nchan == 1:
        reffreq = input_xds.chan.values[0]

    else:
        reffreq = input_xds.chan.values[nchan // 2]

    telname = input_xds.attrs["summary"]["general"]["telescope name"]

    if telname in ["EVLA", "VLA", "JVLA"]:
        telname = "VLA"

    polist = []

    for pol in input_xds.pol:
        polist.append(str(pol.values))

    base_header = {
        "STOKES": ", ".join(polist),
        "WAVELENG": clight / reffreq,
        "FREQUENC": reffreq,
        "TELESCOP": input_xds.attrs["summary"]["general"]["antenna name"],
        "INSTRUME": telname,
        "TIME_CEN": input_xds.attrs["time_centroid"],
        "PADDING": input_params["padding_factor"],
        "GRD_INTR": input_params["grid_interpolation_mode"],
        "CHAN_AVE": "yes" if input_params["chan_average"] is True else "no",
        "CHAN_TOL": input_params["chan_tolerance_factor"],
        "SCAN_AVE": "yes" if input_params["scan_average"] is True else "no",
        "TO_STOKE": "yes" if input_params["to_stokes"] is True else "no",
    }

    ntime = len(input_xds.time)
    if ntime != 1:
        raise RuntimeError("Data with multiple times not supported for FITS export")

    base_header = put_axis_in_fits_header(
        base_header, input_xds.chan.values, 3, "Frequency", "Hz"
    )
    base_header = put_stokes_axis_in_fits_header(base_header, 4)
    rad_to_deg = convert_unit("rad", "deg", "trigonometric")
    beam_header = put_axis_in_fits_header(
        base_header, -input_xds.l.values * rad_to_deg, 1, "RA---SIN", "deg"
    )
    beam_header = put_axis_in_fits_header(
        beam_header, input_xds.m.values * rad_to_deg, 2, "DEC--SIN", "deg"
    )
    beam_header["RADESYSA"] = "FK5"
    beam = input_xds["BEAM"].values
    if param_dict["complex_split"] == "cartesian":
        write_fits(
            beam_header,
            "Complex beam real part",
            beam.real,
            add_prefix(basename, "beam_real") + ".fits",
            "Normalized",
            "image",
        )
        write_fits(
            beam_header,
            "Complex beam imag part",
            beam.imag,
            add_prefix(basename, "beam_imag") + ".fits",
            "Normalized",
            "image",
        )
    else:
        write_fits(
            beam_header,
            "Complex beam amplitude",
            np.absolute(beam),
            add_prefix(basename, "beam_amplitude") + ".fits",
            "Normalized",
            "image",
        )
        write_fits(
            beam_header,
            "Complex beam phase",
            np.angle(beam),
            add_prefix(basename, "beam_phase") + ".fits",
            "Radians",
            "image",
        )
    wavelength = clight / input_xds.chan.values[0]
    aperture_header = put_axis_in_fits_header(
        base_header, input_xds.u.values * wavelength, 1, "X----LIN", "m"
    )
    aperture_header = put_axis_in_fits_header(
        aperture_header, input_xds.u.values * wavelength, 2, "Y----LIN", "m"
    )
    aperture_header = put_resolution_in_fits_header(
        aperture_header, aperture_resolution
    )
    aperture = input_xds["APERTURE"].values
    if param_dict["complex_split"] == "cartesian":
        write_fits(
            aperture_header,
            "Complex aperture real part",
            aperture.real,
            add_prefix(basename, "aperture_real") + ".fits",
            "Normalized",
            "image",
        )
        write_fits(
            aperture_header,
            "Complex aperture imag part",
            aperture.imag,
            add_prefix(basename, "aperture_imag") + ".fits",
            "Normalized",
            "image",
        )
    else:
        write_fits(
            aperture_header,
            "Complex aperture amplitude",
            np.absolute(aperture),
            add_prefix(basename, "aperture_amplitude") + ".fits",
            "Normalized",
            "image",
        )
        write_fits(
            aperture_header,
            "Complex aperture phase",
            np.angle(aperture),
            add_prefix(basename, "aperture_phase") + ".fits",
            "rad",
            "image",
        )

    phase_amp_header = put_axis_in_fits_header(
        base_header, input_xds.u_prime.values * wavelength, 1, "X----LIN", "m"
    )
    phase_amp_header = put_axis_in_fits_header(
        phase_amp_header, input_xds.v_prime.values * wavelength, 2, "Y----LIN", "m"
    )
    phase_amp_header = put_resolution_in_fits_header(
        phase_amp_header, aperture_resolution
    )
    write_fits(
        phase_amp_header,
        "Cropped aperture corrected phase",
        input_xds["CORRECTED_PHASE"].values,
        add_prefix(basename, "corrected_phase") + ".fits",
        "rad",
        "image",
    )
    return


def _plot_aperture_chunk(parm_dict):
    """
    Chunk function for the user facing function plot_apertures
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    destination = parm_dict["destination"]
    input_xds = parm_dict["xdt_data"]
    input_xds.attrs["AIPS"] = False

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
        raise ValueError(msg)

    for pol_state in plot_pol_states:
        if pol_state in avail_pol_states:
            surface = AntennaSurface(
                input_xds.dataset,
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


def _plot_beam_chunk(parm_dict):
    """
    Chunk function for the user facing function plot_beams
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    destination = parm_dict["destination"]
    input_xds = parm_dict["xdt_data"]
    laxis = input_xds.l.values * convert_unit(
        "rad", parm_dict["angle_unit"], "trigonometric"
    )
    maxis = input_xds.m.values * convert_unit(
        "rad", parm_dict["angle_unit"], "trigonometric"
    )
    if input_xds.sizes["chan"] != 1:
        raise RuntimeError("Only single channel holographies supported")

    if input_xds.sizes["time"] != 1:
        raise RuntimeError("Only single mapping holographies supported")

    full_beam = input_xds.BEAM.isel(time=0, chan=0).values
    pol_axis = input_xds.pol.values

    for i_pol, pol in enumerate(pol_axis):
        basename = f"{destination}/{antenna}_{ddi}_pol_{pol}"
        _plot_beam_by_pol(laxis, maxis, pol, full_beam[i_pol, ...], basename, parm_dict)


def _plot_beam_by_pol(laxis, maxis, pol, beam_image, basename, parm_dict):
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
    norm_z_label = "Z Scale [Normalized]"
    x_label = f"L axis [{parm_dict['angle_unit']}]"
    y_label = f"M axis [{parm_dict['angle_unit']}]"

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
    suptitle = "Beam for: " + create_informative_label_from_summary(
        parm_dict["xdt_data"].attrs["summary"], "deg"
    )
    suptitle += f", Pol = {pol}"
    close_figure(fig, suptitle, plot_name, parm_dict["dpi"], parm_dict["display"])
    return


def _export_phase_fit_chunk(parm_dict):
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    destination = parm_dict["destination"]
    phase_fit_results = parm_dict["xdt_data"].attrs["phase_fitting"]
    if phase_fit_results is None:
        logger.warning(
            f"No phase fit results to export for {create_dataset_label(antenna, ddi)}"
        )
        return

    angle_unit = parm_dict["angle_unit"]
    length_unit = parm_dict["length_unit"]
    field_names = ["Parameter", "Value", "Unit"]
    alignment = ["l", "r", "c"]
    outstr = ""
    summary = parm_dict["xdt_data"].attrs["summary"]
    azel_unit = "deg"
    for mapkey, map_dict in phase_fit_results.items():
        for freq, freq_dict in map_dict.items():
            for pol, pol_dict in freq_dict.items():
                outstr += (
                    f"* {mapkey.replace(undscr, spc)}, "
                    f"{create_informative_label_from_summary(summary, azel_unit)}, "
                    f"polarization state {pol}:{2 * lnbr} "
                )
                table = create_pretty_table(field_names, alignment)
                for par_name in aips_par_names:
                    item = pol_dict[par_name]
                    val = item["value"]
                    err = item["error"]
                    unit = item["unit"]
                    if unit in length_units:
                        fac = convert_unit(unit, length_unit, "length")
                    elif unit in trigo_units:
                        fac = convert_unit(unit, angle_unit, "trigonometric")
                    else:
                        msg = f"Unknown unit {unit}"
                        logger.error(msg)
                        raise ValueError(msg)

                    row = [
                        format_label(par_name),
                        format_value_error(fac * val, fac * err, 1.0, 1e-4),
                        unit,
                    ]
                    table.add_row(row)

                outstr += table.get_string() + 2 * lnbr

    string_to_ascii_file(outstr, f"{destination}/image_phase_fit_{antenna}_{ddi}.txt")


def _export_zernike_fit_chunk(parm_dict):
    antenna = parm_dict["this_ant"]
    ddi = parm_dict["this_ddi"]
    xdt_data = parm_dict["xdt_data"]
    zernike_coeffs = xdt_data["ZERNIKE_COEFFICIENTS"].values
    rms = xdt_data["ZERNIKE_FIT_RMS"].values
    corr_axis = xdt_data.orig_pol.values
    freq_axis = xdt_data.chan.values
    ntime = zernike_coeffs.shape[0]
    osa_indices = xdt_data.osa.values
    destination = parm_dict["destination"]

    field_names = ["Indices", "Real", "Imaginary"]
    alignment = ["l", "c", "c"]
    outstr = ""

    for itime in range(ntime):
        for ichan, freq in enumerate(freq_axis):
            for icorr, corr in enumerate(corr_axis):
                outstr += f"* map {itime}, "
                outstr += create_informative_label_from_summary(
                    xdt_data.attrs["summary"], "deg"
                )
                outstr += f", Correlation {corr}:{lnbr}"
                outstr += (
                    f"   Fit RMS = {rms[itime, ichan, icorr].real:.8f} + {rms[itime, ichan, icorr].imag:.8f}*i"
                    + 2 * lnbr
                )
                table = create_pretty_table(field_names, alignment)
                for icoeff, coeff in enumerate(zernike_coeffs[itime, ichan, icorr]):
                    row = [
                        osa_indices[icoeff],
                        f"{coeff.real:.8f}",
                        f"{coeff.imag:.8f}",
                    ]
                    table.add_row(row)

                outstr += table.get_string() + 2 * lnbr

    string_to_ascii_file(outstr, f"{destination}/image_zernike_fit_{antenna}_{ddi}.txt")


def _plot_zernike_model_chunk(parm_dict):
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
    input_xds = parm_dict["xdt_data"]

    if input_xds.sizes["chan"] != 1:
        raise RuntimeError("Only single channel holographies supported")

    if input_xds.sizes["time"] != 1:
        raise RuntimeError("Only single mapping holographies supported")

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
    suptitle = f"Zernike model (N<={zernike_n_order}) for: "
    suptitle += create_informative_label_from_summary(input_xds.attrs["summary"], "deg")
    suptitle += ", correlation: "

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


def _plot_zernike_cartesian_component(
    ax, fig, aperture, model, u_axis, v_axis, colormap, comp_label
):
    maxabs = float(np.nanmax(np.abs(aperture)))
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
    _plot_zernike_cartesian_component(
        ax[0],
        fig,
        aperture.real,
        model_aperture.real,
        u_axis,
        v_axis,
        parm_dict["colormap"],
        "real",
    )
    _plot_zernike_cartesian_component(
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
