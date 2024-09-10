import pathlib
import toolviper.utils.parameter

import numpy as np
import toolviper.utils.logger as logger

from toolviper.utils.console import Colorize

from astrohack.utils import create_pretty_table
from astrohack.utils.validation import custom_plots_checker
from astrohack.utils.validation import custom_unit_checker
from astrohack.utils.validation import custom_split_checker

from astrohack.utils.graph import compute_graph

from astrohack.visualization.diagnostics import calibration_plot_chunk
from astrohack.visualization.diagnostics import plot_lm_coverage
from astrohack.visualization.diagnostics import plot_sky_coverage_chunk
from astrohack.visualization.diagnostics import plot_delays_chunk
from astrohack.visualization.diagnostics import plot_position_corrections
from astrohack.visualization.diagnostics import plot_antenna_chunk
from astrohack.visualization.diagnostics import plot_aperture_chunk
from astrohack.visualization.diagnostics import plot_beam_chunk

from astrohack.utils.file import load_panel_file
from astrohack.utils.file import load_image_file
from astrohack.utils.file import load_holog_file
from astrohack.utils.file import load_point_file
from astrohack.utils.file import load_locit_file
from astrohack.utils.file import load_position_file

from astrohack.utils.data import read_meta_data
from astrohack.utils.data import export_to_aips
from astrohack.visualization.textual_data import export_locit_fit_results, export_screws_chunk, \
    export_gains_table_chunk, export_phase_fit_chunk, print_array_configuration
from astrohack.visualization.fits import export_to_fits_panel_chunk, export_to_fits_holog_chunk

from astrohack.core.extract_locit import plot_source_table
from astrohack.core.extract_locit import plot_array_configuration

from astrohack.antenna.antenna_surface import AntennaSurface

from astrohack.antenna.telescope import Telescope

from astrohack.utils.text import print_method_list
from astrohack.utils.text import print_dict_table
from astrohack.utils.text import print_data_contents
from astrohack.utils.text import print_summary_header
from astrohack.utils.text import rad_to_deg_str
from astrohack.utils.text import rad_to_hour_str

from prettytable import PrettyTable

from typing import Any, List, Union, Tuple

colorize = Colorize()


class AstrohackDataFile:
    """ Base class for the Astrohack data files
    """

    def __init__(self, file_stem: str, path: str = './'):

        self._image_path = None
        self._holog_path = None
        self._panel_path = None
        self._point_path = None

        self.holog = None
        self.image = None
        self.panel = None
        self.point = None

        self._verify_holog_files(file_stem, path)

    def _verify_holog_files(self, file_stem: str, path: str):

        logger.info("Verifying {stem}.* files in path={path} ...".format(stem=file_stem, path=path))

        file_path = "{path}/{stem}.holog.zarr".format(path=path, stem=file_stem)

        if pathlib.Path(file_path).is_dir():
            logger.info("Found {stem}.holog.zarr directory ...".format(stem=file_stem))

            self._holog_path = file_path
            self.holog = AstrohackHologFile(file_path)

        file_path = "{path}/{stem}.image.zarr".format(path=path, stem=file_stem)

        if pathlib.Path(file_path).is_dir():
            logger.info("Found {stem}.image.zarr directory ...".format(stem=file_stem))

            self._image_path = file_path
            self.image = AstrohackImageFile(file_path)

        file_path = "{path}/{stem}.panel.zarr".format(path=path, stem=file_stem)

        if pathlib.Path(file_path).is_dir():
            logger.info("Found {stem}.panel.zarr directory ...".format(stem=file_stem))

            self._image_path = file_path
            self.panel = AstrohackPanelFile(file_path)

        file_path = "{path}/{stem}.point.zarr".format(path=path, stem=file_stem)

        if pathlib.Path(file_path).is_dir():
            logger.info("Found {stem}.point.zarr directory ...".format(stem=file_stem))

            self._point_path = file_path
            self.point = AstrohackPointFile(file_path)


class AstrohackImageFile(dict):
    """ Data class for holography image data.

    Data within an object of this class can be selected for further inspection, plotted or output to FITS files.
    """

    def __init__(self, file: str):
        """ Initialize an AstrohackImageFile object.
        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackImageFile object
        :rtype: AstrohackImageFile
        """
        super().__init__()
        self._meta_data = None
        self._input_pars = None
        self.file = file
        self._file_is_open = False

    def __getitem__(self, key: str):
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(key, value)

    @property
    def is_open(self) -> bool:
        """ Check whether the object has opened the corresponding hack file.

        :return: True if open, else False.
        :rtype: bool
        """
        return self._file_is_open

    def open(self, file: str = None) -> bool:
        """ Open holography image file.
        
        :param file: File to be opened, if None defaults to the previously defined file
        :type file: str, optional

        :return: True if file is properly opened, else returns False
        :rtype: bool
        """

        if file is None:
            file = self.file

        try:
            load_image_file(file, image_dict=self)
            self._meta_data = read_meta_data(file + '/.image_attr')
            self._input_pars = read_meta_data(file + '/.image_input')
            self._file_is_open = True

        except Exception as error:
            logger.error(f"There was an exception opening the file: {error}")
            self._file_is_open = False

        return self._file_is_open

    def summary(self):
        """ Prints summary of the AstrohackImageFile object, with available data, attributes and available methods
        """
        print_summary_header(self.file)
        print_dict_table(self._input_pars)
        print_data_contents(self, ["Antenna", "DDI"])
        print_method_list([self.summary, self.select, self.export_to_fits, self.plot_beams, self.plot_apertures,
                           self.export_phase_fit_results])

    @toolviper.utils.parameter.validate(
        custom_checker=custom_split_checker
    )
    def select(
            self,
            ant: str,
            ddi: int,
            complex_split: str = 'cartesian'
    ) -> object:
        """ Select data on the basis of ddi, scan, ant. This is a convenience function.

        :param ddi: Data description ID, ex. 0.
        :type ddi: int
        :param ant: Antenna ID, ex. ea25.
        :type ant: str
        :param complex_split: Is the data to b left as is (Real + imag: cartesian, default) or split into Amplitude \
        and Phase (polar)
        :type complex_split: str, optional

        :return: Corresponding xarray dataset, or self if selection is None
        :rtype: xarray.Dataset or AstrohackImageFile
        """

        ant = 'ant_' + ant
        ddi = f'ddi_{ddi}'

        if ant is None or ddi is None:
            logger.info("No selections made ...")
            return self
        else:
            if complex_split == 'polar':
                return self[ant][ddi].apply(np.absolute), self[ant][ddi].apply(np.angle, deg=True)
            else:
                return self[ant][ddi]

    @toolviper.utils.parameter.validate(
        custom_checker=custom_split_checker
    )
    def export_to_fits(
            self,
            destination: str,
            complex_split: str = 'cartesian',
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            parallel: bool = False
    ) -> None:
        """ Export contents of an AstrohackImageFile object to several FITS files in the destination folder

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
        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        param_dict['metadata'] = self._meta_data
        compute_graph(
            self,
            export_to_fits_holog_chunk,
            param_dict,
            ['ant', 'ddi'],
            parallel=parallel
        )

    @toolviper.utils.parameter.validate(
        custom_checker=custom_plots_checker
    )
    def plot_apertures(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            polarization_state: Union[str, List[str]] = "I",
            plot_screws: bool = False,
            amplitude_limits: Union[List[float], Tuple, np.array] = None,
            phase_unit: str = 'deg',
            phase_limits: Union[List[float], Tuple, np.array] = None,
            deviation_unit: str = 'mm',
            deviation_limits: Union[List[float], Tuple, np.array] = None,
            panel_labels: bool = False,
            display: bool = False,
            colormap: str = 'viridis',
            figure_size: Union[Tuple, List[float], np.array] = None,
            dpi: int = 300,
            parallel: bool = False
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

        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        compute_graph(self, plot_aperture_chunk, param_dict, ['ant', 'ddi'], parallel=parallel)

    @toolviper.utils.parameter.validate(
        custom_checker=custom_plots_checker
    )
    def plot_beams(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            complex_split: str = 'polar',
            angle_unit: str = 'deg',
            phase_unit: str = 'deg',
            display: bool = False,
            colormap: str = 'viridis',
            figure_size: Union[Tuple, List[float], np.array] = (8, 4.5),
            dpi: int = 300,
            parallel: bool = False
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

        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        compute_graph(self, plot_beam_chunk, param_dict, ['ant', 'ddi'], parallel=parallel)

    @toolviper.utils.parameter.validate(
        custom_checker=custom_unit_checker
    )
    def export_phase_fit_results(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            angle_unit: str = 'deg',
            length_unit: str = 'mm',
            parallel: bool = False
    ) -> None:
        """ Export phase fit resutls from the data in an AstrohackImageFIle object to ASCII files.

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

        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        compute_graph(self, export_phase_fit_chunk, param_dict, ['ant', 'ddi'], parallel=parallel)


class AstrohackHologFile(dict):
    """ Data Class for extracted holography data

    Data within an object of this class can be selected for further inspection or plotted for calibration diagnostics.
    """

    def __init__(self, file: str):
        """ Initialize an AstrohackHologFile object.
        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackHologFile object
        :rtype: AstrohackHologFile
        """
        super().__init__()

        self.file = file
        self._meta_data = None
        self._input_pars = None
        self._file_is_open = False

    def __getitem__(self, key: str):
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(key, value)

    @property
    def is_open(self) -> bool:
        """ Check whether the object has opened the corresponding hack file.

        :return: True if open, else False.
        :rtype: bool
        """
        return self._file_is_open

    def open(self, file: str = None, dask_load: bool = True) -> bool:
        """ Open extracted holography file.
        :param file: File to be opened, if None defaults to the previously defined file
        :type file: str, optional
        :param dask_load: Is file to be loaded with dask?, default is True
        :type dask_load: bool, optional

        :return: True if file is properly opened, else returns False
        :rtype: bool
        """

        if file is None:
            file = self.file

        try:
            load_holog_file(file=file, dask_load=dask_load, load_pnt_dict=False, holog_dict=self)
            self._meta_data = read_meta_data(file + '/.holog_attr')
            self._input_pars = read_meta_data(file + '/.holog_input')
            self._file_is_open = True

        except Exception as error:
            logger.error(f"There was an exception opening the file: {error}")
            self._file_is_open = False

        return self._file_is_open

    def summary(self) -> None:
        """ Prints summary of the AstrohackHologFile object, with available data, attributes and available methods
        """
        print_summary_header(self.file)
        print_dict_table(self._input_pars)
        print_data_contents(self, ["DDI", "Map", "Antenna"])
        print_method_list([self.summary, self.select, self.plot_diagnostics, self.plot_lm_sky_coverage])

    @toolviper.utils.parameter.validate()
    def select(
            self,
            ddi: int,
            map_id: int,
            ant: str
    ) -> object:
        """ Select data on the basis of ddi, scan, ant. This is a convenience function.

        :param ddi: Data description ID, ex. 0.
        :type ddi: int
        :param map_id: Mapping ID, ex. 0.
        :type map_id: int
        :param ant: Antenna ID, ex. ea25.
        :type ant: str

        :return: Corresponding xarray dataset, or self if selection is None
        :rtype: xarray.Dataset or AstrohackHologFile
        """

        ant = 'ant_' + ant
        ddi = f'ddi_{ddi}'
        map_id = f'map_{map_id}'

        if ant is None or ddi is None or map_id is None:
            logger.info("No selection made ...")
            return self
        else:
            return self[ddi][map_id][ant]

    @property
    def meta_data(self):
        """ Retrieve AstrohackHologFile JSON metadata.

        :return: JSON metadata for this AstrohackHologFile object
        :rtype: dict
        """

        return self._meta_data

    @toolviper.utils.parameter.validate(
        custom_checker=custom_plots_checker
    )
    def plot_diagnostics(
            self,
            destination: str,
            delta: float = 0.01,
            ant: Union[str, List[str]] = "all",
            ddi: Union[str, List[str]] = "all",
            map_id: Union[int, List[int]] = "all",
            complex_split: str = 'polar',
            display: bool = False,
            figure_size: Union[Tuple, List[float], np.array] = None,
            dpi: int = 300,
            parallel: bool = False
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

        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        key_order = ["ddi", "map", "ant"]
        compute_graph(self, calibration_plot_chunk, param_dict, key_order, parallel)

    @toolviper.utils.parameter.validate(
        custom_checker=custom_plots_checker
    )
    def plot_lm_sky_coverage(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            map_id: Union[int, List[int]] = "all",
            angle_unit: str = 'deg',
            time_unit: str = 'hour',
            plot_correlation: Union[str, List[str]] = None,
            complex_split: str = 'polar',
            phase_unit: str = 'deg',
            display: bool = False,
            figure_size: Union[Tuple, List[float], np.array] = None,
            dpi: int = 300,
            parallel: bool = False
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

        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        key_order = ["ddi", "map", "ant"]
        compute_graph(self, plot_lm_coverage, param_dict, key_order, parallel)
        return

    @toolviper.utils.parameter.validate(
        custom_checker=custom_plots_checker
    )
    def export_to_aips(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            map_id: Union[int, List[int]] = "all",
            parallel: bool = False
    ) -> None:
        """ Export data compatible to AIPS's HOLOG task

        :param destination: Name of the destination folder to contain SCII files
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

        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        key_order = ["ddi", "map", "ant"]
        compute_graph(self, export_to_aips, param_dict, key_order, parallel)
        return


class AstrohackPanelFile(dict):
    """ Data class for holography panel data.

    Data within an object of this class can be selected for further inspection, plotted or exported to FITS for analysis
    or exported to csv for panel adjustments.
    """

    def __init__(self, file: str):
        """ Initialize an AstrohackPanelFile object.
        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPanelFile object
        :rtype: AstrohackPanelFile
        """
        super().__init__()

        self.file = file
        self._file_is_open = False
        self._input_pars = None

    def __getitem__(self, key: str):
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(key, value)

    @property
    def is_open(self) -> bool:
        """ Check whether the object has opened the corresponding hack file.

        :return: True if open, else False.
        :rtype: bool
        """
        return self._file_is_open

    def open(self, file: str = None) -> bool:
        """ Open panel holography file.
        :param file: File to be opened, if None defaults to the previously defined file
        :type file: str, optional

        :return: True if file is properly opened, else returns False
        :rtype: bool
        """

        if file is None:
            file = self.file

        try:
            load_panel_file(file, panel_dict=self)
            self._input_pars = read_meta_data(file + '/.panel_input')
            self._file_is_open = True

        except Exception as error:
            logger.error(f"There was an exception opening the file: {error}")
            self._file_is_open = False

        return self._file_is_open

    def summary(self) -> None:
        """ Prints summary of the AstrohackPanelFile object, with available data, attributes and available methods
        """
        print_summary_header(self.file)
        print_dict_table(self._input_pars)
        print_data_contents(self, ["Antenna", "DDI"])
        print_method_list([self.summary, self.get_antenna, self.export_screws, self.export_to_fits,
                           self.plot_antennas, self.export_gain_tables])

    @toolviper.utils.parameter.validate()
    def get_antenna(
            self,
            ant: str,
            ddi: int
    ) -> AntennaSurface:
        """ Retrieve an AntennaSurface object for interaction

        :param ant: Antenna to be retrieved, ex. ea25.
        :type ant: str
        :param ddi: DDI to be retrieved for ant_id, ex. 0
        :type ddi: int

        :return: AntennaSurface object describing for further interaction
        :rtype: AntennaSurface
        """
        ant = 'ant_' + ant
        ddi = f'ddi_{ddi}'
        xds = self[ant][ddi]
        telescope = Telescope(xds.attrs['telescope_name'])
        return AntennaSurface(xds, telescope, reread=True)

    @toolviper.utils.parameter.validate(
        custom_checker=custom_plots_checker
    )
    def export_screws(
            self,
            destination: str,
            ant: Union[str, List[str]] = None,
            ddi: Union[int, List[int]] = None,
            unit: str = 'mm',
            threshold: float = None,
            panel_labels: bool = True,
            display: bool = False,
            colormap: str = 'RdBu_r',
            figure_size: Union[Tuple, List[float], np.array] = None,
            dpi: int = 300
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

        .. _Description:

        Produce the screw adjustments from ``astrohack.panel`` results to be used at the antenna site to improve \
        the antenna surface

        """
        param_dict = locals()

        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        compute_graph(self, export_screws_chunk, param_dict, ['ant', 'ddi'], parallel=False)

    @toolviper.utils.parameter.validate(
        custom_checker=custom_plots_checker
    )
    def plot_antennas(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            plot_type: str = 'deviation',
            plot_screws: bool = False,
            amplitude_limits: Union[Tuple, List[float], np.array] = None,
            phase_unit: str = 'deg',
            phase_limits: Union[Tuple, List[float], np.array] = None,
            deviation_unit: str = 'mm',
            deviation_limits: Union[Tuple, List[float], np.array] = None,
            panel_labels: bool = False,
            display: bool = False,
            colormap: str = 'viridis',
            figure_size: Union[Tuple, List[float], np.array] = (8.0, 6.4),
            dpi: int = 300,
            parallel: bool = False
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
        - *deviation*: Surface deviation estimated from phase and wavelength, three plots are produced for each antenna
                       and ddi combination, surface before correction, the corrections applied and the corrected
                       surface, most length units available
        - *phase*: Phase deviations over the surface, three plots are produced for each antenna and ddi combination,
                   phase before correction, the corrections applied and the corrected phase, deg and rad available as
                   units
        - *ancillary*: Two ancillary plots with useful information: The mask used to select data to be fitted, the
                       amplitude data used to derive the mask, units are irrelevant for these plots
        - *all*: All the plots listed above. In this case the unit parameter is taken to mean the deviation unit, the
                 phase unit is set to degrees
        """

        param_dict = locals()

        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        compute_graph(self, plot_antenna_chunk, param_dict, ['ant', 'ddi'], parallel=parallel)

    @toolviper.utils.parameter.validate()
    def export_to_fits(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            parallel: bool = False
    ) -> None:
        """ Export contents of an Astrohack MDS file to several FITS files in the destination folder

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

        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        compute_graph(self, export_to_fits_panel_chunk, param_dict, ['ant', 'ddi'],
                      parallel=parallel)

    @toolviper.utils.parameter.validate(
        custom_checker=custom_unit_checker
    )
    def export_gain_tables(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            wavelengths: Union[float, List[float]] = None,
            wavelength_unit: str = 'cm',
            frequencies: Union[float, List[float]] = None,
            frequency_unit: str = 'GHz',
            parallel: bool = False
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
        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        compute_graph(self, export_gains_table_chunk, param_dict, ['ant', 'ddi'],
                      parallel=parallel)


class AstrohackPointFile(dict):
    """ Data Class for holography pointing data.
    """

    def __init__(self, file: str):
        """ Initialize an AstrohackPointFile object.
        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPointFile object
        :rtype: AstrohackPointFile
        """
        super().__init__()

        self.file = file
        self._input_pars = None
        self._file_is_open = False

    def __getitem__(self, key: str):
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(key, value)

    @property
    def is_open(self) -> bool:
        """ Check whether the object has opened the corresponding hack file.

        :return: True if open, else False.
        :rtype: bool
        """
        return self._file_is_open

    def open(self, file: str = None, dask_load: bool = True) -> bool:
        """ Open holography pointing file.
        :param file: File to be opened, if None defaults to the previously defined file
        :type file: str, optional
        :param dask_load: Is file to be loaded with dask?, default is True
        :type dask_load: bool, optional

        :return: True if file is properly opened, else returns False
        :rtype: bool
        """

        if file is None:
            file = self.file

        try:
            load_point_file(file=file, dask_load=dask_load, pnt_dict=self)
            self._file_is_open = True
            self._input_pars = read_meta_data(file + '/.point_input')

        except FileNotFoundError:
            logger.error("Requested file {} doesn't exist ...".format(colorize.blue(file)))
            self._file_is_open = False

        except Exception as error:
            logger.error(f"There was an exception opening the file: {error}")
            self._file_is_open = False

        return self._file_is_open

    def summary(self) -> None:
        """ Prints summary of the AstrohackPointFile object, with available data, attributes and available methods
        """
        print_summary_header(self.file)
        print_dict_table(self._input_pars)
        print_data_contents(self, ["Antenna"])
        print_method_list([self.summary])


class AstrohackLocitFile(dict):
    """ Data Class for extracted gains for antenna location determination
    """

    def __init__(self, file: str):
        """ Initialize an AstrohackLocitFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackLocitFile object
        :rtype: AstrohackLocitFile
        """
        super().__init__()

        self.file = file
        self._input_pars = None
        self._meta_data = None
        self._file_is_open = False

    def __getitem__(self, key: str):
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(key, value)

    @property
    def is_open(self) -> bool:
        """ Check whether the object has opened the corresponding hack file.

        :return: True if open, else False.
        :rtype: bool
        """
        return self._file_is_open

    def open(self, file: str = None, dask_load: bool = True) -> bool:
        """ Open antenna location file.
        :param file: File to be opened, if None defaults to the previously defined file
        :type file: str, optional

        :param dask_load: Is file to be loaded with dask?, default is True
        :type dask_load: bool, optional

        :return: True if file is properly opened, else returns False
        :rtype: bool
        """


        if file is None:
            file = self.file

        try:
            load_locit_file(file=file, dask_load=dask_load, locit_dict=self)
            self._input_pars = read_meta_data(file + '/.locit_input')
            self._meta_data = read_meta_data(file + '/.locit_attr')
            self._file_is_open = True

        except Exception as error:
            logger.error(f"There was an exception opening the file: {error}")
            self._file_is_open = False

        return self._file_is_open

    def print_source_table(self) -> None:
        """ Prints a table with the sources observed for antenna location determination
        """
        print("\nSources:")
        field_names = ['Id', 'Name', 'RA FK5', 'DEC FK5', 'RA precessed', 'DEC precessed']
        table = create_pretty_table(field_names, 'l')
        for source in self['observation_info']['src_dict'].values():
            table.add_row([source['id'], source['name'], rad_to_hour_str(source['fk5'][0]),
                           rad_to_deg_str(source['fk5'][1]), rad_to_hour_str(source['precessed'][0]),
                           rad_to_deg_str(source['precessed'][1])])
        print(table)

    @toolviper.utils.parameter.validate()
    def print_array_configuration(
            self,
            relative: bool = True
    ) -> None:
        """ Prints a table containing the array configuration

        :param relative: Print relative antenna coordinates or geocentric coordinates, default is True
        :type relative: bool, optional

        .. _Description:

        Print arrayx configuration in the dataset. Also marks the reference antenna and the antennas that are
        absent from the dataset. Coordinates of antenna stations can be relative to the array center or Geocentric
        (longitude, latitude and radius)

        """
        param_dict = locals()
        print_array_configuration(param_dict, self['antenna_info'], self['observation_info']['telescope_name'])

    @toolviper.utils.parameter.validate()
    def plot_source_positions(
            self,
            destination: str,
            labels: bool = False,
            precessed: bool = False,
            display: bool = False,
            figure_size: Union[Tuple, List[float], np.array] = None,
            dpi: int = 300
    ) -> None:
        """ Plot source positions in either FK5 or precessed right ascension and declination.

        :param destination: Name of the destination folder to contain plot
        :type destination: str

        :param labels: Add source labels to the plot, defaults to False
        :type labels: bool, optional

        :param precessed: Plot in precessed coordinates? defaults to False (FK5)
        :type precessed: bool, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        .. _Description:

        Plot the sources on the source list to a full 24 hours 180 degrees flat 2D representation of the full sky.
        If precessed is set to True the coordinates precessd to the midpoint of the observations is plotted, otherwise
        the FK5 coordinates are plotted.
        The source names can be plotted next to their positions if label is True, however plots may become too crowded
        if that is the case.

        """
        param_dict = locals()
        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)

        if precessed:
            filename = str(pathlib.Path(destination).joinpath('locit_source_table_precessed.png'))
            time_range = self['observation_info']['time_range']
            obs_midpoint = (time_range[1] + time_range[0]) / 2.

        else:
            filename = str(pathlib.Path(destination).joinpath('locit_source_table_fk5.png'))
            obs_midpoint = None

        plot_source_table(
            filename,
            self['observation_info']['src_dict'],
            precessed=precessed,
            obs_midpoint=obs_midpoint,
            display=display,
            figure_size=figure_size,
            dpi=dpi,
            label=labels
        )

        return

    @toolviper.utils.parameter.validate(
        custom_checker=custom_unit_checker
    )
    def plot_array_configuration(
            self,
            destination: str,
            stations: bool = True,
            zoff: bool = False,
            unit: str = 'm',
            box_size: Union[int, float] = 5000,
            display: bool = False,
            figure_size: Union[Tuple, List[float], np.array] = None,
            dpi: int = 300
    ) -> None:
        """ Plot antenna positions.

        :param destination: Name of the destination folder to contain plot
        :type destination: str

        :param stations: Add station names to the plot, defaults to True
        :type stations: bool, optional

        :param zoff: Add Elevation offsets to the plots, defaults to False
        :type zoff: bool, optional

        :param unit: Unit for the plot, valid values are length units, default is km
        :type unit: str, optional

        :param box_size: Size of the box for plotting the inner part of the array in unit, default is 5 km
        :type box_size: int, float, optional

        :param display: Display plots inline or suppress, defaults to True
        :type display: bool, optional

        :param figure_size: 2 element array/list/tuple with the plot sizes in inches
        :type figure_size: numpy.ndarray, list, tuple, optional

        :param dpi: dots per inch to be used in plots, default is 300
        :type dpi: int, optional

        .. _Description:


        """
        param_dict = locals()
        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        plot_array_configuration(self['antenna_info'], self['observation_info']['telescope_name'], param_dict)
        return

    def summary(self) -> None:
        """ Prints summary of the AstrohackLocitFile object, with available data, attributes and available methods
        """
        print_summary_header(self.file)
        print_dict_table(self._input_pars)
        print_data_contents(self, ["Antenna", "Contents"])
        print_method_list(
            [
                self.summary,
                self.print_source_table,
                self.print_array_configuration,
                self.plot_source_positions,
                self.plot_array_configuration
            ]
        )


class AstrohackPositionFile(dict):
    """ Data Class for extracted antenna location determination
    """

    def __init__(self, file: str):
        """ Initialize an AstrohackPositionFile object.
        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackPositionFile object
        :rtype: AstrohackPositionFile
        """
        super().__init__()

        self.combined = None
        self.file = file
        self._meta_data = None
        self._input_pars = None
        self._file_is_open = False

    def __getitem__(self, key: str):
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return super().__setitem__(key, value)

    @property
    def is_open(self) -> bool:
        """ Check whether the object has opened the corresponding hack file.

        :return: True if open, else False.
        :rtype: bool
        """
        return self._file_is_open

    def open(self, file: str = None, dask_load: bool = True) -> bool:
        """ Open antenna location file.
        :param file: File to be opened, if None defaults to the previously defined file
        :type file: str, optional

        :param dask_load: Is file to be loaded with dask?, default is True
        :type dask_load: bool, optional

        :return: True if file is properly opened, else returns False
        :rtype: bool
        """

        if file is None:
            file = self.file

        self._meta_data = read_meta_data(file + '/.position_attr')
        self.combined = self._meta_data['combine_ddis'] != 'no'
        self._input_pars = read_meta_data(file + '/.position_input')

        try:
            load_position_file(
                file=file,
                dask_load=dask_load,
                position_dict=self,
                combine=self.combined
            )
            self._meta_data = read_meta_data(file + '/.position_attr')
            self.combined = self._meta_data['combine_ddis'] != 'no'
            self._input_pars = read_meta_data(file + '/.position_input')

            self._file_is_open = True

        except Exception as error:
            logger.error(f"There was an exception opening the file: {error}")
            self._file_is_open = False

        return self._file_is_open

    @toolviper.utils.parameter.validate(
        custom_checker=custom_unit_checker
    )
    def export_locit_fit_results(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            position_unit: str = 'm',
            time_unit: str = 'hour',
            delay_unit: str = 'nsec'
    ) -> None:
        """ Export antenna position fit results to a text file.

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

        :param delay_unit: Unit for delays, defaults to 'ns'
        :type delay_unit: str, optional

        .. _Description:

        Produce a text file with the fit results from astrohack.locit for better determination of antenna locations.
        """

        param_dict = locals()
        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        param_dict['combined'] = self.combined
        export_locit_fit_results(self, param_dict)

    @toolviper.utils.parameter.validate(
        custom_checker=custom_unit_checker
    )
    def plot_sky_coverage(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            time_unit: str = 'hour',
            angle_unit: str = 'deg',
            display: bool = False,
            figure_size: Union[Tuple, List[float], np.array] = None,
            dpi: int = 300,
            parallel: bool = False
    ) -> None:
        """ Plot the sky coverage of the data used for antenna position fitting

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
        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        param_dict['combined'] = self.combined
        if self.combined:
            compute_graph(self, plot_sky_coverage_chunk, param_dict, ['ant'], parallel=parallel)
        else:
            compute_graph(self, plot_sky_coverage_chunk, param_dict, ['ant', 'ddi'],
                          parallel=parallel)

    @toolviper.utils.parameter.validate(
        custom_checker=custom_unit_checker
    )
    def plot_delays(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            time_unit: str = 'hour',
            angle_unit: str = 'deg',
            delay_unit: str = 'nsec',
            plot_model: bool = True,
            display: bool = False,
            figure_size: Union[Tuple, List[float], np.array] = None,
            dpi: int = 300,
            parallel: bool = False
    ) -> None:
        """ Plot the delays used for antenna position fitting and optionally the resulting fit.

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
        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)

        param_dict['combined'] = self.combined
        param_dict['comb_type'] = self._meta_data["combine_ddis"]
        if self.combined:
            compute_graph(self, plot_delays_chunk, param_dict, ['ant'], parallel=parallel)
        else:
            compute_graph(self, plot_delays_chunk, param_dict, ['ant', 'ddi'],
                          parallel=parallel)

    @toolviper.utils.parameter.validate(
        custom_checker=custom_unit_checker
    )
    def plot_position_corrections(
            self,
            destination: str,
            ant: Union[str, List[str]] = "all",
            ddi: Union[int, List[int]] = "all",
            unit: str = 'km',
            box_size: Union[int, float] = 5,
            scaling: Union[int, float] = 250,
            figure_size: Union[Tuple, List[float], np.array] = None,
            display: bool = True,
            dpi: int = 300
    ) -> None:
        """ Plot Antenna position corrections on an array configuration plot

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

        :param display: Display plots inline or suppress, defaults to True
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
        pathlib.Path(param_dict['destination']).mkdir(exist_ok=True)
        param_dict['combined'] = self.combined
        plot_position_corrections(param_dict, self)

    def summary(self) -> None:
        """ Prints summary of the AstrohackPositionFile object, with available data, attributes and available methods
        """
        print_summary_header(self.file)
        print_dict_table(self._input_pars)
        if self.combined:
            print_data_contents(self, ["Antenna"])
        else:
            print_data_contents(self, ["Antenna", "Contents"])
        print_method_list([self.summary, self.export_locit_fit_results, self.plot_sky_coverage, self.plot_delays,
                           self.plot_position_corrections])
