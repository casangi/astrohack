import os
import numpy as np
import numbers
from matplotlib import colormaps as cmaps

from astrohack._utils._dio import _load_image_xds
from prettytable import PrettyTable
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._dio import _read_meta_data
from astrohack._utils._dio import _load_holog_file
from astrohack._utils._dio import _load_image_file
from astrohack._utils._dio import _load_panel_file
from astrohack._utils._dio import _load_point_file
from astrohack._utils._parm_utils._check_parms import _check_parms
from astrohack._utils._constants import length_units, trigo_units, plot_types
from astrohack._utils._dask_graph_tools import _generate_antenna_ddi_graph_and_compute

from astrohack._utils._panel import _plot_antenna_chunk, _export_to_fits_panel_chunk, _export_screws_chunk
from astrohack._utils._holog import _export_to_fits_holog_chunk

from astrohack._classes.antenna_surface import AntennaSurface
from astrohack._classes.telescope import Telescope


class AstrohackDataFile:
    def __init__(self, file_stem, path='./'):
                        
        self._image_path = None
        self._holog_path = None
        self._panel_path = None
        self._point_path = None

        self.holog = None
        self.image = None
        self.panel = None
        self.point = None
            
        self._verify_holog_files(file_stem, path)
            

    def _verify_holog_files(self, file_stem, path):
        logger = _get_astrohack_logger()
        logger.info("Verifying {stem}.* files in path={path} ...".format(stem=file_stem, path=path))

        file_path = "{path}/{stem}.holog.zarr".format(path=path, stem=file_stem)
            
        if os.path.isdir(file_path):
            logger.info("Found {stem}.holog.zarr directory ...".format(stem=file_stem))
            
            self._holog_path = file_path
            self.holog = AstrohackHologFile(file_path)
                

        file_path = "{path}/{stem}.image.zarr".format(path=path, stem=file_stem)

        if os.path.isdir(file_path):
            logger.info("Found {stem}.image.zarr directory ...".format(stem=file_stem))
            
            self._image_path = file_path
            self.image = AstrohackImageFile(file_path)

        file_path = "{path}/{stem}.panel.zarr".format(path=path, stem=file_stem)

        if os.path.isdir(file_path):
            logger.info("Found {stem}.panel.zarr directory ...".format(stem=file_stem))
            
            self._image_path = file_path
            self.panel = AstrohackPanelFile(file_path)

        file_path = "{path}/{stem}.point.zarr".format(path=path, stem=file_stem)

        if os.path.isdir(file_path):
            logger.info("Found {stem}.point.zarr directory ...".format(stem=file_stem))
            
            self._point_path = file_path
            self.point = AstrohackPointFile(file_path)


class AstrohackImageFile(dict):
    """
        Data class for holography image data.
    """
    def __init__(self, file):
        super().__init__()
        self._meta_data = None
        self.file = file
        self._open = False

    def __getitem__(self, key):
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        return super().__setitem__(key, value)
        
    def is_open(self):
        return self._open

    def open(self, file=None):
        """ Open hologgraphy file.
        Args:self =_
            file (str, optional): Path to holography file. Defaults to None.
        Returns:
            bool: bool describing whether the file was opened properly
        """
        logger = _get_astrohack_logger()
        if file is None:
            file = self.file

        try:
            _load_image_file(file, image_dict=self)

            self._open = True

        except Exception as e:
            logger.error("[AstroHackImageFile.open()]: {}".format(e))
            self._open = False

        self._meta_data = _read_meta_data(file, 'image', ['combine', 'holog'])

        return self._open

    def summary(self):
        """
           Prints summary table of holog image file. 
        """

        print("Atributes:")
        for key in self._meta_data.keys():
            print(f'{key:26s}= {self._meta_data[key]}')

        table = PrettyTable()
        table.field_names = ["antenna", "ddi"]
        table.align = "l"
        
        for ant in self.keys():
            table.add_row([ant, list(self[ant].keys())])

        print('\nContents:')
        print(table)

    def select(self, ant=None, ddi=None, polar=False):
        """Select data on the basis of ddi, scan, ant. This is a convenience function.
        Args:
            ddi (int, optional): Data description ID. Defaults to None.
            ant (int, optional): Antenna ID. Defaults to None.
        Returns:
            xarray.Dataset: xarray dataset of corresponding ddi, scan, antenna ID.
        """
        logger = _get_astrohack_logger()
        
        if ant is None and ddi is None:
            logger.info("No selections made ...")
            return self
        else:
            if polar:
                return self[ant][ddi].apply(np.absolute), self[ant][ddi].apply(np.angle, deg=True)

            return self[ant][ddi]


class AstrohackHologFile(dict):
    """
        Data Class to interact ith holography imaging data.
    """
    def __init__(self, file):
        super().__init__()
        
        self.file = file
        self._meta_data = None
        self._open = False

    def __getitem__(self, key):
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    def is_open(self):
        return self._open

    def open(self, file=None, dask_load=True):
        """ Open hologgraphy file.
        Args:self =_
            file (str, optional): Path to holography file. Defaults to None.
            dask_load (bool, optional): If True the file is loaded with Dask. Defaults to True.
        Returns:
            bool: bool describing whether the file was opened properly
        """
        logger = _get_astrohack_logger()

        if file is None:
            file = self.file

        try:
            _load_holog_file(holog_file=file, dask_load=dask_load, load_pnt_dict=False, holog_dict=self)
            self._open = True

        except Exception as e:
            logger.error("[AstrohackHologFile]: {}".format(e))
            self._open = False

        self._meta_data = _read_meta_data(file, 'holog', 'extract_holog')

        return self._open

    def summary(self):
        """
            Prints summary table of holog file.
        """
        print("Atributes:")
        for key in self._meta_data.keys():
            if key == 'n_pix':
                n_side = int(np.sqrt(self._meta_data[key]))
                print(f'{key:26s}= {n_side:d} x {n_side:d}')
            else:
                print(f'{key:26s}= {self._meta_data[key]}')
        table = PrettyTable()
        table.field_names = ["ddi", "map", "antenna"]
        table.align = "l"
        
        for ddi in self.keys():
            for scan in self[ddi].keys():
                table.add_row([ddi, scan, list(self[ddi][scan].keys())])
        print('\nContents:')
        print(table)

    def select(self, ddi=None, scan=None, ant=None):
        """ Select data on the basis of ddi, scan, ant. This is a convenience function.
        Args:
            ddi (int, optional): Data description ID. Defaults to None.
            scan (int, optional): Scan number. Defaults to None.
            ant (int, optional): Antenna ID. Defaults to None.
        Returns:
            xarray.Dataset: xarray dataset of corresponding ddi, scan, antenna ID.
        """
        logger = _get_astrohack_logger()
        
        if ant is None or ddi is None or scan is None:
            logger.info("No selections made ...")
            return self
        else:
            return self[ddi][scan][ant]

    @property
    def meta_data(self):
        """ Holog file meta data.
        Returns:
            JSON: JSON file of holography meta data.
        """

        return self._meta_data


class AstrohackPanelFile(dict):
    """
        Data class for holography panel data.
    """
    def __init__(self, file):
        super().__init__()

        self.file = file
        self._open = False
        self._meta_data = None

    def __getitem__(self, key):
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        return super().__setitem__(key, value)
        
    def is_open(self):
        return self._open

    def open(self, file=None):
        """ Open panel file.
        Args:self =_
            file (str, optional): Path to holography file. Defaults to None.
        Returns:
            bool: bool describing whether the file was opened properly
        """
        logger = _get_astrohack_logger()

        if file is None:
            file = self.file

        try:
            _load_panel_file(file, panel_dict=self)
            self._open = True
        except Exception as e:
            logger.error("[AstroHackPanelFile.open()]: {}".format(e))
            self._open = False

        self._meta_data = _read_meta_data(file, 'panel', 'panel')

        return self._open

    def summary(self):
        """
           Prints summary table of panel image file.
        """

        print("Atributes:")
        for key in self._meta_data.keys():
            print(f'{key:26s}= {self._meta_data[key]}')

        table = PrettyTable()
        table.field_names = ["antenna", "ddi"]
        table.align = "l"
        
        for ant in self.keys():
            table.add_row([ant, list(self[ant].keys())])

        print('\nContents:')
        print(table)

    def get_antenna(self, antenna, ddi, dask_load=True):
        """
        Return an AntennaSurface object for interaction
        Args:
            antenna: Which antenna is to be used
            ddi: Which ddi is to be used
            dask_load: Load xds using dask?
        Returns:
            AntennaSurface object contaning relevant information for panel adjustments
        """
        xds = _load_image_xds(self.file, antenna, ddi, dask_load)
        telescope = Telescope(xds.attrs['telescope_name'])
        
        return AntennaSurface(xds, telescope, reread=True)

    def export_screws(self, destination, ant_name=None, ddi=None, unit='mm', threshold=None, plot_map=False,
                      colormap='seismic', figuresize=None, dpi=300):
        """ Export screw adjustment from panel to text file and save to disk.

        :param destination: Name of the destination folder to contain exported screw adjustments
        :type destination: str
        :param ant_name: List of antennae/antenna to be exported, defaults to "all" when None
        :type ant_name: list or str, optional, ex. ant_ea25
        :param ddi: List of ddis/ddi to be exported, defaults to "all" when None
        :type ddi: list or str, optional, ex. ddi_0
        :param unit: Unit for screws adjustments, most length units supported, defaults to "mm"
        :type unit: str
        :param threshold: Threshold below which data is considered negligable, value is assumed to be in the same unit as the plot, if not given defaults to 10% of the maximal deviation
        :type threshold: float, optional
        :param plot_map: Plot the map of screw adjustments, default is False
        :type plot_map: bool
        :param colormap: Colormap for screw adjustment map
        :type colormap: str
        :param figuresize: 2 element array/list/tuple with the screw adjustment map size in inches
        :type figuresize: numpy.ndarray, list, tuple, optional
        :param dpi: Screw adjustment map resolution in pixels per inch
        :type dpi: int

        .. _Description:

        Produce the screw adjustments from ``astrohack.panel`` results to be used at the antenna site to improve the antenna surface

        """
        logger = _get_astrohack_logger()
        parm_dict = {'ant_name': ant_name,
                     'ddi': ddi,
                     'destination': destination,
                     'unit': unit,
                     'threshold': threshold,
                     'plot_map': plot_map,
                     'colormap': colormap,
                     'figuresize': figuresize,
                     'dpi': dpi}

        parms_passed = _check_parms(parm_dict, 'ant_name', [list], list_acceptable_data_types=[str], default='all')
        parms_passed = parms_passed and _check_parms(parm_dict, 'ddi', [list], list_acceptable_data_types=[str], default='all')
        parms_passed = parms_passed and _check_parms(parm_dict, 'destination', [str], default=None)
        parms_passed = parms_passed and _check_parms(parm_dict, 'unit', [str], acceptable_data=length_units, default='mm')
        parms_passed = parms_passed and _check_parms(parm_dict, 'threshold', [numbers.Number], default=None)
        parms_passed = parms_passed and _check_parms(parm_dict, 'plot_map', [bool], default=False)
        parms_passed = parms_passed and _check_parms(parm_dict, 'colormap', [str], acceptable_data=cmaps, default='RdBu_r')
        parms_passed = parms_passed and _check_parms(parm_dict, 'figuresize', [list, np.ndarray],
                                                     list_acceptable_data_types=[numbers.Number], list_len=2,
                                                     default='None', log_default_setting=False)
        parms_passed = parms_passed and _check_parms(parm_dict, 'dpi', [int], default=300)

        if not parms_passed:
            logger.error("export_screws parameter checking failed.")
            raise Exception("export_screws parameter checking failed.")

        parm_dict['panel_mds'] = self
        parm_dict['filename'] = self.file

        try:
            os.mkdir(parm_dict['destination'])
        except FileExistsError:
            logger.warning('Destination folder already exists, results may be overwritten')

        _generate_antenna_ddi_graph_and_compute('export_screws', _export_screws_chunk, parm_dict, False)

    def plot_antennas(self, destination, ant_name=None, ddi=None, plot_type='deviation', plot_screws=False, unit=None,
                      colormap='viridis', figuresize=None, dpi=300, parallel=True):
        """ Create diagnostic plots of antenna surface deviations from panel data file. Available plots listed in additional information.

        :param destination: Name of the destination folder to contain plots
        :type destination: str
        :param ant_name: List of antennae/antenna to be plotted, defaults to "all" when None
        :type ant_name: list or str, optional, ex. ant_ea25
        :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None
        :type ddi: list or str, optional, ex. ddi_0
        :param plot_type: type of plot to be produced, deviation, phase or ancillary
        :type plot_type: str
        :param plot_screws: Add screw positions to plot
        :type plot_screws: bool
        :param unit: Unit for phase or deviation plots, defaults to "mm" for deviation and 'deg' for phase
        :type unit: str
        :param colormap: Colormap for plots
        :type colormap: str
        :param figuresize: 2 element array/list/tuple with the plot sizes in inches
        :type figuresize: numpy.ndarray, list, tuple, optional
        :param dpi: dots per inch to be used in plots
        :type dpi: int
        :param parallel: If True will use an existing astrohack client to produce plots in parallel
        :type parallel: bool

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
        - *all*: All the plots listed above
        """
        logger = _get_astrohack_logger()
        parm_dict = {'ant_name': ant_name,
                     'ddi': ddi,
                     'destination': destination,
                     'unit': unit,
                     'plot_type': plot_type,
                     'plot_screws': plot_screws,
                     'colormap': colormap,
                     'figuresize': figuresize,
                     'dpi': dpi,
                     'parallel': parallel}

        parms_passed = _check_parms(parm_dict, 'ant_name', [list], list_acceptable_data_types=[str], default='all')
        parms_passed = parms_passed and _check_parms(parm_dict, 'ddi', [list], list_acceptable_data_types=[str],
                                                     default='all')
        parms_passed = parms_passed and _check_parms(parm_dict, 'destination', [str], default=None)
        parms_passed = parms_passed and _check_parms(parm_dict, 'plot_type', [str], acceptable_data=plot_types,
                                                     default=plot_types[0])
        if parm_dict['plot_type'] == plot_types[0]:  # Length units for deviation plots
            parms_passed = parms_passed and _check_parms(parm_dict, 'unit', [str], acceptable_data=length_units,
                                                         default='mm')
        elif parm_dict['plot_type'] == plot_types[1]:  # Trigonometric units for phase plots
            parms_passed = parms_passed and _check_parms(parm_dict, 'unit', [str], acceptable_data=trigo_units,
                                                         default='deg')
        else:  # Units ignored for ancillary plots
            logger.info('Unit ignored for ancillary plots')
        parms_passed = parms_passed and _check_parms(parm_dict, 'parallel', [bool], default=True)
        parms_passed = parms_passed and _check_parms(parm_dict, 'plot_screws', [bool], default=False)
        parms_passed = parms_passed and _check_parms(parm_dict, 'colormap', [str], acceptable_data=cmaps, default='viridis')
        parms_passed = parms_passed and _check_parms(parm_dict, 'figuresize', [list, np.ndarray],
                                                     list_acceptable_data_types=[numbers.Number], list_len=2,
                                                     default='None', log_default_setting=False)
        parms_passed = parms_passed and _check_parms(parm_dict, 'dpi', [int], default=300)

        if not parms_passed:
            logger.error("plot_antennas parameter checking failed.")
            raise Exception("plot_antennas parameter checking failed.")

        parm_dict['panel_mds'] = self
        parm_dict['filename'] = self.file

        try:
            os.mkdir(parm_dict['destination'])
        except FileExistsError:
            logger.warning('Destination folder already exists, results may be overwritten')

        _generate_antenna_ddi_graph_and_compute('plot_antennas', _plot_antenna_chunk, parm_dict, parallel)


class AstrohackPointFile(dict):
    """
        Data Class to interact ith holography pointing data.
    """
    def __init__(self, file):
        super().__init__()
        
        self.file = file
        self._meta_data = None
        self._open = False

    def __getitem__(self, key):
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    def is_open(self):
        return self._open

    def open(self, file=None, dask_load=True):
        """ Open pointing file.
        Args:self =_
            file (str, optional): Path to pointing file. Defaults to None.
            dask_load (bool, optional): If True the file is loaded with Dask. Defaults to True.
        Returns:
            bool: bool describing whether the file was opened properly
        """
        logger = _get_astrohack_logger()

        if file is None:
            file = self.file

        try:
            _load_point_file(file=file, dask_load=dask_load, pnt_dict=self)
            self._open = True

        except Exception as e:
            logger.error("[AstrohackPointFile]: {}".format(e))
            self._open = False

        self._meta_data = _read_meta_data(file, 'point', 'extract_holog')

        return self._open

    def summary(self):
        """
            Prints summary table of pointing file.
        """
        print("Atributes:")
        for key in self._meta_data.keys():
            print(f'{key:26s}= {self._meta_data[key]}')

        table = PrettyTable()
        table.field_names = ["antenna"]
        table.align = "l"
        
        for ant in self.keys():
            table.add_row([ant])

        print('\nContents:')
        print(table)
