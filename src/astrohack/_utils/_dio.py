import os
import numpy as np
from astrohack._utils._io import _load_image_xds
from prettytable import PrettyTable
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._io import _read_meta_data
from astrohack._utils._io import _load_holog_file
from astrohack._utils._io import _load_image_file
from astrohack._utils._io import _load_panel_file
from astrohack._utils._io import _load_point_file

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

        return self._open

    def summary(self):
        """
           Prints summary table of holog image file. 
        """

        table = PrettyTable()
        table.field_names = ["antenna", "ddi"]
        table.align = "l"
        
        for ant in self.keys():
            table.add_row([ant, list(self[ant].keys())])
        
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

    def open(self, file=None, dask_load=False):
        """ Open hologgraphy file.
        Args:self =_
            file (str, optional): Path to holography file. Defaults to None.
            dask_load (bool, optional): If True the file is loaded with Dask. Defaults to False.
        Returns:
            bool: bool describing whether the file was opened properly
        """
        logger = _get_astrohack_logger()

        if file is None:
            file = self.file

        self._meta_data = _read_meta_data(holog_file=file)

        try:
            _load_holog_file(holog_file=file, dask_load=dask_load, load_pnt_dict=False, holog_dict=self)
            self._open = True

        except Exception as e:
            logger.error("[AstrohackHologFile]: {}".format(e))
            self._open = False
        
        return self._open

    def summary(self):
        """
            Prints summary table of holog file.
        """

        table = PrettyTable()
        table.field_names = ["ddi", "map", "antenna"]
        table.align = "l"
        
        for ddi in self.keys():
            for scan in self[ddi].keys():
                table.add_row([ddi, scan, list(self[ddi][scan].keys())])
        
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

        return self._open

    def summary(self):
        """
           Prints summary table of panel image file.
        """

        table = PrettyTable()
        table.field_names = ["antenna", "ddi"]
        table.align = "l"
        
        for ant in self.keys():
            table.add_row([ant, list(self[ant].keys())])
        
        print(table)

    def get_antenna(self, antenna, ddi):
        """
        Return an AntennaSurface object for interaction
        Args:
            antenna: Which antenna in to be used
            ddi: Which ddi is to be used
        Returns:
            AntennaSurface object contaning relevant information for panel adjustments
        """
        xds = _load_image_xds(self.file, antenna, ddi)
        telescope = Telescope(xds.attrs['telescope_name'])
        
        return AntennaSurface(xds, telescope, reread=True)


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

    def open(self, file=None, dask_load=False):
        """ Open pointing file.
        Args:self =_
            file (str, optional): Path to pointing file. Defaults to None.
            dask_load (bool, optional): If True the file is loaded with Dask. Defaults to False.
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
        
        return self._open

    def summary(self):
        """
            Prints summary table of pointing file.
        """

        table = PrettyTable()
        table.field_names = ["antenna"]
        table.align = "l"
        
        for ant in self.keys():
            table.add_row(ant)
        
        print(table)