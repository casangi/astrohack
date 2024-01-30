import sys
import xarray as xr
import astrohack
import os
import skriba.logger

py310 = sys.version_info >= (3, 10)

if py310:
    from importlib.resources import files as pkgfiles
else:
    from importlib_resources import files as pkgfiles

tel_data_path = pkgfiles(astrohack)/'data/telescopes/'


def _find_cfg_file(name, path):
    """
    Search for the correct telescope configuration file
    Args:
        name: Name of the telescope configuration file
        path: Path to search for the telescope configuration file

    Returns:
        fullpath to the telescope configuration file
    """
    newpath = None
    for root, dirs, files in os.walk(path):
        if name in dirs:
            newpath = os.path.join(root, name)

    if newpath is None:
        raise FileNotFoundError
    else:
        return newpath


class Telescope:
    def __init__(self, name: str, path=None):
        """
        Initializes antenna surface relevant information based on the telescope name
        Args:
            name: telescope name, spaces are replaced by _ when searching
            path: Path in which to look for telescope configuration files, defaults to the astrohack package
            data directory if None
        """

        filename = self._get_telescope_file_name(name).lower().replace(" ", "_") + ".zarr"

        try:
            if path is None:
                filepath = _find_cfg_file(filename, tel_data_path)
            else:
                filepath = _find_cfg_file(name, path)
        except FileNotFoundError:
            raise Exception("Unknown telescope: " + name)
        
        self.read(filepath)
        
        if self.ringed:
            self._ringed_consistency()
        
        else:
            self._general_consistency()
        
        return

    @staticmethod
    def _get_telescope_file_name(name):
        """
        Open correct telescope based on the telescope name string.
        Args:
            name: telescope name string

        Returns:
        appropriate telescope object
        """
        if 'VLA' in name:
            name = 'VLA'
        elif 'ALMA' in name:
            # It does not matter which ALMA layout since the array center is the same
            name = 'ALMA_DA'

        return name

    def _ringed_consistency(self):
        """
        Performs a consistency check on the telescope parameters for the ringed telescope case
        """
        error = False
        logger = skriba.logger.get_logger(logger_name="astrohack")
        if not self.nrings == len(self.inrad) == len(self.ourad):
            logger.error("Number of panels don't match radii or number of panels list sizes")
            error = True
        if not self.onaxisoptics:
            logger.error("Off axis optics not yet supported")
            error = True
        if error:
            raise Exception("Failed Consistency check")
        return

    def _general_consistency(self):
        """
        For the moment simply raises an Exception since only ringed telescopes are supported at the moment
        """
        raise Exception("General layout telescopes not yet supported")

    def write(self, filename):
        """
        Write the telescope object to an X array .zarr telescope configuration file
        Args:
            filename: Name of the output file
        """
        ledict = vars(self)
        xds = xr.Dataset()
        xds.attrs = ledict
        xds.to_zarr(filename, mode="w", compute=True, consolidated=True)
        return

    def read(self, filename):
        """
        Read the telescope object from an X array .zarr telescope configuration file
        Args:
            filename: name of the input file
        """
        xds = xr.open_zarr(filename)
        for key in xds.attrs:
            setattr(self, key, xds.attrs[key])
        return

    def print(self):
        """
        Prints all the parameters defined for the telescope object
        """
        ledict = vars(self)
        for key in ledict:
            print("{0:15s} = ".format(key)+str(ledict[key]))
