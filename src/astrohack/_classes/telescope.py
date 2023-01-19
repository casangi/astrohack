import xarray as xr
import pkg_resources
import os

tel_data_path = pkg_resources.resource_filename('astrohack', '../../data/telescopes')


def _find_cfg_file(name, path):
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
            name: telescope name
        """
        filename = name.lower()+'.zarr'
        try:
            if path is None:
                filepath = _find_cfg_file(filename, tel_data_path)
            else:
                filepath = _find_cfg_file(name, path)
        except FileNotFoundError:
            raise Exception('Unknown telescope: '+name)
        self.read(filepath)
        if self.ringed:
            self._ringed_consistency()
        else:
            self._general_consistency()
        return

    def _ringed_consistency(self):
        if not self.nrings == len(self.inrad) == len(self.ourad):
            raise Exception('Number of panels don\'t match radii or number of panels list sizes')
        if not self.onaxisoptics:
            raise Exception('Off axis optics not yet supported')
        return

    def _general_consistency(self):
        raise Exception('General layout telescopes not yet supported')

    def write(self, filename):
        ledict = vars(self)
        xds = xr.Dataset()
        xds.attrs = ledict
        xds.to_zarr(filename, mode='w', compute=True, consolidated=True)
        return

    def read(self, filename):
        xds = xr.open_zarr(filename)
        for key in xds.attrs:
            setattr(self, key, xds.attrs[key])
        return
