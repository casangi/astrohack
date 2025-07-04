import astrohack
import pathlib

import xarray as xr
import toolviper.utils.logger as logger

import astrohack.utils.tools


class Telescope:
    def __init__(self, name: str, path=None):
        """
        Initializes antenna surface relevant information based on the telescope name
        Args:
            name: telescope name, spaces are replaced by _ when searching
            path: Path in which to look for telescope configuration files, defaults to the astrohack package
            data directory if None
        """
        self.onaxisoptics = None
        self.ourad = None
        self.inrad = None
        self.nrings = None
        self.ringed = None
        self.diam = None
        self.focus = None

        self.ant_list = []

        self.filename = (
            self._get_telescope_file_name(name).lower().replace(" ", "_") + ".zarr"
        )

        if path is None:
            self.filepath = astrohack.utils.tools.file_search(
                root=astrohack.__path__[0], file_name=self.filename
            )
        else:
            self.filepath = astrohack.utils.tools.file_search(
                root=path, file_name=self.filename
            )

        self.read(pathlib.Path(self.filepath).joinpath(self.filename))

        if self.ringed:
            self._ringed_consistency()

        else:
            self._general_consistency()

        return

    @classmethod
    def from_xds(cls, xds):
        tel_name = xds.attrs["summary"]["general"]["telescope name"]
        if tel_name == "ALMA":
            telescope_name = "_".join((tel_name, xds.attrs["ant_name"][0:2]))
            return cls(telescope_name)
        elif tel_name == "EVLA" or tel_name == "VLA":
            telescope_name = "VLA"
            return cls(telescope_name)
        else:
            raise ValueError("Unsupported telescope {0:s}".format(tel_name))

    @staticmethod
    def _get_telescope_file_name(name):
        """
        Open correct telescope based on the telescope name string.
        Args:
            name: telescope name string

        Returns:
        appropriate telescope object
        """
        name = name.lower()
        if "ngvla" in name:
            name = "ngVLA_prototype"
        elif "vla" in name:
            name = "VLA"
        elif "alma" in name:
            if "dv" in name:
                name = "ALMA_DV"
            elif "tp" in name:
                name = "ALMA_TP"
            else:
                name = "ALMA_DA"

        return name

    def _ringed_consistency(self):
        """
        Performs a consistency check on the telescope parameters for the ringed telescope case
        """
        error = False

        if not self.nrings == len(self.inrad) == len(self.ourad):
            logger.error(
                "Number of panels don't match radii or number of panels list sizes"
            )
            error = True

        if not self.onaxisoptics:
            logger.error("Off axis optics not yet supported")
            error = True

        if error:
            raise Exception("Failed Consistency check")

        return

    def _general_consistency(self):
        """
        For the moment does nothing as a general consistency test is not yet available
        """
        pass

    def write(self, filename):
        """
        Write the telescope object to an X array .zarr telescope configuration file
        Args:
            filename: Name of the output file
        """
        obj_dict = vars(self)
        xds = xr.Dataset()
        xds.attrs = obj_dict
        xds.to_zarr(filename, mode="w", compute=True, consolidated=True)
        return

    def _save_to_dist(self):
        obj_dict = vars(self)
        filename = f"{self.filepath}/{self.filename}"
        obj_dict.pop("filepath", None)
        obj_dict.pop("filename", None)
        xds = xr.Dataset()
        xds.attrs = obj_dict
        xds.to_zarr(filename, mode="w", compute=True, consolidated=True)
        return

    def read(self, filename):
        """
        Read the telescope object from an X array .zarr telescope configuration file
        Args:
            filename: name of the input file
        """
        try:
            xds = xr.open_zarr(filename)
            for key in xds.attrs:
                setattr(self, key, xds.attrs[key])

        except FileNotFoundError:
            logger.error(f"Telescope file not found: {filename}")
            raise FileNotFoundError

    def print(self):
        """
        Prints all the parameters defined for the telescope object
        """
        print(self)

    def __repr__(self):
        outstr = ""
        obj_dict = vars(self)
        for key, item in obj_dict.items():
            outstr += f"{key:20s} = {str(item)}\n"
        return outstr
