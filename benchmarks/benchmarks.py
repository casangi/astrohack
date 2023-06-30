import os
from astrohack.gdown_utils import download
from astrohack import astrohack_local_client
from astrohack.extract_holog import extract_holog

from astrohack.holog import holog


class Stakeholder:
    def setup_cache(self):
        # initialize a local cluster for parallel processing
        self.client = astrohack_local_client(cores=2, memory_limit="8GB")

        def setup(self):
        # download test datasets
        download(
            "J1924-2914.ms.calibrated.split.SPW3", folder=os.getcwd(), unpack=True
        )
        self.alma_ms = os.path.join(os.getcwd(), "J1924-2914.ms.calibrated.split.SPW3")
        print(self.alma_ms)
        self.alma_zarr = "alma.split.holog.zarr"
        extract_holog(
            ms_name=self.alma_ms,
            holog_name=self.alma_zarr,
            data_column="DATA",
            parallel=True,
            overwrite=True,
            reuse_point_zarr=False,
        )

    def time_holog(self):
        holog(
            holog_name=self.alma_zarr,
            padding_factor=50,
            grid_interpolation_mode="linear",
            chan_average=True,
            scan_average=True,
            overwrite=True,
            phase_fit=True,
            apply_mask=True,
            to_stokes=True,
            parallel=True,
        )

class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def time_keys(self):
        for key in self.d.keys():
            pass

    def time_values(self):
        for value in self.d.values():
            pass

    def time_range(self):
        d = self.d
        for key in range(500):
            x = d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256
