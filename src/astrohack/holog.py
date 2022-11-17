import dask
import time
import os

import numpy as np
import xarray as xr
import dask.array as da

from numba import njit
from numba.core import types
from numba.typed import Dict

from astrohack._utils._io import _load_pnt_dict
from astrohack._utils._io import _make_ant_pnt_dict
from astrohack._utils._io import _extract_holog_chunk

from casacore import tables as ctables


def holog(hack_name):
        print(hack_name)



