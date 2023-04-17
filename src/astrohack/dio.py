import os
import dask
import sys

import xarray as xr
import numpy as np

from casacore import tables as ctables

from prettytable import PrettyTable

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

from astrohack._utils._holog import _create_holog_meta_data 

from astrohack._utils._io import _load_pnt_dict 
from astrohack._utils._io import _extract_holog_chunk
from astrohack._utils._io import _open_no_dask_zarr
from astrohack._utils._io import _read_data_from_holog_json
from astrohack._utils._io import _read_meta_data
from astrohack._utils._io import _load_holog_file
from astrohack._utils._io import _load_image_file
from astrohack._utils._io import _load_panel_file

from astrohack._utils._dio import AstrohackDataFile
from astrohack._utils._dio import AstrohackImageFile
from astrohack._utils._dio import AstrohackHologFile
from astrohack._utils._dio import AstrohackPanelFile

def open_holog(file_stem, path="./"):
  """ Method to return an instance of the holography data file

  :param file_stem: Name of holog file (.holog.zarr)
  :type file_stem: str
  :param path: Path to holog file if not current directory.
  :type path: str
  """

  logger = _get_astrohack_logger()

  _data_file = AstrohackDataFile(file_stem=file_stem, path=path)
  
  if _data_file.holog.open():
    return _data_file.holog

  else:
    logger.error("Error opening holgraphy file: {file}".format(file_stem))

def open_image(file_stem, path="./"):
  """ Method to return an instance of the holography image file

  :param file_stem: Name of holog image file (.image.zarr)
  :type file_stem: str
  :param path: Path to holog image file if not current directory.
  :type path: str
  """

  logger = _get_astrohack_logger()

  _data_file = AstrohackDataFile(file_stem=file_stem, path=path)
  
  if _data_file.image.open():
    return _data_file.image

  else:
    logger.error("Error opening holgraphy image file: {file}".format(file_stem))

def open_panel(file_stem, path="./"):
  """ Method to return an instance of the holography panel file

  :param file_stem: Name of holog panel file (.panel.zarr)
  :type file_stem: str

  :param path: Path to holog panel file if not current directory.
  :type path: str
  """

  logger = _get_astrohack_logger()

  _data_file = AstrohackDataFile(file_stem=file_stem, path=path)
  
  if _data_file.panel.open():
    return _data_file.panel

  else:
    logger.error("Error opening holgraphy panel file: {file}".format(file_stem))


