import os
import dask
import sys

import xarray as xr
import numpy as np

from casacore import tables as ctables

from prettytable import PrettyTable

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

from astrohack._utils._holog import _create_holog_meta_data 

from astrohack._utils._io import _load_point_file 
from astrohack._utils._io import _extract_holog_chunk
from astrohack._utils._io import _open_no_dask_zarr
from astrohack._utils._io import _read_data_from_holog_json
from astrohack._utils._io import _read_meta_data
from astrohack._utils._io import _load_holog_file
from astrohack._utils._io import _load_image_file
from astrohack._utils._io import _load_panel_file

from astrohack._utils._dio import AstrohackImageFile
from astrohack._utils._dio import AstrohackHologFile
from astrohack._utils._dio import AstrohackPanelFile
from astrohack._utils._dio import AstrohackPointFile

def open_holog(file):
  """Method to return an instance of the holography data object.

  :param file: Path to holog file.
  :type file: str
  
  :return: Holography holog object.
  :rtype: AstrohackHologFile

  .. parsed-literal::
    holog_mds = 
      {
      ddi_0:{
          map_0:{
                 ant_0: holog_ds,
                          ⋮
                 ant_n: holog_ds
                },
              ⋮
          map_p: …
         },
       ⋮
      ddi_m: …
    }

  """

  logger = _get_astrohack_logger()

  _data_file = AstrohackHologFile(file=file)
  
  if _data_file.open():
    return _data_file

  else:
    logger.error("Error opening holgraphy file: {file}".format(file))

def open_image(file):
  """Method to return an instance of the image data object.

  :param file: Path to image file.
  :type file: str
  
  :return: Holography image object.
  :rtype: AstrohackImageFile

  .. parsed-literal::
    image_mds = 
      {
        ant_0:{
          ddi_0: image_ds,
                 ⋮               
          ddi_m: image_ds
         },
        ⋮
        ant_n: …
      }

  """

  logger = _get_astrohack_logger()

  _data_file = AstrohackImageFile(file=file)
  
  if _data_file.open():
    return _data_file

  else:
    logger.error("Error opening holgraphy image file: {file}".format(file))

def open_panel(file):
  """Method to return an instance of the holography panel object.

  :param file: Path ot panel file.
  :type file: str
  
  :return: Holography panel object.
  :rtype: AstrohackPanelFile

  .. parsed-literal::
    panel_mds = 
      {
        ant_0:{
          ddi_0: panel_ds,
                 ⋮               
          ddi_m: panel_ds
         },
        ⋮
        ant_n: …
      }

  """

  logger = _get_astrohack_logger()

  _data_file = AstrohackPanelFile(file=file)
  
  if _data_file.open():
    return _data_file

  else:
    logger.error("Error opening holgraphy panel file: {file}".format(file))

def open_pointing(file):
  """Method to return an instance of the holography point object.

  :param file: Path to pointing file.
  :type file: str
  
  :return: Holography pointing object.
  :rtype: AstrohackPointFile

  .. parsed-literal::
    point_mds = 
      {
        ant_0: point_ds,
            ⋮
        ant_n: point_ds
      }


  """

  logger = _get_astrohack_logger()

  _data_file = AstrohackPointFile(file=file)
  
  if _data_file.open():
    return _data_file

  else:
    logger.error("Error opening holgraphy pointing file: {file}".format(file))


