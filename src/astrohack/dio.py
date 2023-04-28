import os
import dask
import numpy as np

from astropy.time import Time
from casacore import tables

from astrohack._utils._constants import length_units, trigo_units, plot_types
from astrohack._utils._parm_utils._check_parms import _check_parms
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._utils import _parm_to_list

from astrohack._utils._io import _load_point_file
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

from astrohack._utils._panel import _plot_antenna_chunk


def export_screws(panel_mds_name, destination, ant_name=None, ddi=None,  unit='mm'):
    """ Export screw adjustment from panel to text file and save to disk.

    :param panel_mds_name: Input panel_mds file
    :type panel_mds_name: str
    :param destination: Name of the destination folder to contain exported screw adjustments
    :type destination: str
    :param ant_name: List of antennae/antenna to be exported, defaults to "all" when None
    :type ant_name: list or str, optional, ex. ant_ea25
    :param ddi: List of ddis/ddi to be exported, defaults to "all" when None
    :type ddi: list or str, optional, ex. ddi_0
    :param unit: Unit for screws adjustments, most length units supported, defaults to "mm"
    :type unit: str

    .. _Description:

    Produce the screw adjustments from ``astrohack.panel`` results to be used at the antenna site to improve the antenna surface

    """
    logger = _get_astrohack_logger()
    parm_dict = {'filename': panel_mds_name,
                 'ant_name': ant_name,
                 'ddi': ddi,
                 'destination': destination,
                 'unit': unit}

    parms_passed = _check_parms(parm_dict, 'filename', [str], default=None)
    parms_passed = parms_passed and _check_parms(parm_dict, 'ant_name', [list], list_acceptable_data_types=[str],
                                                 default='all')
    parms_passed = parms_passed and _check_parms(parm_dict, 'ddi', [list], list_acceptable_data_types=[str],
                                                 default='all')
    parms_passed = parms_passed and _check_parms(parm_dict, 'destination', [str], default=None)
    parms_passed = parms_passed and _check_parms(parm_dict, 'unit', [str], acceptable_data=length_units, default='mm')

    if not parms_passed:
        logger.error("export_scews parameter checking failed.")
        raise Exception("export_screws parameter checking failed.")

    panel_mds = AstrohackPanelFile(panel_mds_name)
    panel_mds.open()

    try:
        os.mkdir(parm_dict['destination'])
    except FileExistsError:
        logger.warning('Destination folder already exists, results may be overwritten')

    antennae = _parm_to_list(parm_dict['ant_name'], parm_dict['filename'])
    for antenna in antennae:
        if 'ant' in antenna:
            ddis = _parm_to_list(parm_dict['ddi'], parm_dict['filename']+'/'+antenna)
            for ddi in ddis:
                if 'ddi' in ddi:
                    export_name = parm_dict['destination']+f'/screws_{antenna}_{ddi}.txt'
                    surface = panel_mds.get_antenna(antenna, ddi)
                    surface.export_screws(export_name, unit=unit)


def plot_antenna(panel_mds_name, destination, ant_name=None, ddi=None, plot_type='deviation', plot_screws=False,
                 dpi=300, unit=None, parallel=True):
    """ Create diagnostic plots of antenna surface deviations from panel data file. Available plots listed in additional information.

    :param panel_mds_name: Input panel_mds file
    :type panel_mds_name: str
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
    :param dpi: dots per inch to be used in plots
    :type dpi: int
    :param unit: Unit for phase or deviation plots, defaults to "mm" for deviation and 'deg' for phase
    :type unit: str
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
        - *ancillary*: Three ancillary plots with useful information: The mask used to select data to be fitted, the
                       amplitude data used to derive the mask and the panel assignments of each pixel, units are
                       irrelevant for these plots
    """
    logger = _get_astrohack_logger()
    parm_dict = {'filename': panel_mds_name,
                 'ant_name': ant_name,
                 'ddi': ddi,
                 'destination': destination,
                 'unit': unit,
                 'plot_type': plot_type,
                 'plot_screws': plot_screws,
                 'dpi': dpi,
                 'parallel': parallel}

    parms_passed = _check_parms(parm_dict, 'filename', [str], default=None)
    parms_passed = parms_passed and _check_parms(parm_dict, 'ant_name', [list], list_acceptable_data_types=[str],
                                                 default='all')
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
    parms_passed = parms_passed and _check_parms(parm_dict, 'dpi', [int], default=300)

    if not parms_passed:
        logger.error("export_scews parameter checking failed.")
        raise Exception("export_screws parameter checking failed.")

    panel_mds = AstrohackPanelFile(panel_mds_name)
    panel_mds.open()
    parm_dict['panel_mds'] = panel_mds

    try:
        os.mkdir(parm_dict['destination'])
    except FileExistsError:
        logger.warning('Destination folder already exists, results may be overwritten')

    delayed_list = []
    antennae = _parm_to_list(parm_dict['ant_name'], parm_dict['filename'])
    for antenna in antennae:
        if 'ant' in antenna:
            parm_dict['this_antenna'] = antenna
            ddis = _parm_to_list(parm_dict['ddi'], parm_dict['filename']+'/'+antenna)
            for ddi in ddis:
                if 'ddi' in ddi:
                    parm_dict['this_ddi'] = ddi
                    if parallel:
                        delayed_list.append(dask.delayed(_plot_antenna_chunk)(dask.delayed(parm_dict)))
                    else:
                        _plot_antenna_chunk(parm_dict)

    if parallel:
        dask.compute(delayed_list)

def open_holog(file):
  """ Open holog file and return instance of the holog data object. Object includes summary function to list available dictionary keys.

  :param file: Path to holog file.
  :type file: str
  
  :return: Holography holog object.
  :rtype: AstrohackHologFile

  .. _Description:
  **AstrohackHologFile**
  Holog object allows the user to access holog data via compound dictionary keys with values, in order of depth, `ddi` -> `map` -> `ant`. The holog object also provides a `summary()` helper function to list available keys for each file. An outline of the holog object structure is show below:

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
  """ Open image file and return instance of the image data object. Object includes summary function to list available dictionary keys.

  :param file: Path to image file.
  :type file: str
  
  :return: Holography image object.
  :rtype: AstrohackImageFile

  .. _Description:
  **AstrohackImageFile**
  Image object allows the user to access image data via compound dictionary keys with values, in order of depth, `ant` -> `ddi`. The image object also provides a `summary()` helper function to list available keys for each file. An outline of the image object structure is show below:

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
  """ Open panel file and return instance of the panel data object. Object includes summary function to list available dictionary keys.

  :param file: Path ot panel file.
  :type file: str
  
  :return: Holography panel object.
  :rtype: AstrohackPanelFile

  .. _Description:
  **AstrohackPanelFile**
  Panel object allows the user to access panel data via compound dictionary keys with values, in order of depth, `ant` -> `ddi`. The panel object also provides a `summary()` helper function to list available keys for each file. An outline of the panel object structure is show below:

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
  """ Open pointing file and return instance of the pointing data object. Object includes summary function to list available dictionary keys.

  :param file: Path to pointing file.
  :type file: str
  
  :return: Holography pointing object.
  :rtype: AstrohackPointFile

  .. _Description:

  **AstrohackPointFile**
  Pointing object allows the user to access pointing data via dictionary key with value based on `ant`. The pointing object also provides a `summary()` helper function to list available keys for each file. An outline of the pointing object structure is show below:

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

def fix_pointing_table(ms_name, reference_antenna):
  """ Fix pointing table for a user defined subset of reference antennas.

  Args:
      ms_name (str): Measurement set.
      reference_antenna (list): List of reference antennas.
  """
    
  ms_table = "/".join((ms_name, 'ANTENNA'))

  query = 'select NAME from {table}'.format(table=ms_table)

  ant_names = np.array(tables.taql(query).getcol('NAME'))
  ant_id = np.arange(len(ant_names))

  query_ant = np.searchsorted(ant_names, reference_antenna)

  ms_table = "/".join((ms_name, 'POINTING'))

  ant_list = " or ".join(["ANTENNA_ID=={ant}".format(ant=ant) for ant in query_ant])

  update = "update {table} set POINTING_OFFSET=0, TARGET=DIRECTION where {antennas}".format(table=ms_table, antennas=ant_list)

  tables.taql(update)

  ms_table = "/".join((ms_name, "HISTORY"))
  tb = tables.table(ms_table, readonly=False)
    
  message = tb.getcol("MESSAGE")
    
  if "pnt_tbl:fixed" not in message:
    tb.addrows(nrows=1)
    length = len(message)
    tb.putcol(columnname="MESSAGE", value='pnt_tbl:fixed', startrow=length)
