import os
import numbers
import numpy as np
from matplotlib import colormaps as cmaps

from casacore import tables

from astrohack._utils._io import _check_mds_origin, check_if_file_exists
from astrohack._utils._constants import length_units, trigo_units, plot_types
from astrohack._utils._parm_utils._check_parms import _check_parms
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._dask_graph_tools import _generate_antenna_ddi_graph_and_compute

from astrohack._utils._dio import AstrohackImageFile
from astrohack._utils._dio import AstrohackHologFile
from astrohack._utils._dio import AstrohackPanelFile
from astrohack._utils._dio import AstrohackPointFile

from astrohack._utils._panel import _plot_antenna_chunk, _export_to_fits_panel_chunk, _export_screws_chunk
from astrohack._utils._holog import _export_to_fits_holog_chunk


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
        logger.error(f"Error opening holgraphy file: {file}")


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
        logger.error(f"Error opening holgraphy image file: {file}")


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
        logger.error(f"Error opening holgraphy panel file: {file}")


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
        logger.error(f"Error opening holgraphy pointing file: {file}")


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

    update = "update {table} set POINTING_OFFSET=0, TARGET=DIRECTION where {antennas}".format(table=ms_table,
                                                                                              antennas=ant_list)

    tables.taql(update)

    ms_table = "/".join((ms_name, "HISTORY"))
    tb = tables.table(ms_table, readonly=False)

    message = tb.getcol("MESSAGE")

    if "pnt_tbl:fixed" not in message:
        tb.addrows(nrows=1)
        length = len(message)
        tb.putcol(columnname="MESSAGE", value='pnt_tbl:fixed', startrow=length)


def export_screws(panel_mds_name, destination, ant_name=None, ddi=None, unit='mm', threshold=None, plot_map=False,
                  colormap='seismic', figuresize=None, dpi=300):
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
    parm_dict = {'filename': panel_mds_name,
                 'ant_name': ant_name,
                 'ddi': ddi,
                 'destination': destination,
                 'unit': unit,
                 'threshold': threshold,
                 'plot_map': plot_map,
                 'colormap': colormap,
                 'figuresize': figuresize,
                 'dpi': dpi}

    parms_passed = _check_parms(parm_dict, 'filename', [str], default=None)
    parms_passed = parms_passed and _check_parms(parm_dict, 'ant_name', [list], list_acceptable_data_types=[str],
                                                 default='all')
    parms_passed = parms_passed and _check_parms(parm_dict, 'ddi', [list], list_acceptable_data_types=[str],
                                                 default='all')
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

    check_if_file_exists(panel_mds_name)
    panel_mds = AstrohackPanelFile(panel_mds_name)
    panel_mds.open()

    try:
        os.mkdir(parm_dict['destination'])
    except FileExistsError:
        logger.warning('Destination folder already exists, results may be overwritten')

    _generate_antenna_ddi_graph_and_compute('export_screws', _export_screws_chunk, parm_dict, False)


def plot_antenna(panel_mds_name, destination, ant_name=None, ddi=None, plot_type='deviation', plot_screws=False,
                 unit=None, colormap='viridis', figuresize=None, dpi=300, parallel=True):
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
    parm_dict = {'filename': panel_mds_name,
                 'ant_name': ant_name,
                 'ddi': ddi,
                 'destination': destination,
                 'unit': unit,
                 'plot_type': plot_type,
                 'plot_screws': plot_screws,
                 'colormap': colormap,
                 'figuresize': figuresize,
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
    parms_passed = parms_passed and _check_parms(parm_dict, 'colormap', [str], acceptable_data=cmaps, default='viridis')
    parms_passed = parms_passed and _check_parms(parm_dict, 'figuresize', [list, np.ndarray],
                                                 list_acceptable_data_types=[numbers.Number], list_len=2,
                                                 default='None', log_default_setting=False)
    parms_passed = parms_passed and _check_parms(parm_dict, 'dpi', [int], default=300)

    if not parms_passed:
        logger.error("plot_antenna parameter checking failed.")
        raise Exception("plot_antenna parameter checking failed.")

    check_if_file_exists(panel_mds_name)
    panel_mds = AstrohackPanelFile(panel_mds_name)
    panel_mds.open()
    parm_dict['panel_mds'] = panel_mds

    try:
        os.mkdir(parm_dict['destination'])
    except FileExistsError:
        logger.warning('Destination folder already exists, results may be overwritten')

    _generate_antenna_ddi_graph_and_compute('plot_antenna', _plot_antenna_chunk, parm_dict, parallel)


def export_to_fits(mds_name, destination, complex_split='cartesian', ant_name=None, ddi=None, parallel=True):
    """ Export contents of an Astrohack MDS file to several FITS files in the destination folder

    :param mds_name: Input panel_mds file
    :type mds_name: str
    :param destination: Name of the destination folder to contain plots
    :type destination: str
    :param complex_split: How to split complex data, cartesian (real + imaginary) or polar (amplitude + phase)
    :type complex_split: str
    :param ant_name: List of antennae/antenna to be plotted, defaults to "all" when None
    :type ant_name: list or str, optional, ex. ant_ea25
    :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None
    :type ddi: list or str, optional, ex. ddi_0
    :param parallel: If True will use an existing astrohack client to produce plots in parallel
    :type parallel: bool

    .. _Description:
    Export the products from either holog or panel onto FITS files to be read by other software packages

    **Additional Information**
    The image products of holog are complex images due to the nature of interferometric measurements and Fourier
    transforms, currently complex128 FITS files are not supported by astropy, hence the need to split complex images
    onto two real image products, we present the user with two options to carry out this split.

        .. rubric:: Available complex splitting possibilities:
        - *cartesian*: Split is done to a real part and an imaginary part FITS files
        - *polar*:     Split is done to an amplitude and a phase FITS files


    The FITS produced by this function have been tested are known to work with CARTA and DS9
    """

    logger = _get_astrohack_logger()
    parm_dict = {'filename': mds_name,
                 'ant_name': ant_name,
                 'ddi': ddi,
                 'destination': destination,
                 'complex_split': complex_split,
                 'parallel': parallel}

    parms_passed = _check_parms(parm_dict, 'filename', [str], default=None)
    parms_passed = parms_passed and _check_parms(parm_dict, 'complex_split', [str],
                                                 acceptable_data=['cartesian', 'polar'], default="cartesian")
    parms_passed = parms_passed and _check_parms(parm_dict, 'ant_name', [list], list_acceptable_data_types=[str],
                                                 default='all')
    parms_passed = parms_passed and _check_parms(parm_dict, 'ddi', [list], list_acceptable_data_types=[str],
                                                 default='all')
    parms_passed = parms_passed and _check_parms(parm_dict, 'destination', [str], default=None)
    parms_passed = parms_passed and _check_parms(parm_dict, 'parallel', [bool], default=True)

    if not parms_passed:
        logger.error("export_screws parameter checking failed.")
        raise Exception("export_screws parameter checking failed.")

    check_if_file_exists(mds_name)
    mds_origin = _check_mds_origin(mds_name, ['image', 'panel', 'point', 'holog'])

    if mds_origin in ['combine', 'holog']:
        chunk_function = _export_to_fits_holog_chunk
        holog_mds = AstrohackImageFile(mds_name)
        holog_mds.open()
        parm_dict['holog_mds'] = holog_mds
    elif mds_origin == 'panel':
        chunk_function = _export_to_fits_panel_chunk
        panel_mds = AstrohackPanelFile(mds_name)
        panel_mds.open()
        parm_dict['panel_mds'] = panel_mds
    else:
        logger.error(f"Cannot export mds_files created by {mds_origin} to FITS")
        return

    try:
        os.mkdir(parm_dict['destination'])
    except FileExistsError:
        logger.warning('Destination folder already exists, results may be overwritten')

    _generate_antenna_ddi_graph_and_compute('export_to_fits', chunk_function, parm_dict, parallel)
