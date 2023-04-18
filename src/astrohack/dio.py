import os
import dask

from astrohack._utils._constants import length_units, trigo_units
from astrohack._utils._dio_classes import AstrohackPanelFile
from astrohack._utils._parm_utils._check_parms import _check_parms
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._utils import _parm_to_list


plot_types = ['deviation', 'phase', 'ancillary']


def export_screws(panel_mds_name, destination, ant_name=None, ddi=None,  unit='mm'):
    """Export screws adjustments from a panel_mds file

    :param panel_mds_name: Input panel_mds file
    :type panel_mds_name: str
    :param destination: Name of the destination folder to contain exported screw adjustments
    :type destination: str
    :param ant_name: List of antennae/antenna to be exported, defaults to "all" when None
    :type ant_name: list or str, optional
    :param ddi: List of ddis/ddi to be exported, defaults to "all" when None
    :type ddi: list or str, optional
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
        ddis = _parm_to_list(parm_dict['ddi'], parm_dict['filename']+'/'+antenna)
        for ddi in ddis:
            export_name = parm_dict['destination']+f'/screws_{antenna}_{ddi}.txt'
            surface = panel_mds.get_antenna(antenna, ddi)
            surface.export_screws(export_name, unit=unit)


def plot_antenna(panel_mds_name, destination, ant_name=None, ddi=None, plot_type='deviation', plot_screws=False,
                 dpi=300, unit=None, parallel=True):
    """Plot Antenna surfaces from a panel_mds file

    :param panel_mds_name: Input panel_mds file
    :type panel_mds_name: str
    :param destination: Name of the destination folder to contain plots
    :type destination: str
    :param ant_name: List of antennae/antenna to be plotted, defaults to "all" when None
    :type ant_name: list or str, optional
    :param ddi: List of ddis/ddi to be plotted, defaults to "all" when None
    :type ddi: list or str, optional
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
        parm_dict['this_antenna'] = antenna
        ddis = _parm_to_list(parm_dict['ddi'], parm_dict['filename']+'/'+antenna)
        for ddi in ddis:
            parm_dict['this_ddi'] = ddi
            if parallel:
                delayed_list.append(dask.delayed(_plot_antenna_chunk)(dask.delayed(parm_dict)))
            else:
                _plot_antenna_chunk(parm_dict)

    if parallel:
        dask.compute(delayed_list)


def _plot_antenna_chunk(parm_dict):
    antenna = parm_dict['this_antenna']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    plot_type = parm_dict['plot_type']
    plot_name = f'{destination}/{plot_type}_{antenna}_{ddi}.png'

    surface = parm_dict['panel_mds'].get_antenna(antenna, ddi)
    if plot_type == plot_types[0]:  # deviation plot
        surface.plot_surface(filename=plot_name, mask=False, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'],
                             plotphase=False, unit=parm_dict['unit'])
    elif plot_type == plot_types[1]:  # phase plot
        surface.plot_surface(filename=plot_name, mask=False, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'],
                             plotphase=True, unit=parm_dict['unit'])
    else:  # Ancillary plot
        surface.plot_surface(filename=plot_name, mask=True, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'],
                             plotphase=False, unit=None)


