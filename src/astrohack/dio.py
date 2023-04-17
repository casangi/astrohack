import os

from astrohack._utils._constants import length_units
from astrohack._utils._dio_classes import AstrohackPanelFile
from astrohack._utils._parm_utils._check_parms import _check_parms
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._utils import _parm_to_list


def export_screws(panel_mds_name, destination, ant_name=None, ddi=None,  unit='mm'):
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
        logger.warning('Destination folder already exists, results my be overwritten')

    antennae = _parm_to_list(parm_dict['ant_name'], parm_dict['filename'])
    for antenna in antennae:
        ddis = _parm_to_list(parm_dict['ddi'], parm_dict['filename']+'/'+antenna)
        for ddi in ddis:
            export_name = parm_dict['destination']+f'/{antenna}_{ddi}_screws.txt'
            surface = panel_mds.get_antenna(antenna, ddi)
            surface.export_screws(export_name, unit=unit)







