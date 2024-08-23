from graphviper.utils import logger as logger
from astrohack.antenna import Telescope, AntennaSurface


def export_to_fits_panel_chunk(parm_dict):
    """
    Panel side chunk function for the user facing function export_to_fits
    Args:
        parm_dict: parameter dictionary
    """

    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    logger.info(f'Exporting panel contents of {antenna} {ddi} to FITS files in {destination}')
    xds = parm_dict['xds_data']
    telescope = Telescope(xds.attrs['telescope_name'])
    surface = AntennaSurface(xds, telescope, reread=True)
    basename = f'{destination}/{antenna}_{ddi}'
    surface.export_to_fits(basename)
    return
