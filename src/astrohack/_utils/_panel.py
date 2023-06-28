import xarray as xr

from astrohack._utils._constants import plot_types

from astrohack._utils._panel_classes.telescope import Telescope
from astrohack._utils._panel_classes.antenna_surface import AntennaSurface
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

# global constants


def _get_correct_telescope_from_name(xds):
    if xds.attrs['telescope_name'] == "ALMA":
        tname = xds.attrs['telescope_name']+'_'+xds.attrs['ant_name'][0:2]
        telescope = Telescope(tname)
    elif xds.attrs['telescope_name'] == "EVLA":
        tname = "VLA"
        telescope = Telescope(tname)
    else:
        raise ValueError('Unsuported telescope {0:s}'.format(xds.attrs['telescope_name']))
    return telescope


def _panel_chunk(panel_chunk_params):
    """
    Process a chunk of the holographies, usually a chunk consists of an antenna over a ddi
    Args:
        panel_chunk_params: dictionary of inputs
    """
    logger = _get_astrohack_logger()
    if panel_chunk_params['origin'] == 'AIPS':
        inputxds = xr.open_zarr(panel_chunk_params['image_name'])
        telescope = Telescope(inputxds.attrs['telescope_name'])
        antenna = inputxds.attrs['ant_name']
        ddi = 0

    else:
        ddi = panel_chunk_params['this_ddi']
        antenna = panel_chunk_params['this_ant']
        inputxds = panel_chunk_params['xds_data']

        logger.info(f'[panel]: processing {antenna} {ddi}')
        inputxds.attrs['AIPS'] = False

        telescope = _get_correct_telescope_from_name(inputxds)

    surface = AntennaSurface(inputxds, telescope, panel_chunk_params['cutoff'], panel_chunk_params['panel_kind'],
                             panel_margins=panel_chunk_params['panel_margins'])

    surface.compile_panel_points()
    surface.fit_surface()
    surface.correct_surface()
    
    xds_name = panel_chunk_params['panel_name'] + f'/{antenna}/{ddi}'
    xds = surface.export_xds()
    xds.to_zarr(xds_name, mode='w')


def _plot_antenna_chunk(parm_dict):
    """
    Chunk function for the user facing function plot_antenna
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    plot_type = parm_dict['plot_type']
    basename = f'{destination}/{antenna}_{ddi}'
    xds = parm_dict['xds_data']
    telescope = Telescope(xds.attrs['telescope_name'])
    surface = AntennaSurface(xds, telescope, reread=True)
    if plot_type == plot_types[0]:  # deviation plot
        surface.plot_deviation(basename, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'], unit=parm_dict['unit'],
                               colormap=parm_dict['colormap'], figuresize=parm_dict['figuresize'], caller='panel',
                               display=parm_dict['display'])
    elif plot_type == plot_types[1]:  # phase plot
        surface.plot_phase(basename, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'], unit=parm_dict['unit'],
                           colormap=parm_dict['colormap'], figuresize=parm_dict['figuresize'], caller='panel',
                           display=parm_dict['display'])
    elif plot_type == plot_types[2]:  # Ancillary plot
        surface.plot_mask(basename=basename, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'],
                          colormap=parm_dict['colormap'], figuresize=parm_dict['figuresize'], caller='panel',
                          display=parm_dict['display'])
        surface.plot_amplitude(basename=basename, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'],
                               colormap=parm_dict['colormap'], figuresize=parm_dict['figuresize'], caller='panel',
                               display=parm_dict['display'])
    else:  # all plots
        surface.plot_deviation(basename, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'], unit=parm_dict['unit'],
                               colormap=parm_dict['colormap'], figuresize=parm_dict['figuresize'], caller='panel',
                               display=parm_dict['display'])
        surface.plot_phase(basename, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'], unit='deg',
                           colormap=parm_dict['colormap'], figuresize=parm_dict['figuresize'], caller='panel',
                           display=parm_dict['display'])
        surface.plot_mask(basename=basename, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'],
                          colormap=parm_dict['colormap'], figuresize=parm_dict['figuresize'], caller='panel',
                          display=parm_dict['display'])
        surface.plot_amplitude(basename=basename, screws=parm_dict['plot_screws'], dpi=parm_dict['dpi'],
                               colormap=parm_dict['colormap'], figuresize=parm_dict['figuresize'], caller='panel',
                               display=parm_dict['display'])


def _export_to_fits_panel_chunk(parm_dict):
    """
    Panel side chunk function for the user facing function export_to_fits
    Args:
        parm_dict: parameter dictionary
    """
    logger = _get_astrohack_logger()
    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    logger.info(f'[export_to_fits]: Exporting panel contents of {antenna} {ddi} to FITS files in {destination}')
    xds = parm_dict['xds_data']
    telescope = Telescope(xds.attrs['telescope_name'])
    surface = AntennaSurface(xds, telescope, reread=True)
    basename = f'{destination}/{antenna}_{ddi}'
    surface.export_to_fits(basename)
    return


def _export_screws_chunk(parm_dict):
    """
    Chunk function for the user facing function export_screws
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    export_name = parm_dict['destination'] + f'/panel_screws_{antenna}_{ddi}.'
    xds = parm_dict['xds_data']
    telescope = Telescope(xds.attrs['telescope_name'])
    surface = AntennaSurface(xds, telescope, reread=True)
    surface.export_screws(export_name + 'txt', unit=parm_dict['unit'])
    surface.plot_screw_adjustments(export_name + 'png', unit=parm_dict['unit'], threshold=parm_dict['threshold'],
                                   colormap=parm_dict['colormap'], figuresize=parm_dict['figuresize'],
                                   dpi=parm_dict['dpi'], display=parm_dict['display'])
