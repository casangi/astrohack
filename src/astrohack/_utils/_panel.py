import xarray as xr
import graphviper.utils.logger as logger

from astrohack._utils._constants import plot_types

from astrohack._utils._panel_classes.telescope import Telescope
from astrohack._utils._panel_classes.antenna_surface import AntennaSurface, SUPPORTED_POL_STATES
from astrohack._utils._panel_classes.base_panel import PANEL_MODELS


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
    
    if panel_chunk_params['origin'] == 'AIPS':
        inputxds = xr.open_zarr(panel_chunk_params['image_name'])
        telescope = Telescope(inputxds.attrs['telescope_name'])
        antenna = inputxds.attrs['ant_name']
        ddi = 0
    else:
        ddi = panel_chunk_params['this_ddi']
        antenna = panel_chunk_params['this_ant']
        inputxds = panel_chunk_params['xds_data']
        logger.info(f'processing {antenna} {ddi}')
        inputxds.attrs['AIPS'] = False
        telescope = _get_correct_telescope_from_name(inputxds)

    surface = AntennaSurface(inputxds, telescope, clip_type=panel_chunk_params['clip_type'],
                             pol_state=panel_chunk_params['polarization_state'],
                             clip_level=panel_chunk_params['clip_level'], pmodel=panel_chunk_params['panel_model'],
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
        surface.plot_deviation(basename, 'panel', parm_dict)
    elif plot_type == plot_types[1]:  # phase plot
        surface.plot_phase(basename, 'panel', parm_dict)
    elif plot_type == plot_types[2]:  # Ancillary plot
        surface.plot_mask(basename,  'panel', parm_dict)
        surface.plot_amplitude(basename,  'panel', parm_dict)
    else:  # all plots
        surface.plot_deviation(basename, 'panel', parm_dict)
        surface.plot_phase(basename, 'panel', parm_dict)
        surface.plot_mask(basename,  'panel', parm_dict)
        surface.plot_amplitude(basename,  'panel', parm_dict)


def _export_to_fits_panel_chunk(parm_dict):
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
    surface.plot_screw_adjustments(export_name + 'png', parm_dict)


def custom_panel_checker(check_type):
    if check_type == "panel.models":
        return PANEL_MODELS
    elif check_type == "panel.pol_states":
        return SUPPORTED_POL_STATES
    else:
        return "Not found"

