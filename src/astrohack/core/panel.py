import toolviper.utils.logger as logger
import xarray as xr

from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.antenna.telescope import Telescope
from astrohack.utils import create_dataset_label


def process_panel_chunk(panel_chunk_params):
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
        logger.info(f'processing {create_dataset_label(antenna, ddi)}')
        inputxds.attrs['AIPS'] = False
        telescope = Telescope.from_xds(inputxds)

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


