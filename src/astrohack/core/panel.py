import graphviper.utils.logger as logger
import xarray as xr
from prettytable import PrettyTable

from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.antenna.telescope import Telescope
from astrohack.utils import clight
from astrohack.utils.text import param_to_list, format_label, format_frequency, format_wavelength


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
        logger.info(f'processing {antenna} {ddi}')
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


def export_gains_table(data_dict, parm_dict):
    wavelengths = [0.20, 0.13, 0.06, 0.03, 0.02, 0.013, 0.01, 0.007]

    field_names = ['Frequency', 'Wavelength', 'Before panel (dB)', 'After panel (dB)', 'Theoretical Max. (dB)']

    antenna_sel = param_to_list(parm_dict['ant'], data_dict, 'ant')
    for ant in antenna_sel:
        ddi_sel = param_to_list(parm_dict['ddi'], data_dict[ant], 'ddi')
        for ddi in ddi_sel:
            xds = data_dict[ant][ddi]
            telescope = Telescope.from_xds(xds)
            antenna = AntennaSurface(xds, telescope, reread=True)
            frequency = clight/antenna.wavelength

            outstr = f'# Gain estimates for {telescope.name} antenna {ant.split("_")[1]}\n'
            outstr += f'# Based on a measurement at {format_frequency(frequency)}, {format_wavelength(antenna.wavelength)}'
            outstr += 3*'\n'
            table = PrettyTable()
            table.field_names = field_names
            table.align = 'c'

            for wavelength in wavelengths:
                prior, theo = antenna.gain_at_wavelength(False, wavelength)
                after, _  = antenna.gain_at_wavelength(True, wavelength)
                row = [format_frequency(clight/wavelength), format_wavelength(wavelength), f'{prior:.2f}', f'{after:.2f}',
                       f'{theo:.2f}']
                table.add_row(row)

            outstr += table.get_string()
            outname = parm_dict['destination'] + f'/panel_gains_{ant}_{ddi}.txt'
            outfile = open(outname, 'w')
            outfile.write(outstr + '\n')
            outfile.close()

    return None


