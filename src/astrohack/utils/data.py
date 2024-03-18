import copy
import inspect
import json
from datetime import date

import graphviper.utils.logger as logger
import numpy as np
from prettytable import PrettyTable

import astrohack

from astrohack.antenna.telescope import Telescope
from astrohack.antenna.antenna_surface import AntennaSurface
from astrohack.utils import compute_average_stokes_visibilities, convert_unit, clight, notavail, rotate_to_gmt
from astrohack.utils.text import NumpyEncoder, param_to_list, add_prefix, format_value_error


def read_meta_data(file_name):
    """Reads dimensional data from holog meta file.

        Args:
            file_name (str): astrohack file name.

        Returns:
            dict: dictionary containing dimension data.
        """

    try:
        with open(file_name) as json_file:
            json_dict = json.load(json_file)

    except Exception as error:
        logger.error(str(error))
        raise Exception

    return json_dict


def write_meta_data(file_name, input_dict):
    """
        Creates a metadata dictionary that is compatible with JSON and writes it to a file
        Args:
            file_name: Output json file name
            input_dict: Dictionary to be included in the metadata
        """

    calling_function = 1

    meta_data = copy.deepcopy(input_dict)

    meta_data.update({
        'version': astrohack.__version__,
        'origin': inspect.stack()[calling_function].function
    })

    try:
        with open(file_name, "w") as json_file:
            json.dump(meta_data, json_file, cls=NumpyEncoder)

    except Exception as error:
        logger.error(f'{error}')


def export_to_aips(param_dict):
    xds_data = param_dict['xds_data']
    stokes = 'I'
    stokes_vis = compute_average_stokes_visibilities(xds_data, stokes)
    filename = f'{param_dict["destination"]}/holog_visibilities_{param_dict["this_map"]}_{param_dict["this_ant"]}_' \
               f'{param_dict["this_ddi"]}.txt'
    ant_num = xds_data.attrs['antenna_name'].split('a')[1]
    cmt = '#! '

    today = date.today().strftime("%y%m%d")
    outstr = cmt + f"RefAnt = ** Antenna = {ant_num} Stokes = '{stokes}_' Freq =  {stokes_vis.attrs['frequency']:.9f}" \
                   f" DATE-OBS = '{today}'\n"
    outstr += cmt + "MINsamp =   0  Npoint =   1\n"
    outstr += cmt + "IFnumber =   2   Channel =    32.0\n"
    outstr += cmt + "TimeRange = -99,  0,  0,  0,  999,  0,  0,  0\n"
    outstr += cmt + "Averaged Ref-Ants = 10, 15,\n"
    outstr += cmt + "DOCAL = T  DOPOL =-1\n"
    outstr += cmt + "BCHAN=     4 ECHAN=    60 CHINC=  1 averaged\n"
    outstr += cmt + "   LL             MM             AMPLITUDE      PHASE         SIGMA(AMP)   SIGMA(PHASE)\n"
    lm = xds_data['DIRECTIONAL_COSINES'].values
    amp = stokes_vis['AMPLITUDE'].values
    pha = stokes_vis['PHASE'].values
    sigma_amp = stokes_vis['SIGMA_AMP']
    sigma_pha = stokes_vis['SIGMA_PHA']
    for i_time in range(len(xds_data.time)):
        if np.isfinite(sigma_amp[i_time]):
            outstr += f"{lm[i_time, 0]:15.7f}{lm[i_time, 1]:15.7f}{amp[i_time]:15.7f}{pha[i_time]:15.7f}" \
                      f"{sigma_amp[i_time]:15.7f}{sigma_pha[i_time]:15.7f}\n"
    outstr += f"{cmt}Average number samples per point =   1.000"

    with open(filename, 'w') as outfile:
        outfile.write(outstr)

    return


def export_locit_fit_results(data_dict, parm_dict):
    """
    Export fit results to a txt file listing the different DDIs as different solutions if data is not combined
    Args:
        data_dict: the mds content
        parm_dict: Dictionary of the parameters given to the calling function

    Returns:
    text file with fit results in convenient units
    """
    pos_unit = parm_dict['position_unit']
    del_unit = parm_dict['delay_unit']
    len_fact = convert_unit('m', pos_unit, 'length')
    del_fact = convert_unit('sec', del_unit, kind='time')
    pos_fact = len_fact * clight
    combined = parm_dict['combined']

    if combined:
        field_names = ['Antenna', f'RMS [{del_unit}]', f'F. delay [{del_unit}]', f'X offset [{pos_unit}]',
                       f'Y offset [{pos_unit}]', f'Z offset [{pos_unit}]']
        specifier = 'combined_' + data_dict._meta_data['combine_ddis']

    else:
        field_names = ['Antenna', 'DDI', f'RMS [{del_unit}]', f'F. delay [{del_unit}]', f'X offset [{pos_unit}]',
                       f'Y offset [{pos_unit}]', f'Z offset [{pos_unit}]']
        specifier = 'separated_ddis'
    kterm_present = data_dict._meta_data["fit_kterm"]
    rate_present = data_dict._meta_data['fit_delay_rate']
    if kterm_present:
        field_names.extend([f'K offset [{pos_unit}]'])
    if rate_present:
        tim_unit = parm_dict['time_unit']
        slo_unit = f'{del_unit}/{tim_unit}'
        slo_fact = del_fact / convert_unit('day', tim_unit, 'time')
        field_names.extend([f'Rate [{slo_unit}]'])
    else:
        slo_unit = notavail
        slo_fact = 1.0

    table = PrettyTable()
    table.field_names = field_names
    table.align = 'c'
    full_antenna_list = Telescope(data_dict._meta_data['telescope_name']).ant_list
    selected_antenna_list = param_to_list(parm_dict['ant'], data_dict, 'ant')

    for ant_name in full_antenna_list:
        ant_key = add_prefix(ant_name, 'ant')
        row = [ant_name]
        if ant_key in selected_antenna_list:
            if ant_key in data_dict.keys():
                if ant_name == data_dict._meta_data['reference_antenna']:
                    ant_name += ' (ref)'

                antenna = data_dict[ant_key]
                if combined:
                    table.add_row(_export_locit_xds(row, antenna.attrs, del_fact, pos_fact, slo_fact, kterm_present,
                                                    rate_present))
                else:
                    ddi_list = param_to_list(parm_dict['ddi'], data_dict[ant_key], 'ddi')
                    for ddi_key in ddi_list:
                        row = [ant_name, ddi_key.split('_')[1]]
                        table.add_row(
                            _export_locit_xds(row, data_dict[ant_key][ddi_key].attrs, del_fact, pos_fact, slo_fact,
                                              kterm_present, rate_present))

    outname = parm_dict['destination'] + f'/position_{specifier}_fit_results.txt'
    outfile = open(outname, 'w')
    outfile.write(table.get_string() + '\n')
    outfile.close()


def _export_locit_xds(row, attributes, del_fact, pos_fact, slo_fact, kterm_present, rate_present):
    """
    Export the data from a single X array DataSet attributes to a table row (a list)
    Args:
        row: row onto which the data results are to be added
        attributes: The XDS attributes dictionary
        del_fact: Delay unit scaling factor
        pos_fact: Position unit scaling factor
        slo_fact: Delay rate unit scaling factor
        kterm_present: Is the elevation axis offset term present?
        rate_present: Is the delay rate term present?

    Returns:
    The filled table row
    """
    tolerance = 1e-4

    rms = np.sqrt(attributes["chi_squared"]) * del_fact
    row.append(f'{rms:.2e}')
    row.append(format_value_error(attributes['fixed_delay_fit'], attributes['fixed_delay_error'], del_fact,
                                  tolerance))
    position, poserr = rotate_to_gmt(np.copy(attributes['position_fit']), attributes['position_error'],
                                     attributes['antenna_info']['longitude'])
    for i_pos in range(3):
        row.append(format_value_error(position[i_pos], poserr[i_pos], pos_fact, tolerance))
    if kterm_present:
        row.append(format_value_error(attributes['koff_fit'], attributes['koff_error'], pos_fact, tolerance))
    if rate_present:
        row.append(format_value_error(attributes['rate_fit'], attributes['rate_error'], slo_fact, tolerance))
    return row


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


def export_screws_chunk(parm_dict):
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
