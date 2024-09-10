import copy
import inspect
import json
from datetime import date

import toolviper.utils.logger as logger
import numpy as np

import astrohack

from astrohack.utils import compute_average_stokes_visibilities
from astrohack.utils.text import NumpyEncoder


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


