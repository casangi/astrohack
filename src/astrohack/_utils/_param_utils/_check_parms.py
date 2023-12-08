#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger


def _check_parm(parm, parm_name, acceptable_data_types, acceptable_data=None, acceptable_range=None,
                list_acceptable_data_types=None, list_len=None, default=None):
    parm_dict = {parm_name:parm}
    parm_passed = _check_parms(parm_dict, parm_name, acceptable_data_types, acceptable_data, acceptable_range,
                               list_acceptable_data_types, list_len, default)
    
    parm = parm_dict[parm_name]
    
    return parm_passed, parm


def _check_parms(parm_dict, string_key, acceptable_data_types, acceptable_data=None, acceptable_range=None,
                 list_acceptable_data_types=None, list_len=None, default=None, log_default_setting=True):
    """

    Parameters
    ----------
    parm_dict: dict
        The dictionary in which the parameter will be checked
    string_key :
    acceptable_data_types : list
    acceptable_data : list
    acceptable_range : list (length of 2)
    list_acceptable_data_types : list
    list_len : int
        If list_len is -1 than the list can be any length.
    default :
    Returns
    -------
    parm_passed : bool
        
    """
    logger = _get_astrohack_logger()

    if (string_key in parm_dict) and (parm_dict[string_key] is not None):
        parm_type = type(parm_dict[string_key])
        if parm_type in acceptable_data_types:
            if (parm_type is list) or (parm_type is np.ndarray):
                if (len(parm_dict[string_key]) != list_len) and (list_len is not None):
                    logger.error(f'Parameter {str(string_key)} must be a list of '
                                 f'{str(list_acceptable_data_types)} and length {str(list_len)}. Wrong length.')
                    return False
                else:
                    list_len = len(parm_dict[string_key])
                    for i in range(list_len):
                        type_check = False
                        for lt in list_acceptable_data_types:
                            if isinstance(parm_dict[string_key][i], lt):
                                type_check = True
                        if not type_check:
                            logger.error(f'Parameter {str(string_key)} must be a list of '
                                         f'{str(list_acceptable_data_types)} and length {str(list_len)}. Wrong type of '
                                         f'{str(type(parm_dict[string_key][i]))}.')
                            return False

                        if acceptable_data is not None:
                            if not(parm_dict[string_key][i] in acceptable_data):
                                logger.error(f'Invalid {str(string_key)}. Can only be one of '
                                             f'{+str(acceptable_data)}.')
                                return False

                        if acceptable_range is not None:
                            if (parm_dict[string_key][i] < acceptable_range[0]) or (parm_dict[string_key][i] >
                                                                                    acceptable_range[1]):
                                logger.error(f'Invalid {str(string_key)}. Must be within the range '
                                             f'{str(acceptable_range)}.')
                                return False
            elif parm_type is dict:
                parms_passed = True

                if default is None:
                    logger.error(f'Dictionary parameters must have a default. Please report bug.')
                    return False
                for default_element in default:
                    if default_element in parm_dict[string_key]:
                        if not(_check_parm(parm_dict[string_key], default_element, [type(default[default_element])],
                                           default=default[default_element])):
                            parms_passed = False
                    else:
                        parm_dict[string_key][default_element] = default[default_element]

                        if log_default_setting:
                            logger.info(f'Setting default {str(string_key)} [\'{str(default_element)}\'] to '
                                        f'{str(default[default_element])}.')

                return parms_passed

            else:

                type_check = False
                for adt in acceptable_data_types:
                    if isinstance(parm_dict[string_key], adt):
                        type_check = True
                if not type_check:
                    logger.error(f'Parameter {str(string_key)} must be of type {str(acceptable_data_types)}.')
                    return False

                if acceptable_data is not None:
                    if not(parm_dict[string_key] in acceptable_data):
                        logger.error(f'Invalid {str(string_key)}. Can only be one of {str(acceptable_data)}.')
                        return False

                if acceptable_range is not None:
                    if (parm_dict[string_key] < acceptable_range[0]) or (parm_dict[string_key] > acceptable_range[1]):
                        logger.error(f'Invalid {str(string_key)}. Must be within the range '
                                     f'{str(acceptable_range)}.')
                        return False
        else:
            logger.error(f'Parameter {str(string_key)} is not of correct type.')
            return False
    else:
        if default is not None:
            parm_dict[string_key] = default
            
            if log_default_setting:
                logger.info(f'Setting default {str(string_key)} to {str(parm_dict[string_key])}.')
        else:
            logger.error(f'Parameter {str(string_key)} must be specified.')
            return False
    
    return True


def _parm_check_passed(caller, parms_passed):
    logger = _get_astrohack_logger()
    if not parms_passed:
        logger.error(f"{caller} parameter checking failed.")
        raise Exception(f"{caller} parameter checking failed.")
    

    
