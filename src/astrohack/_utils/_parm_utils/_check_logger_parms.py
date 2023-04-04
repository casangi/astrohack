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

#ducting - code is complex and might fail after some time if parameters is wrong. Sensable values are also checked. Gives printout of all wrong parameters.

import numpy as np
from  ._check_parms import _check_parms


def _check_logger_parms(logger_parms):
    import numbers
    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)
    
    if not(_check_parms(logger_parms, 'log_to_term', [bool],default=True)): parms_passed = False
    if not(_check_parms(logger_parms, 'log_to_file', [bool],default=False)): parms_passed = False
    if not(_check_parms(logger_parms, 'log_file', [str],default='hack_')): parms_passed = False
    if not(_check_parms(logger_parms, 'log_level', [str],default='INFO',acceptable_data=['DEBUG','INFO','WARNING','ERROR'])): parms_passed = False
    
    return parms_passed


def _check_worker_logger_parms(logger_parms):
    import numbers
    parms_passed = True
    arc_sec_to_rad = np.pi / (3600 * 180)
    
    if not(_check_parms(logger_parms, 'log_to_term', [bool],default=False)): parms_passed = False
    if not(_check_parms(logger_parms, 'log_to_file', [bool],default=False)): parms_passed = False
    if not(_check_parms(logger_parms, 'log_file', [str],default='hack_')): parms_passed = False
    if not(_check_parms(logger_parms, 'log_level', [str],default='INFO',acceptable_data=['DEBUG','INFO','WARNING','ERROR'])): parms_passed = False
    
    return parms_passed
