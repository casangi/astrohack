from astrohack.antenna.antenna_surface import SUPPORTED_POL_STATES
from astrohack.antenna.panel_fitting import PANEL_MODEL_DICT
from astrohack.utils import trigo_units, length_units, time_units, freq_units
from astrohack.utils import possible_splits
from astrohack.visualization.plot_tools import astrohack_cmaps


def custom_plots_checker(allowed_type):
    if allowed_type == 'colormaps':
        return astrohack_cmaps
    elif 'split' in allowed_type:
        return custom_split_checker(allowed_type)
    elif 'units' in allowed_type:
        return custom_unit_checker(allowed_type)
    else:
        return "Not found"


def custom_unit_checker(unit_type):
    if unit_type == "units.trig":
        return trigo_units

    elif unit_type == "units.length":
        return length_units

    elif unit_type == "units.time":
        return time_units

    elif unit_type == "units.frequency":
        return freq_units

    else:
        return "Not found"


def custom_split_checker(split_type):
    if split_type == 'split.complex':
        return possible_splits
    else:
        return "Not found"


def custom_panel_checker(check_type):
    if check_type == "panel.models":
        return PANEL_MODEL_DICT.keys()
    elif check_type == "panel.pol_states":
        return SUPPORTED_POL_STATES
    else:
        return "Not found"
