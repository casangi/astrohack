import json
import inspect
import textwrap

import numpy as np
import graphviper.utils.logger as logger

from prettytable import PrettyTable
from astrohack.utils.conversion import convert_unit
from astrohack.utils.algorithms import significant_figures_round


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, type(None)):
            return "None"

        return json.JSONEncoder.default(self, obj)


def _reshape(array, columns):
    size = len(array)
    rows = int(size / columns)
    if rows <= 0:
        return 1, 0
    else:
        remainder = size - (rows * columns)

        return rows, remainder


def print_array(array, columns, indent=4):
    rows, remainder = _reshape(array, columns)

    if columns > len(array):
        columns = len(array)

    str_line = ""

    for i in range(rows):
        temp = []
        for j in range(columns):
            k = columns * i + j
            if j == 0:
                temp.append("{:>3}".format(array[k]).rjust(indent, ' '))
            else:
                temp.append("{:>3}".format(array[k]))

        str_line += ", ".join(temp)
        str_line += "\n"

    temp = []
    if remainder > 0:
        for i in range(remainder):
            index = columns * rows + i

            if i == 0:
                temp.append("{:>3}".format(array[index]).rjust(indent, ' '))
            else:
                temp.append("{:>3}".format(array[index]))

        str_line += ", ".join(temp)

    print(str_line)


def approve_prefix(key):
    approved_prefix = ["ant_", "map_", "ddi_"]

    for prefix in approved_prefix:
        if key.startswith(prefix):
            return True

    logger.warning(f"File meta data contains and unknown key ({key}), the file may not complete properly.")

    return False


def add_prefix(input_string, prefix):
    """
    Adds a prefix to a string filename, if the filename is a path with /, adds the prefix to the actual filename at the
    end of the path
    Args:
        input_string: filename or file path
        prefix: prefix to be added to the filename

    Returns: filename or path plus prefix added to the filename

    """
    wrds = input_string.split('/')
    wrds[-1] = prefix + '_' + wrds[-1]
    return '/'.join(wrds)


def print_holog_obs_dict(holog_obj):
    OPEN_DICT = ":{"
    CLOSE_DICT = "}"

    OPEN_LIST = ":["
    CLOSE_LIST = "]"

    logger.info("| ********************************************************** |")
    logger.info("|                 HOLOG OBSERVATION DICTIONARY               |")
    logger.info("| ********************************************************** |\n\n")

    for ddi_key, ddi_value in holog_obj.items():
        logger.info("{ddi_key} {open_bracket}".format(ddi_key=ddi_key, open_bracket=OPEN_DICT))
        for map_key, map_value in holog_obj[ddi_key].items():
            logger.info("{map_key: >10} {open_bracket}".format(map_key=map_key, open_bracket=OPEN_DICT))
            for attr_key, attr_value in holog_obj[ddi_key][map_key].items():
                if "scans" in attr_key:
                    logger.info("{attr_key: >12} {open_list}".format(attr_key=attr_key, open_list=OPEN_LIST))

                    scan_list = ", ".join(list(map(str, holog_obj[ddi_key][map_key][attr_key])))

                    # The print justification in notebook is weird on this and seems to move according to list length

                    logger.info("{scan: >18}".format(scan=scan_list))
                    logger.info("{close_bracket: >10}".format(close_bracket=CLOSE_LIST))

                elif "ant" in attr_key:
                    logger.info("{attr_key: >12} {open_bracket}".format(attr_key=attr_key, open_bracket=OPEN_DICT))
                    for ant_key, ant_value in holog_obj[ddi_key][map_key][attr_key].items():
                        logger.info("{ant_key: >18} {open_list}".format(ant_key=ant_key, open_list=OPEN_LIST))
                        logger.info("{antenna: >25}".format(antenna=", ".join(ant_value)))
                        logger.info("{close_list: >15}".format(close_list=CLOSE_LIST))

                    logger.info("{close_bracket: >10}".format(close_bracket=CLOSE_DICT))

                else:
                    pass
        logger.info("{close_bracket: >5}".format(close_bracket=CLOSE_DICT))

    logger.info("{close_bracket}".format(close_bracket=CLOSE_DICT))


def param_to_list(param, data_dict, prefix):
    """
    Transforms a string parameter to a list if parameter is all or a single string
    Args:
        param: string or list parameter
        data_dict: Dictionary in which to search for data to be listed
        prefix: prefix to be added to parameter

    Returns: parameter converted to a list

    """

    if param == 'all':
        out_list = list(data_dict.keys())

    elif isinstance(param, str):
        out_list = [add_prefix(param, prefix)]

    elif isinstance(param, int):
        out_list = [f'{prefix}_{param}']

    elif isinstance(param, (list, tuple)):
        out_list = []
        for item in param:
            if isinstance(item, str):
                out_list.append(add_prefix(item, prefix))
            elif isinstance(item, int):
                out_list.append(f'{prefix}_{item}')
            else:
                msg = f'Cannot interpret parameter {item} of type {type(item)}'
                logger.error(msg)
                raise Exception(msg)
    else:
        msg = f'Cannot interpret parameter {param} of type {type(param)}'
        logger.error(msg)
        raise Exception(msg)

    return out_list


def get_default_file_name(input_file: str, output_type: str) -> str:
    known_data_types = [".ms", ".holog.zarr", ".image.zarr", ".locit.zarr", ".combine.zarr", ".position.zarr"]

    output_file = None

    for suffix in known_data_types:
        if input_file.endswith(suffix):
            base_name = input_file.rstrip(suffix)
            output_file = "".join((base_name, output_type))

    if not output_file:
        output_file = "".join((input_file, output_type))

    logger.info("Creating output file name: {}".format(output_file))

    return output_file


def print_data_contents(data_dict, field_names, alignment='l'):
    """
    Factorized printing of the prettytable with the data contents
    Args:
        data_dict: Dictionary with data to be displayed
        field_names: Field names in the table
        alignment: Contents of the table to be aligned Left or Right
    """
    table = PrettyTable()
    table.field_names = field_names
    table.align = alignment
    depth = len(field_names)
    if depth == 3:
        for item_l1 in data_dict.keys():
            for item_l2 in data_dict[item_l1].keys():
                table.add_row([item_l1, item_l2, list(data_dict[item_l1][item_l2].keys())])
    elif depth == 2:
        for item_l1 in data_dict.keys():
            if 'info' in item_l1:
                pass
            else:
                table.add_row([item_l1, list(data_dict[item_l1].keys())])
    elif depth == 1:
        for item_l1 in data_dict.keys():
            table.add_row([item_l1])
    else:
        raise Exception(f'Unhandled case len(field_names) == {depth}')

    print('\nContents:')
    print(table)


def print_dict_table(input_parameters, split_key=None, alignment='l', heading="Input Parameters"):
    """
    Print a summary of the attributes
    Args:
        input_parameters: Dictionary containing metadata attributes
        split_key: key to be sqrt and displayed as nx X ny

    Returns:

    """
    print(f"\n{heading}:")
    table = PrettyTable()
    table.field_names = ['Parameter', 'Value']
    table.align = alignment

    for key, item in input_parameters.items():
        if key == split_key:
            n_side = int(np.sqrt(input_parameters[key]))
            table.add_row([key, f'{n_side:d} x {n_side:d}'])
        if isinstance(item, dict):
            table.add_row([key, _dict_to_key_list(item)])
        else:
            table.add_row([key, item])
    print(table)


def _dict_to_key_list(attr_dict):
    out_list = []
    for key in attr_dict.keys():
        out_list.append(f'{key}: ...')
    return out_list


def rad_to_hour_str(rad):
    """
    Converts an angle in radians to hours minutes and seconds
    Args:
        rad: angle in radians

    Returns:
    xxhyymzz.zzzs
    """
    h_float = rad * convert_unit('rad', 'hour', 'trigonometric')
    h_int = np.floor(h_float)
    m_float = (h_float - h_int) * 60
    m_int = np.floor(m_float)
    s_float = (m_float - m_int) * 60
    return f'{int(h_int):02d}h{int(m_int):02d}m{s_float:06.3f}s'


def rad_to_deg_str(rad):
    """
    Converts an angle in radians to degrees minutes and seconds
    Args:
        rad: angle in radians

    Returns:
    xx\u00B0yymzz.zzzs
    """
    d_float = rad * convert_unit('rad', 'deg', 'trigonometric')
    if d_float < 0:
        d_float *= -1
        sign = '-'
    else:
        sign = '+'
    d_int = np.floor(d_float)
    m_float = (d_float - d_int) * 60
    m_int = np.floor(m_float)
    s_float = (m_float - m_int) * 60
    return f'{sign}{int(d_int):02d}\u00B0{int(m_int):02d}m{s_float:06.3f}s'


def print_summary_header(filename, print_len=100, frame_char='#', frame_width=3):
    """
    Print a summary header dynamically adjusted to the filename
    Args:
        filename: filename
        print_len: Length of the print on screen
        frame_char: Character to frame header
        frame_width: Width of the frame

    Returns:

    """
    title = 'Summary for:'
    filename, file_nlead, file_ntrail, print_len = _compute_spacing(filename, print_len=print_len,
                                                                    frame_width=frame_width)
    title, title_nlead, title_ntrail, _ = _compute_spacing(title, print_len=print_len, frame_width=frame_width)
    print(print_len * frame_char)
    _print_centralized(title, title_nlead, title_ntrail, frame_width, frame_char)
    _print_centralized(filename, file_nlead, file_ntrail, frame_width, frame_char)
    print(print_len * frame_char)

    stack = inspect.stack()
    class_name = stack[1][0].f_locals["self"].__class__.__name__
    doc_string = f"\nFull documentation for {class_name} objects' API at: \n" \
                 f'https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/mds/index.html#' \
                 f'astrohack.mds.{class_name}'
    print(doc_string)


def _compute_spacing(string, print_len=100, frame_width=3):
    spc = ' '
    nchar = len(string)
    if 2 * (nchar // 2) != nchar:
        nchar += 1
        string += spc
    cont_len = nchar + 2 * frame_width + 2
    if 2 * (print_len // 2) != print_len:
        print_len += 1
    if cont_len > print_len:
        print_len += cont_len - print_len

    nlead = int(print_len // 2 - nchar // 2 - frame_width)
    ntrail = print_len - nlead - 2 * frame_width - nchar
    return string, nlead, ntrail, print_len


def _print_centralized(string, nlead, ntrail, frame_width, frame_char):
    spc = ' '
    print(f'{frame_width * frame_char}{nlead * spc}{string}{ntrail * spc}{frame_width * frame_char}')


def print_method_list(method_list, alignment='l', print_len=100):
    """Print the method list of a mds object"""
    name_len = 0
    for obj_method in method_list:
        meth_len = len(obj_method.__name__)
        if meth_len > name_len:
            name_len = meth_len
    desc_len = print_len - name_len - 3 - 4  # Separators and padding

    print('\nAvailable methods:')
    table = PrettyTable()
    table.field_names = ['Methods', 'Description']
    table.align = alignment
    for obj_method in method_list:
        table.add_row([obj_method.__name__, textwrap.fill(obj_method.__doc__.splitlines()[0][1:], width=desc_len)])
    print(table)
    print()


def format_value_error(value, error, scaling, tolerance):
    """Format values based and errors based on the significant digits"""
    if np.isfinite(value) and np.isfinite(error):
        value *= scaling
        error *= scaling
        if abs(value) < tolerance:
            value = 0.0
        if abs(error) < tolerance:
            error = 0.0
        if value == 0 and error == 0:
            return f'{value} \u00b1 {error}'
        elif error > abs(value):
            places = round(np.log10(error))
            if places < 0:
                places = abs(places)
                return f'{value:.{places}f} \u00B1 {error:.{places}f}'
            else:
                if places in [-1, 0, 1]:
                    places = 2
                if value == 0:
                    digits = places - round(np.log10(abs(error)))
                else:
                    digits = places - round(np.log10(abs(value)))
                value = significant_figures_round(value, digits)
                error = significant_figures_round(error, places)
                return f'{value} \u00b1 {error}'
        else:
            digits = round(abs(np.log10(abs(value)))) - 1
            if digits in [-1, 0, 1]:
                digits = 2
            value = significant_figures_round(value, digits)
            error = significant_figures_round(error, digits - 1)
            return f'{value} \u00b1 {error}'
    else:
        return f'{value} \u00b1 {error}'


def get_str_idx_in_list(target, array):
    for i_tgt, item in enumerate(array):
        if target == item:
            return i_tgt
    logger.error(f'Target {target} not found in {array}')
    return None


def bool_to_str(boolean):
    if boolean:
        return 'yes'
    else:
        return 'no'


