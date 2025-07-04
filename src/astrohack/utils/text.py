import json
import inspect
import textwrap

import numpy as np
from astropy.time import Time
from prettytable import PrettyTable
from toolviper.utils import logger as logger

from astrohack.utils.conversion import convert_unit


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
                temp.append("{:>3}".format(array[k]).rjust(indent, " "))
            else:
                temp.append("{:>3}".format(array[k]))

        str_line += ", ".join(temp)
        str_line += "\n"

    temp = []
    if remainder > 0:
        for i in range(remainder):
            index = columns * rows + i

            if i == 0:
                temp.append("{:>3}".format(array[index]).rjust(indent, " "))
            else:
                temp.append("{:>3}".format(array[index]))

        str_line += ", ".join(temp)

    print(str_line)


def approve_prefix(key):
    approved_prefix = ["ant_", "map_", "ddi_"]

    for prefix in approved_prefix:
        if key.startswith(prefix):
            return True

    if not key.endswith("_info"):
        logger.warning(
            f"File meta data contains and unknown key ({key}), the file may not complete properly."
        )

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
    wrds = input_string.split("/")
    wrds[-1] = prefix + "_" + wrds[-1]
    return "/".join(wrds)


def print_holog_obs_dict(holog_obj):
    OPEN_DICT = ":{"
    CLOSE_DICT = "}"

    OPEN_LIST = ":["
    CLOSE_LIST = "]"

    logger.info("| ********************************************************** |")
    logger.info("|                 HOLOG OBSERVATION DICTIONARY               |")
    logger.info("| ********************************************************** |\n\n")

    for ddi_key, ddi_value in holog_obj.items():
        logger.info(
            "{ddi_key} {open_bracket}".format(ddi_key=ddi_key, open_bracket=OPEN_DICT)
        )
        for map_key, map_value in holog_obj[ddi_key].items():
            logger.info(
                "{map_key: >10} {open_bracket}".format(
                    map_key=map_key, open_bracket=OPEN_DICT
                )
            )
            for attr_key, attr_value in holog_obj[ddi_key][map_key].items():
                if "scans" in attr_key:
                    logger.info(
                        "{attr_key: >12} {open_list}".format(
                            attr_key=attr_key, open_list=OPEN_LIST
                        )
                    )

                    scan_list = ", ".join(
                        list(map(str, holog_obj[ddi_key][map_key][attr_key]))
                    )

                    # The print justification in notebook is weird on this and seems to move according to list length

                    logger.info("{scan: >18}".format(scan=scan_list))
                    logger.info("{close_bracket: >10}".format(close_bracket=CLOSE_LIST))

                elif "ant" in attr_key:
                    logger.info(
                        "{attr_key: >12} {open_bracket}".format(
                            attr_key=attr_key, open_bracket=OPEN_DICT
                        )
                    )
                    for ant_key, ant_value in holog_obj[ddi_key][map_key][
                        attr_key
                    ].items():
                        logger.info(
                            "{ant_key: >18} {open_list}".format(
                                ant_key=ant_key, open_list=OPEN_LIST
                            )
                        )
                        logger.info(
                            "{antenna: >25}".format(antenna=", ".join(ant_value))
                        )
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

    if param == "all":
        out_list = list(data_dict.keys())

    elif isinstance(param, str):
        out_list = [add_prefix(param, prefix)]

    elif isinstance(param, int):
        out_list = [f"{prefix}_{param}"]

    elif isinstance(param, (list, tuple)):
        out_list = []
        for item in param:
            if isinstance(item, str):
                out_list.append(add_prefix(item, prefix))
            elif isinstance(item, int):
                out_list.append(f"{prefix}_{item}")
            else:
                msg = f"Cannot interpret parameter {item} of type {type(item)}"
                logger.error(msg)
                raise Exception(msg)
    else:
        msg = f"Cannot interpret parameter {param} of type {type(param)}"
        logger.error(msg)
        raise Exception(msg)

    return out_list


def get_default_file_name(input_file: str, output_type: str) -> str:
    known_data_types = [
        ".ms",
        ".holog.zarr",
        ".image.zarr",
        ".locit.zarr",
        ".combine.zarr",
        ".position.zarr",
    ]

    output_file = None

    for suffix in known_data_types:
        if input_file.endswith(suffix):
            base_name = input_file.rstrip(suffix)
            output_file = "".join((base_name, output_type))

    if not output_file:
        output_file = "".join((input_file, output_type))

    logger.info("Creating output file name: {}".format(output_file))

    return output_file


def print_data_contents(data_dict, field_names, alignment="l"):
    """
    Factorized printing of the prettytable with the data contents
    Args:
        data_dict: Dictionary with data to be displayed
        field_names: Field names in the table
        alignment: Contents of the table to be aligned Left or Right
    """
    table = create_pretty_table(field_names, alignment)
    depth = len(field_names)
    if depth == 3:
        for item_l1 in data_dict.keys():
            for item_l2 in data_dict[item_l1].keys():
                table.add_row(
                    [item_l1, item_l2, list(data_dict[item_l1][item_l2].keys())]
                )
    elif depth == 2:
        for item_l1 in data_dict.keys():
            if "info" in item_l1:
                pass
            else:
                table.add_row([item_l1, list(data_dict[item_l1].keys())])
    elif depth == 1:
        for item_l1 in data_dict.keys():
            table.add_row([item_l1])
    else:
        raise Exception(f"Unhandled case len(field_names) == {depth}")

    print("\nContents:")
    print(table)


def print_dict_table(
    input_parameters, split_key=None, alignment="l", heading="Input Parameters"
):
    """
    Print a summary of the attributes
    Args:
        input_parameters: Dictionary containing metadata attributes
        split_key: key to be sqrt and displayed as nx X ny
        alignment: Column alignment
        heading: a small heading for the table

    Returns:

    """
    print(f"\n{heading}:")
    table = create_pretty_table(["Parameter", "Value"], alignment)

    for key, item in input_parameters.items():
        if key == split_key:
            n_side = int(np.sqrt(input_parameters[key]))
            table.add_row([key, f"{n_side:d} x {n_side:d}"])
        if isinstance(item, dict):
            table.add_row([key, _dict_to_key_list(item)])
        else:
            table.add_row([key, item])
    print(table)


def _dict_to_key_list(attr_dict):
    out_list = []
    for key in attr_dict.keys():
        out_list.append(f"{key}: ...")
    return out_list


def rad_to_hour_str(rad):
    """
    Converts an angle in radians to hours minutes and seconds
    Args:
        rad: angle in radians

    Returns:
    xxhyymzz.zzzs
    """
    h_float = rad * convert_unit("rad", "hour", "trigonometric")
    h_int = np.floor(h_float)
    m_float = (h_float - h_int) * 60
    m_int = np.floor(m_float)
    s_float = (m_float - m_int) * 60
    return f"{int(h_int):02d}h{int(m_int):02d}m{s_float:06.3f}s"


def rad_to_deg_str(rad):
    """
    Converts an angle in radians to degrees minutes and seconds
    Args:
        rad: angle in radians

    Returns:
    xx\u00b0yymzz.zzzs
    """
    d_float = rad * convert_unit("rad", "deg", "trigonometric")
    if d_float < 0:
        d_float *= -1
        sign = "-"
    else:
        sign = "+"
    d_int = np.floor(d_float)
    m_float = (d_float - d_int) * 60
    m_int = np.floor(m_float)
    s_float = (m_float - m_int) * 60
    return f"{sign}{int(d_int):02d}\u00b0{int(m_int):02d}m{s_float:06.3f}s"


def print_summary_header(filename, print_len=100, frame_char="#", frame_width=3):
    """
    Print a summary header dynamically adjusted to the filename
    Args:
        filename: filename
        print_len: Length of the print on screen
        frame_char: Character to frame header
        frame_width: Width of the frame

    Returns:

    """
    title = "Summary for:"
    filename, file_nlead, file_ntrail, print_len = _compute_spacing(
        filename, print_len=print_len, frame_width=frame_width
    )
    title, title_nlead, title_ntrail, _ = _compute_spacing(
        title, print_len=print_len, frame_width=frame_width
    )
    print(print_len * frame_char)
    _print_centralized(title, title_nlead, title_ntrail, frame_width, frame_char)
    _print_centralized(filename, file_nlead, file_ntrail, frame_width, frame_char)
    print(print_len * frame_char)

    stack = inspect.stack()
    class_name = stack[1][0].f_locals["self"].__class__.__name__
    doc_string = (
        f"\nFull documentation for {class_name} objects' API at: \n"
        f"https://astrohack.readthedocs.io/en/stable/_api/autoapi/astrohack/mds/index.html#"
        f"astrohack.mds.{class_name}"
    )
    print(doc_string)


def _compute_spacing(string, print_len=100, frame_width=3):
    spc = " "
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
    spc = " "
    print(
        f"{frame_width * frame_char}{nlead * spc}{string}{ntrail * spc}{frame_width * frame_char}"
    )


def print_method_list(method_list, alignment="l", print_len=100):
    """Print the method list of a mds object"""
    name_len = 0
    for obj_method in method_list:
        meth_len = len(obj_method.__name__)
        if meth_len > name_len:
            name_len = meth_len
    desc_len = print_len - name_len - 3 - 4  # Separators and padding

    print("\nAvailable methods:")
    table = create_pretty_table(["Methods", "Description"], alignment)
    for obj_method in method_list:
        table.add_row(
            [
                obj_method.__name__,
                textwrap.fill(obj_method.__doc__.splitlines()[0][1:], width=desc_len),
            ]
        )
    print(table)
    print()


def format_frequency(freq_value, unit="Hz", decimal_places=4):
    if isinstance(freq_value, str):
        freq_value = float(freq_value)
    if freq_value >= 1e12:
        unitout = "THz"
    elif freq_value >= 1e9:
        unitout = "GHz"
    elif freq_value >= 1e6:
        unitout = "MHz"
    elif freq_value >= 1e3:
        unitout = "kHz"
    else:
        unitout = unit
    fac = convert_unit(unit, unitout, "frequency")
    return format_value_unit(fac * freq_value, unitout, decimal_places)


def format_wavelength(wave_value, unit="m", decimal_places=2):
    if isinstance(wave_value, str):
        wave_value = float(wave_value)
    if wave_value >= 1:
        unitout = "m"
    elif wave_value >= 1e-2:
        unitout = "cm"
    elif wave_value >= 1e-3:
        unitout = "mm"
    elif wave_value >= 1e-6:
        unitout = "um"
    elif wave_value >= 1e-9:
        unitout = "nm"
    else:
        unitout = unit
    fac = convert_unit(unit, unitout, "length")
    return format_value_unit(fac * wave_value, unitout, decimal_places)


def format_duration(duration, unit="sec", decimal_places=2):
    duration = np.abs(duration * convert_unit(unit, "sec", "time"))
    oneminu = convert_unit("min", "sec", "time")
    onehour = convert_unit("hour", "sec", "time")
    oneday = convert_unit("day", "sec", "time")

    if duration < 1:
        if duration < 1e-6:
            unitout = "nsec"
        elif duration < 1e-3:
            unitout = "usec"
        else:
            unitout = "msec"
        fac = convert_unit("sec", unitout, "time")
        return format_value_unit(fac * duration, unitout, decimal_places)
    elif duration < oneminu:
        return format_value_unit(duration, "sec", decimal_places)
    elif oneminu <= duration < onehour:
        minu = int(np.floor(duration / oneminu))
        seco = duration - minu * oneminu
        return f"{minu} min, {format_value_unit(seco, 'sec', decimal_places)}"
    elif onehour <= duration < oneday:
        hour = int(np.floor(duration / onehour))
        rest = duration - hour * onehour
        minu = int(np.floor(rest / oneminu))
        seco = rest - minu * oneminu
        return (
            f"{hour} hour, {minu} min, {format_value_unit(seco, 'sec', decimal_places)}"
        )
    else:
        day = int(np.floor(duration / oneday))
        rest = duration - day * oneday
        hour = int(np.floor(rest / onehour))
        rest -= hour * onehour
        minu = int(np.floor(rest / oneminu))
        seco = rest - minu * oneminu
        return f"{day} day, {hour} hour, {minu} min, {format_value_unit(seco, 'sec', decimal_places)}"


def format_angular_distance(user_value, unit="rad", decimal_places=2):
    one_deg = np.pi / 180
    dist_value = np.abs(user_value)
    if dist_value >= np.pi / 180:
        unitout = "deg"
    elif dist_value >= one_deg / 60:
        unitout = "amin"
    elif dist_value >= one_deg / 3.6e3:
        unitout = "asec"
    elif dist_value >= one_deg / 3.6e6:
        unitout = "masec"
    else:
        unitout = "uasec"
    fac = convert_unit(unit, unitout, "trigonometric")
    return format_value_unit(fac * user_value, unitout, decimal_places)


def format_label(label, separators=("_", "\n"), new_separator=" "):
    if isinstance(label, str):
        out_label = label
    else:
        out_label = str(label)
    for sep in separators:
        out_label = out_label.replace(sep, new_separator)
    return out_label.capitalize()


def format_value_unit(value, unit, decimal_places=2):
    return f"{value:.{decimal_places}f} {unit}"


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
            return f"{value} \u00b1 {error}"
        elif error > abs(value):
            places = round(np.log10(error))
            if places < 0:
                places = abs(places)
                return f"{value:.{places}f} \u00b1 {error:.{places}f}"
            else:
                if places in [-1, 0, 1]:
                    places = 2
                if value == 0:
                    digits = places - round(np.log10(abs(error)))
                else:
                    digits = places - round(np.log10(abs(value)))
                value = significant_figures_round(value, digits)
                error = significant_figures_round(error, places)
                return f"{value} \u00b1 {error}"
        else:
            digits = round(abs(np.log10(abs(value)))) - 1
            if digits in [-1, 0, 1]:
                digits = 2
            value = significant_figures_round(value, digits)
            error = significant_figures_round(error, digits - 1)
            return f"{value} \u00b1 {error}"
    else:
        return f"{value} \u00b1 {error}"


def fixed_format_error(value, error, scaling, significance_scale):
    """
    Format value and error based on a significance scale
    Args:
        value: value to be formatted
        error: error to be formatted
        scaling: scaling to be applied to value and error
        significance_scale: scale for which signifcant values are expected

    Returns:
        formatted string with value +- error
    """
    out_val = value * scaling
    out_err = error * scaling
    after_comma = int(np.ceil(np.max([0, -np.log10(significance_scale)]) + 1))
    out_fmt = f" {after_comma+2}.{after_comma}f"
    return f"{out_val:{out_fmt}} \u00b1 {out_err:{out_fmt}}"


def bool_to_str(boolean):
    if boolean:
        return "yes"
    else:
        return "no"


def string_to_ascii_file(string, filename):
    outfile = open(filename, "w")
    outfile.write(string + "\n")
    outfile.close()


def create_pretty_table(field_names, alignment="c"):
    table = PrettyTable()
    table.field_names = field_names
    if isinstance(alignment, list) or isinstance(alignment, tuple):
        if len(field_names) != len(alignment):
            msg = "If alignment is not a single string alignment must have the same length of field_names"
            logger.error(msg)
            raise Exception(msg)
        for i_field, field in enumerate(field_names):
            table.align[field] = alignment[i_field]
    elif isinstance(alignment, str):
        if len(alignment) != 1:
            msg = "Alignment string must be of length 1"
            logger.error(msg)
            raise Exception(msg)
        table.align = alignment
    return table


def create_dataset_label(ant_id, ddi_id, separator=":"):
    if "ant_" in ant_id:
        ant_name = get_data_name(ant_id)
    else:
        ant_name = ant_id
    if ddi_id is None:
        return f"{ant_name.upper()}"
    else:
        if isinstance(ddi_id, int):
            ddi_name = str(ddi_id)
        elif "ddi_" in ddi_id:
            ddi_name = get_data_name(ddi_id)
        else:
            ddi_name = ddi_id
        return f"{ant_name.upper()}{separator} DDI {ddi_name}"


def get_data_name(data_id):
    return data_id.split("_")[1]


def significant_figures_round(x, digits):
    if np.isscalar(x):
        if x == 0 or not np.isfinite(x):
            return x

        digits = int(digits - np.ceil(np.log10(abs(x))))
        return round(x, digits)

    elif isinstance(x, list) or isinstance(x, np.ndarray):
        return list(map(significant_figures_round, x, [digits] * len(x)))

    else:
        logger.warning("Unknown data type.")

        return x


def statistics_to_text(
    data_statistics: dict, keys: list = None, num_format: str = None
):
    if keys is None:
        key_list = list(data_statistics.keys())
    else:
        key_list = keys

    n_keys = len(key_list)

    if num_format == "dynamic":
        format_list = []
        for key in key_list:
            format_list.append(dynamic_format(data_statistics[key]))
    elif num_format is None:
        format_list = [".2f"] * n_keys
    else:
        format_list = [num_format] * n_keys

    outstr = ""
    for ikey, key in enumerate(key_list):
        outstr += f"{key}={data_statistics[key]:{format_list[ikey]}}, "
    outstr = outstr[:-2]

    return outstr


def dynamic_format(value):
    data_oom = np.log10(np.abs(value))
    if data_oom >= 4 or data_oom < -3:
        return ".3e"
    else:
        return f"{round(abs(data_oom))+1}f"


def format_byte_size(byte_size):
    base = 1024
    labels = ["B", "KB", "MB", "GB", "TB"]
    format_size = byte_size
    i_label = 0
    while format_size > base and i_label < len(labels) - 1:
        i_label += 1
        format_size /= byte_size
    return f"{format_size:.2f} {labels[i_label]}"


def format_object_contents(obj):
    total_size = 0
    outstr = f"Contents of this {type(obj).__name__} object:\n"
    for key, item in obj.__dict__.items():
        size = item.__sizeof__()
        outstr += f"   {key:22s} -> | {type(item).__name__} |"
        if isinstance(item, np.ndarray):
            outstr += " ("
            for dim_size in item.shape:
                outstr += f"{dim_size},"
            outstr = outstr[:-1] + f") [{item.dtype}]"
        outstr += f" -> {format_byte_size(size)}\n"
        total_size += size
    outstr += f"Total size = {format_byte_size(total_size)}\n"
    return outstr


def format_az_el_information(az_el_dict, key="center", unit="deg", precision=".1f"):
    if key == "center":
        prefix = "@ l,m = (0,0),"
    elif key in ["mean", "median"]:
        prefix = key.capitalize()
    else:
        raise ValueError(f"Unrecognized key: {key}")

    az_el = np.array(az_el_dict[key]) * convert_unit("rad", unit, "trigonometric")
    prefix += " Az, El"
    az_el_label = (
        f"{prefix} = ({az_el[0]:{precision}}, {az_el[1]:{precision}}) [{unit}]"
    )
    return az_el_label


def format_general_information(
    obs_dict,
    tab,
    ident,
    key_size,
    az_el_key="mean",
    phase_center_unit="radec",
    az_el_unit="deg",
    time_format="%d %h %Y, %H:%M:%S",
    precision=".1f",
):
    outstr = f"{ident}General:\n"
    tab = tab + ident
    key_order = [
        "telescope name",
        "antenna name",
        "station",
        "reference antennas",
        "source",
        "phase center",
        "az el info",
        "start time",
        "stop time",
        "duration",
    ]
    for key in key_order:
        item = obs_dict[key]
        line = f"{tab}{key.capitalize().replace('_', ' '):{key_size}s} => "
        if "phase center" in key:
            if phase_center_unit == "radec":
                line += f"{rad_to_hour_str(item[0])} {rad_to_deg_str(item[1])} [FK5]"
            else:
                fac = convert_unit("rad", phase_center_unit, "trigonometric")
                line += f"({fac*item[0]:{precision}}, {fac*item[1]:{precision}}) [{phase_center_unit}]"
        elif "time" in key:
            date = Time(item, format="mjd").to_datetime()
            line += f"{date.strftime(time_format)} (UTC)"
        elif "az el info" in key:
            line += f"{format_az_el_information(item, az_el_key, unit=az_el_unit, precision=precision)}"
        elif "duration" == key:
            line += f"{format_duration(item)}"
        else:
            line += str(item)
        outstr += f"{line}\n"

    return outstr


def format_spectral_information(freq_dict, tab, ident, key_size):
    outstr = f"{ident}Spectral:\n"
    tab += ident
    for key, item in freq_dict.items():
        outstr += f"{tab}{key.capitalize().replace('_', ' '):{key_size}s} => "
        if "range" in key:
            outstr += f"{format_frequency(item[0], decimal_places=3)} to {format_frequency(item[1], decimal_places=3)}"
        elif "number" in key:
            outstr += f"{item}"
        elif "wavelength" in key:
            outstr += format_wavelength(item, decimal_places=3)
        else:
            outstr += format_frequency(item, decimal_places=3)
        outstr += "\n"

    return outstr


def format_beam_information(beam_dict, tab, ident, key_size):
    outstr = f"{ident}Beam:\n"
    tab += ident
    for key, item in beam_dict.items():
        outstr += f"{tab}{key.capitalize().replace('_', ' '):{key_size}s} => "
        if key == "cell size":
            if isinstance(item, list):
                outstr += f"{format_angular_distance(item[0])} by {format_angular_distance(item[1])}"
            else:
                outstr += format_angular_distance(item)
        elif key == "grid size":
            outstr += f"{item[0]} by {item[1]} pixels"
        else:
            outstr += f"From {format_angular_distance(item[0])} to {format_angular_distance(item[1])}"
        outstr += "\n"
    return outstr


def format_aperture_information(aperture_dict, tab, ident, key_size):
    outstr = f"{ident}Aperture:\n"
    tab += ident
    for key, item in aperture_dict.items():
        outstr += f"{tab}{key.capitalize().replace('_', ' '):{key_size}s} => "
        if key == "grid size":
            outstr += f"{item[0]} by {item[1]} pixels"
        else:
            outstr += f"{format_wavelength(item[0])} by {format_wavelength(item[1])}"
        outstr += "\n"
    return outstr


def format_observation_summary(
    obs_sum,
    tab_size=3,
    tab_count=0,
    az_el_key="mean",
    phase_center_unit="radec",
    az_el_unit="deg",
    time_format="%d %h %Y, %H:%M:%S",
    precision=".1f",
    key_size=18,
):
    spc = " "
    major_tab = tab_count * tab_size * spc
    one_tab = tab_size * spc
    ident = major_tab

    outstr = format_general_information(
        obs_sum["general"],
        az_el_key=az_el_key,
        phase_center_unit=phase_center_unit,
        az_el_unit=az_el_unit,
        time_format=time_format,
        precision=precision,
        tab=one_tab,
        ident=ident,
        key_size=key_size,
    )
    outstr += "\n"
    outstr += format_spectral_information(obs_sum["spectral"], one_tab, ident, key_size)

    outstr += "\n"
    outstr += format_beam_information(obs_sum["beam"], one_tab, ident, key_size)

    if obs_sum["aperture"] is not None:
        outstr += "\n"
        outstr += format_aperture_information(
            obs_sum["aperture"], one_tab, ident, key_size
        )
    return outstr


def make_header(heading, separator, header_width, buffer_width):
    spc = " "
    sep_line = f"{header_width * separator}\n"
    len_head = len(heading)
    before_blank = (header_width - 2 * buffer_width - len_head) // 2
    if 2 * buffer_width + len_head + 2 * before_blank < header_width:
        after_blank = before_blank + 1
    else:
        after_blank = before_blank
    outstr = sep_line
    buffer = buffer_width * separator
    outstr += f"{buffer}{before_blank*spc}{heading}{after_blank*spc}{buffer}\n"
    outstr += sep_line + "\n"
    return outstr
