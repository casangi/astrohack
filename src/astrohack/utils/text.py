import inspect
import textwrap

import numba.typed.typeddict
import numpy as np
from astropy.time import Time
from prettytable import PrettyTable
from toolviper.utils import logger as logger

from astrohack.utils.conversion import convert_unit

lnbr = "\n"
spc = " "
undscr = "_"


def tuple_inspect(param_tuple):
    outstr = ""
    for idx, item in enumerate(param_tuple):
        # print(idx, type(item))
        outstr += f"{idx:3d} => "
        if isinstance(item, (list, tuple)):
            outstr += f"{len(item)} = {item}"
        elif isinstance(item, np.ndarray):
            outstr += f"{item.shape} sum = {np.sum(item)}"
        elif isinstance(item, numba.typed.typeddict.Dict):
            outstr += f"dict = {dict(item).keys()}"
        elif isinstance(item, dict):
            outstr += f"dict = {item.keys()}"
        else:
            outstr += f"{item}"
        outstr += lnbr
    return outstr


def approve_prefix(key):
    approved_prefix = ["ant_", "map_", "ddi_"]

    for prefix in approved_prefix:
        if key.startswith(prefix):
            return True

    if not key.endswith("_info"):
        logger.warning(
            f"File meta data contains an unknown key ({key}), the file may not complete properly."
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
    wrds[-1] = prefix + undscr + wrds[-1]
    return "/".join(wrds)


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
                raise ValueError(msg)
    else:
        msg = f"Cannot interpret parameter {param} of type {type(param)}"
        logger.error(msg)
        raise ValueError(msg)

    return out_list


def get_default_file_name(
    input_filename: str, output_ext: str, user_filename: str | None
) -> str:

    if user_filename is None:
        known_data_types = [
            ".ms",
            ".cal",
            ".sbd",
            ".point.zarr",
            ".holog.zarr",
            ".image.zarr",
            ".locit.zarr",
            ".combine.zarr",
            ".position.zarr",
            ".beamcut.zarr",
        ]

        output_filename = None

        for suffix in known_data_types:
            if input_filename.endswith(suffix):
                base_name = input_filename.removesuffix(suffix)
                output_filename = "".join((base_name, output_ext))

        if output_filename is None:
            output_filename = "".join((input_filename, output_ext))

    else:
        output_filename = user_filename

    if output_filename[-1] == "/":
        output_filename = output_filename[:-1]

    logger.info(f"Creating output file name: {output_filename}")
    return output_filename


def _get_tree_field_names(data_tree, field_names=None):
    key_labels = {"ant": "Antenna", "ddi": "DDI", "map": "Mapping", "cut": "Cut"}

    if data_tree.is_leaf:
        return field_names
    else:
        this_level_keys = list(data_tree.keys())
        f_key = this_level_keys[0]
        key_label = key_labels[f_key.split(undscr)[0]]
        if field_names is None:
            field_names = [key_label]
        else:
            field_names.append(key_label)
        return _get_tree_field_names(data_tree[f_key], field_names)


def get_data_content_string(data_object, alignment="l", field_names=None):
    """
    Factorized printing of the prettytable with the data contents
    Args:
        data_object: Dictionary with data to be displayed
        field_names: Field names in the table
        alignment: Contents of the table to be aligned Left or Right
    """

    field_names = _get_tree_field_names(data_object, field_names)

    table = create_pretty_table(field_names, alignment)
    depth = len(field_names)
    if depth == 3:
        for item_l1 in data_object.keys():
            for item_l2 in data_object[item_l1].keys():
                table.add_row(
                    [item_l1, item_l2, list(data_object[item_l1][item_l2].keys())]
                )
    elif depth == 2:
        for item_l1 in data_object.keys():
            if "info" in item_l1:
                pass
            else:
                table.add_row([item_l1, list(data_object[item_l1].keys())])
    elif depth == 1:
        for item_l1 in data_object.keys():
            table.add_row([item_l1])
    else:
        raise NotImplementedError(f"Unhandled case len(field_names) == {depth}")

    outstr = f"{lnbr}Data Contents:{lnbr}"
    outstr += table.get_string()
    return outstr


def print_dict_types(le_dict, ident=4, show_values=False):
    for key, value in le_dict.items():
        if isinstance(value, dict):
            print(f"{ident * spc}{key}:")
            print_dict_types(value, ident=ident + 4, show_values=show_values)
        else:
            if show_values:
                print(f"{ident * spc}{key}: {type(value)} => {value}")
            else:
                print(f"{ident * spc}{key}: {type(value)}")


def get_property_string(
    root_attrs, split_key=None, alignment="l", heading="Input Parameters"
):
    outstr = f"{lnbr}Data origin:{lnbr}"
    for key, value in root_attrs["origin_info"].items():
        outstr += f"{f'{key}:':17s} {value}{lnbr}"

    outstr += f"{lnbr}{heading}:{lnbr}"
    input_parameters = root_attrs["input_parameters"]
    table = create_pretty_table(["Parameter", "Value"], alignment)
    for key, item in input_parameters.items():
        if key == split_key:
            n_side = int(np.sqrt(input_parameters[key]))
            table.add_row([key, f"{n_side:d} x {n_side:d}"])
        if isinstance(item, dict):
            table.add_row([key, _dict_to_key_list(item)])
        else:
            table.add_row([key, item])
    outstr += table.get_string()
    return outstr


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


def get_summary_header(filename, print_len=80, frame_char="#", frame_width=3):
    """
    Print a summary header dynamically adjusted to the filename
    Args:
        filename: filename
        print_len: Length of the print on screen
        frame_char: Character to frame header
        frame_width: Width of the frame

    Returns:
        header string

    """
    title = "Summary for:"
    filename_str, file_nlead, file_ntrail, print_len = _compute_spacing(
        filename, print_len=print_len, frame_width=frame_width
    )
    title, title_nlead, title_ntrail, _ = _compute_spacing(
        title, print_len=print_len, frame_width=frame_width
    )
    bar = print_len * frame_char + lnbr
    outstr = bar
    outstr += (
        _centralized_string(title, title_nlead, title_ntrail, frame_width, frame_char)
        + lnbr
    )
    outstr += (
        _centralized_string(
            filename_str, file_nlead, file_ntrail, frame_width, frame_char
        )
        + lnbr
    )
    outstr += bar
    return outstr


def _compute_spacing(string, print_len=100, frame_width=3):
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


def _centralized_string(string, nlead, ntrail, frame_width, frame_char):
    return f"{frame_width * frame_char}{nlead * spc}{string}{ntrail * spc}{frame_width * frame_char}"


def get_method_list_string(astrohack_obj, alignment="l", print_len=80):
    """Print the method list of a mds object"""
    method_list = inspect.getmembers(astrohack_obj, predicate=inspect.ismethod)

    name_len = 0
    for name, method in method_list:
        if name[0:2] == "__":
            continue
        meth_len = len(name)
        if meth_len > name_len:
            name_len = meth_len
    desc_len = print_len - name_len - 3 - 4  # Separators and padding

    outstr = f"{2 * lnbr}Available methods:{lnbr}"
    table = create_pretty_table(["Methods", "Description"], alignment)
    for name, method in method_list:
        # ignore dunder methods
        if name[0:2] == "__":
            continue
        docstring = inspect.getdoc(method)
        if docstring is None:
            continue
        lines = docstring.splitlines()
        method_summary = "Failed to get method summary..."
        for line in lines:
            if line.strip() != "":
                method_summary = line.strip()
                break

        table.add_row(
            [
                name,
                textwrap.fill(method_summary, width=desc_len),
            ]
        )
    outstr += table.get_string() + lnbr
    return outstr


def format_frequency(freq_value, unit="Hz", decimal_places=4, add_nu=False):
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
    if add_nu:
        outstr = r"$\nu$ = "
    else:
        outstr = ""
    outstr += format_value_unit(fac * freq_value, unitout, decimal_places)
    return outstr


def format_wavelength(user_value, unit="m", decimal_places=2):
    wave_value = np.abs(user_value)
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
    return format_value_unit(fac * user_value, unitout, decimal_places)


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


def format_label(label, separators=(undscr, lnbr), new_separator=spc):
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
    out_fmt = f" {after_comma + 2}.{after_comma}f"
    return f"{out_val:{out_fmt}} \u00b1 {out_err:{out_fmt}}"


def bool_to_str(boolean):
    if boolean:
        return "yes"
    else:
        return "no"


def string_to_ascii_file(string, filename):
    outfile = open(filename, "w")
    outfile.write(string + lnbr)
    outfile.close()


def create_pretty_table(field_names, alignment="c"):
    table = PrettyTable()
    table.field_names = field_names
    if isinstance(alignment, list) or isinstance(alignment, tuple):
        if len(field_names) != len(alignment):
            msg = "If alignment is not a single string alignment must have the same length of field_names"
            logger.error(msg)
            raise ValueError(msg)
        for i_field, field in enumerate(field_names):
            table.align[field] = alignment[i_field]
    elif isinstance(alignment, str):
        if len(alignment) != 1:
            msg = "Alignment string must be of length 1"
            logger.error(msg)
            raise ValueError(msg)
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
    return data_id.split(undscr)[1]


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
    data_statistics: dict, keys: list | None = None, num_format: str | None = None
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
        return f"{round(abs(data_oom)) + 1}f"


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


def create_informative_label_from_summary(
    summary,
    azel_unit,
    freq_precision=3,
    add_date=True,
    add_source=False,
):
    ant_name = summary["general"]["antenna name"]
    ant_station = summary["general"]["station"]
    source = summary["general"]["source"]
    freq = summary["spectral"]["rep. frequency"]
    az_el = np.array(summary["general"]["az el info"]["mean"]) * convert_unit(
        "rad", azel_unit, "trigonometric"
    )
    date = Time(summary["general"]["start time"], format="mjd").to_datetime()
    time_str = date.strftime("%Y-%m-%d")
    if azel_unit == "deg":
        azel_precision = ".1f"
    else:
        azel_precision = ".3f"

    outstr = f"{ant_name.upper()} @ {ant_station.upper()}, "
    outstr += f"{format_frequency(freq, add_nu=False, decimal_places=freq_precision)}, "
    if add_source:
        outstr += f"{source} @ "
    outstr += (
        f"Az={az_el[0]:{azel_precision}}, El={az_el[1]:{azel_precision}} {azel_unit}"
    )
    if add_date:
        outstr += f", {time_str}"

    return outstr


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
    outstr = f"{ident}General:{lnbr}"
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
                line += f"({fac * item[0]:{precision}}, {fac * item[1]:{precision}}) [{phase_center_unit}]"
        elif "time" in key:
            date = Time(item, format="mjd").to_datetime()
            line += f"{date.strftime(time_format)} (UTC)"
        elif "az el info" in key:
            line += f"{format_az_el_information(item, az_el_key, unit=az_el_unit, precision=precision)}"
        elif "duration" == key:
            line += f"{format_duration(item)}"
        else:
            line += str(item)
        outstr += f"{line}{lnbr}"

    return outstr


def format_spectral_information(freq_dict, tab, ident, key_size):
    outstr = f"{ident}Spectral:{lnbr}"
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
        outstr += lnbr

    return outstr


def format_beam_information(beam_dict, tab, ident, key_size):
    outstr = f"{ident}Beam:{lnbr}"
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
        outstr += lnbr
    return outstr


def format_aperture_information(aperture_dict, tab, ident, key_size):
    outstr = f"{ident}Aperture:{lnbr}"
    tab += ident
    for key, item in aperture_dict.items():
        outstr += f"{tab}{key.capitalize().replace('_', ' '):{key_size}s} => "
        if key == "grid size":
            outstr += f"{item[0]} by {item[1]} pixels"
        else:
            outstr += f"{format_wavelength(item[0])} by {format_wavelength(item[1])}"
        outstr += lnbr
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
    outstr += lnbr
    outstr += format_spectral_information(obs_sum["spectral"], one_tab, ident, key_size)

    outstr += lnbr
    outstr += format_beam_information(obs_sum["beam"], one_tab, ident, key_size)

    if obs_sum["aperture"] is not None:
        outstr += lnbr
        outstr += format_aperture_information(
            obs_sum["aperture"], one_tab, ident, key_size
        )
    return outstr


def make_header(heading, separator, header_width, buffer_width):
    sep_line = f"{header_width * separator}{lnbr}"
    len_head = len(heading)
    before_blank = (header_width - 2 * buffer_width - len_head) // 2
    if 2 * buffer_width + len_head + 2 * before_blank < header_width:
        after_blank = before_blank + 1
    else:
        after_blank = before_blank
    outstr = sep_line
    buffer = buffer_width * separator
    outstr += f"{buffer}{before_blank * spc}{heading}{after_blank * spc}{buffer}{lnbr}"
    outstr += sep_line + lnbr
    return outstr


def add_heading_to_html(content, heading_level):
    return f"<h{heading_level}>{content}</h{heading_level}>{lnbr}"


def add_paragraph_to_html(content):
    return f"<p>{content}</p>{lnbr}"


def create_html_file_from_body(
    body_text, title, html_file, alignment="left", add_collapsible_tools=True
):
    out_file_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
    body {{ font-family: sans-serif; text-align: {alignment}; margin-top: 50px; }}
    img {{ max-width: 100%; height: auto; border: 2px solid #ccc; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }}
"""
    if add_collapsible_tools:
        out_file_str += """/* Style the button */
.collapsible {
  background-color: #777;
  color: white;
  cursor: pointer;
  padding: 15px;
  width: 35%;
  border: 2px;
  margin: 10px;
  text-align: left;
  outline: none;
  font-size: 20px;
  font-weight: 600;
  display: block;
  text-decoration: none;
}
.content .collapsible {
  font-size: 16px;
}
/* Add a background color to the button if it is clicked on */
.active, .collapsible:hover {
  background-color: #000;
}
/* Hide the collapsible content by default */
.content {
  padding: 0 18px;
  display: none;
  overflow: hidden;
  background-color: #ffffff;
}
.row {
display:flex;
flex-direction:row;
gap:4rem;
}
"""
    out_file_str += """
</style>
</head>
<body>"""
    out_file_str += body_text

    if add_collapsible_tools:
        out_file_str += """<script>
const coll = document.getElementsByClassName("collapsible");
for (let i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    const content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  });
}
</script>"""
    out_file_str += """</body>
</html>"""
    with open(html_file, "w") as f:
        f.write(out_file_str)
    return


def create_embedded_html_image(image_file, description, width=None, height=None):
    img_binary_str = get_image_binary_data(image_file)
    img_ext = image_file.split(".")[-1]
    if height is None:
        height_str = "auto"
    else:
        height_str = str(height)
    if width is None:
        width_str = "auto"
    else:
        width_str = str(width)
    image_str = f"""<img src="data:image/{img_ext};base64,{img_binary_str}" alt="{description}" style="width: {width_str}; height: {height_str};">"""
    return image_str


def create_single_html_image_with_header(
    image_file, description, width="60%", height=None, heading_level: int = 2
):
    image_str = add_heading_to_html(description, heading_level)
    return (
        image_str
        + create_embedded_html_image(image_file, description, width, height)
        + lnbr
    )


def make_collapsible_block(html_content: str, summary: str, html_id: str):
    class_name = "collapsible"
    collapsible_block = (
        f'<a class="{class_name}" id="{html_id}" href="#{html_id}">{summary}</a>{lnbr}'
    )
    collapsible_block += f'<div class="content">{lnbr}'
    collapsible_block += html_content + lnbr
    collapsible_block += f"</div>{lnbr}"
    return collapsible_block


def create_side_by_side_html_images_with_header(
    image_file1,
    image_file2,
    description,
    width=None,
    height=None,
    heading_level: int = 2,
):
    if width is None:
        width = "40%"
    image_str = add_heading_to_html(description, heading_level)
    image_str += f"<div>{lnbr}"
    image_str += (
        create_embedded_html_image(image_file1, description, width, height) + lnbr
    )
    image_str += (
        create_embedded_html_image(image_file2, description, width, height) + lnbr
    )
    image_str += f"</div>{lnbr}"
    return image_str


def add_preformatted_text_file_to_html(
    text_file, header, heading_level: int = 2, alignment: str = "left"
):
    import pathlib

    if len(text_file) > 255:
        full_text_str = text_file
    elif pathlib.Path(text_file).exists():
        with open(text_file) as opened_text_file:
            full_text_str = opened_text_file.read()
    else:
        full_text_str = text_file

    html_str = add_heading_to_html(header, heading_level)
    html_str += f"""<pre style="text-align: {alignment};">{lnbr}{full_text_str}{lnbr}</pre>{lnbr}"""
    return html_str


def get_image_binary_data(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        # Read the raw binary data
        binary_data = image_file.read()
        # Encode to base64 bytes, then decode to an ASCII string
        base64_string = base64.b64encode(binary_data).decode("utf-8")
        return base64_string
