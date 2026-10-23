import argparse
import inspect
import pathlib
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from astrohack.utils.text import (
    add_preformatted_text_file_to_html,
    format_duration,
    lnbr,
    spc,
)


def yesno(prompt):
    user_ans = input(f"{prompt} <(Y)es/(N)o>: ").lower()
    if user_ans == "y" or user_ans == "yes":
        return True
    elif user_ans == "n" or user_ans == "no":
        return False
    else:
        print("Use <yes> or <no>")
        return yesno(prompt)


def file_is_asdm(filename):
    file_path = Path(f"{filename}/ASDM.xml")
    return file_path.exists()


class MessageBoard:
    def __init__(self, width=None, block_char="#", spacing=1, blocking=3):
        if width is None:
            term_size = shutil.get_terminal_size((80, 20))
            self.width = term_size.columns
        else:
            self.width = width
        self.block_char = block_char
        self.spacing = (spacing,)
        self.blocking = blocking

        self.capo = blocking * block_char + spacing * spc
        self.coda = self.capo[::-1] + lnbr
        self.usable_width = self.width - 2 * spacing - 2 * blocking
        self.block_line = self.width * self.block_char + lnbr
        self.block_len = len(self.capo)

    def _end_line(self, line):
        line_len = len(line) + 1 - self.block_len
        spc_to_add = (self.usable_width - line_len - self.blocking) * spc
        out_line = self.capo + line + spc_to_add + self.coda
        return out_line

    def heading(self, user_msg):
        outstr = ""
        outstr += self.block_line
        head_wrds = user_msg.split()

        line = ""
        for wrd in head_wrds:
            wrd_len = len(wrd)
            if wrd_len > self.usable_width:
                raise ValueError(f"Word {wrd} is larger than the usable self.width")
            line_len = len(line) + wrd_len + 1
            if line_len > self.usable_width:
                outstr += self._end_line(line)
                line = wrd
            else:
                line += spc + wrd
        outstr += self._end_line(line)
        outstr += self.block_line
        print(outstr)
        return outstr

    def one_liner(self, msg):
        outstr = self.capo + msg + lnbr
        print(outstr)
        return outstr

    def done(self):
        return self.one_liner("Done!")

    def welcome_message(self, pipeline_type):
        self.heading(f"Welcome to the AstroHACK {pipeline_type} reduction pipeline")

    def goodbye_message(self, pipeline_type: str, param_dict: dict, duration: float):
        msg = (
            f"{pipeline_type.capitalize()} processing finished in {format_duration(duration)}"
            f", individual plots and text results saved at: {param_dict['exports_name']}."
            f" Checkout the HTML report at: {param_dict['report_name']}."
        )
        self.heading(msg)


def run_casatask(
    task_name: str,
    kwargs_dict: dict,
    msger: MessageBoard = None,
    intended_output: str | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Run a casatask and returns True if it has been run, False if it was skipped
    :param task_name: Casatask name
    :param kwargs_dict: Dict containing arguments for casatask
    :param msger: MessageBoard object
    :param intended_output: Possible intended output
    :param overwrite: Overwrite flag (only used when there is an intended output)
    :param verbose: Verbose flag (print messages regarding execution and execution time)
    :return: True when casatask was run, False otherwise
    """
    if intended_output is not None:
        intended_path = pathlib.Path(intended_output)
        if intended_path.exists():
            if overwrite:
                if verbose:
                    msger.one_liner(
                        f"{intended_output} already exists, overwriting it..."
                    )
                if intended_path.is_dir():
                    shutil.rmtree(intended_output)
                    shutil.rmtree(f"{intended_output}.flagversions", ignore_errors=True)
                else:
                    intended_path.unlink(missing_ok=True)
            else:
                if verbose:
                    msger.one_liner(
                        f"{intended_output} already exists, skipping its creation."
                    )
                return False

    if task_name == "plotms":
        import casaplotms

        casatask_func = casaplotms.plotms
    else:
        import casatasks

        casatask_func = getattr(casatasks, task_name)

    if verbose:
        msger.one_liner(f"Running {task_name}...")
    task_start_time = time.time()
    casatask_func(**kwargs_dict)
    task_end_time = time.time()
    if verbose:
        msger.one_liner(
            f"{task_name} finished in {format_duration(task_end_time - task_start_time)}."
        )
    return True


def parse_list_or_none(
    param_dict: dict,
    param_key: str,
    list_type: type = int,
    max_size: int = None,
) -> list:
    parameter_value = param_dict[param_key]
    if parameter_value is None:
        return None
    else:
        wrd_list = parameter_value.split(",")
        if list_type is not str:
            wrd_list = [list_type(wrd) for wrd in wrd_list]
        if max_size is not None:
            if len(wrd_list) > max_size:
                raise ValueError(f"List {param_key} must be at most of size {max_size}")
        return wrd_list


def parse_list_or_all(
    parameter_dict: dict,
    param_key: str,
    list_type: type = str,
    max_size: int = None,
) -> list:
    parameter_value = parameter_dict[param_key]
    if parameter_value == "all":
        return "all"
    else:
        wrd_list = parameter_value.split(",")
        if list_type is not str:
            wrd_list = [list_type(wrd) for wrd in wrd_list]
        if max_size is not None:
            if len(wrd_list) > max_size:
                raise ValueError(f"List {param_key} must be at most of size {max_size}")
        return wrd_list


def make_dict_str_simple(the_dict, ident=4):
    key_len = 0
    for key in the_dict.keys():
        if not isinstance(key, str):
            key = str(key)
        if len(key) > key_len:
            key_len = len(key)

    outstr = ""
    for key, value in the_dict.items():
        if not isinstance(key, str):
            key = str(key)
        outstr += f"{ident * ' '}{key:{key_len}s} => {value}{lnbr}"
    return outstr


def initialization_check(param_dict: dict, title: str):
    print()
    print(f"{title}:")
    print(make_dict_str_simple(param_dict))
    print()
    proceed_check(param_dict)
    print()


def proceed_check(param_dict: dict, prompt: str = "Proceed?"):
    if not param_dict["assume_yes"]:
        if not yesno(prompt):
            exit(0)


def list_input_tooltip(example):
    return f"for a list use comma separated values with no spaces, e.g.: '{example}'"


def created_filtered_kwargs_dict(param_dict: dict, function):
    valid_kwarg_keys = inspect.signature(function).parameters
    filtered_dict = {
        key: value for key, value in param_dict.items() if key in valid_kwarg_keys
    }
    return filtered_dict


def run_astrohack_function(
    param_dict: dict, function, msger: MessageBoard, verbose: bool = True
):
    function_name = function.__name__
    try:
        if verbose:
            msger.one_liner(f"Running {function_name}...")
        function_start_time = time.time()
        function(**created_filtered_kwargs_dict(param_dict, function))
        function_end_time = time.time()
        if verbose:
            msger.one_liner(
                f"{function_name} finished in {format_duration(function_end_time - function_start_time)}."
            )
        return True, None
    except Exception as the_exception:
        return False, the_exception


def base_name_determination(param_dict: dict):
    if param_dict["root_name"] is None:
        base_name = param_dict["filename"]
    else:
        base_name = param_dict["root_name"]
    base_name_wrds = base_name.split(".")
    if base_name_wrds[-1] == "ms":
        base_name = ".".join(base_name_wrds[:-1])
    return base_name


def asdm_test_and_import(
    param_dict: dict, base_name: str, msger: MessageBoard, for_holography: bool = False
):
    param_dict["is_asdm"] = file_is_asdm(param_dict["filename"])

    if param_dict["is_asdm"]:
        param_dict["msname"] = f"{base_name}.ms"
        if pathlib.Path(param_dict["msname"]).is_dir():
            execute_import = param_dict["reimport_asdm"]
        else:
            execute_import = True
        if execute_import:
            msger.one_liner("Input is an ASDM, importing it...")
            base_param_dict = {
                "asdm": param_dict["filename"],
                "vis": param_dict["msname"],
                "overwrite": param_dict["overwrite"],
            }
            if for_holography:
                base_param_dict.update(
                    {
                        "ocorr_mode": "co",
                        "asis": "Receiver CalAtmosphere",
                        "savecmds": True,
                        "outfile": f"{param_dict['msname']}.online-flags.txt",
                        "with_pointing_correction": True,
                        "applyflags": True,
                    }
                )

            run_casatask(
                "importasdm",
                base_param_dict,
                msger,
            )
        else:
            msger.one_liner("ASDM already imported, skipping import.")
    else:
        param_dict["msname"] = param_dict["filename"]

    return param_dict


def get_time_string_from_dict(time_dict, qa):
    time_str = qa.time(
        qa.quantity(
            time_dict["m0"]["value"],
            "d",
        ),
        form="ymd",
    )[0]
    time_str_wrds = time_str.split("/")
    out_str = "-".join(time_str_wrds[:3]) + spc + time_str_wrds[-1]
    return out_str


def get_lst_string_from_time_str(time_str):
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    vla_location = EarthLocation(
        lat=34.0785 * u.deg, lon=-107.6184 * u.deg, height=2124 * u.m
    )
    observing_time = Time(time_str, scale="utc", location=vla_location)
    lst = observing_time.sidereal_time("apparent").to_string(
        unit=u.hour, pad=True, sep=":"
    )
    return lst


def add_basic_info_and_parameters_to_report(param_dict: dict):
    try:
        import casatools

        casatools_available = True
    except ModuleNotFoundError:
        casatools_available = False

    html_str = add_preformatted_text_file_to_html(
        make_dict_str_simple(param_dict), "Pipeline parameters:"
    )

    if casatools_available and Path(param_dict["msname"]).is_dir():
        msmd = casatools.msmetadata()
        msmd.open(param_dict["msname"])
        timerange = msmd.timerangeforobs(0)
        start_time = timerange["begin"]
        end_time = timerange["end"]
        msmd.done()
        qa = casatools.quanta()

        start_time = get_time_string_from_dict(start_time, qa)
        end_time = get_time_string_from_dict(end_time, qa)
        times_dict = {
            "Observation start (UTC)": start_time,
            "Starting LST": get_lst_string_from_time_str(start_time),
            "Observation end (UTC)": end_time,
            "End LST": get_lst_string_from_time_str(end_time),
        }

        html_str += add_preformatted_text_file_to_html(
            make_dict_str_simple(times_dict), "Basic Time information:"
        )

    return html_str


def create_parser_with_base_options(pipeline_type: str, stage_choices: list):
    parser = argparse.ArgumentParser(
        description=f"{pipeline_type.capitalize()} reduction pipeline"
    )

    parser.add_argument(
        "filename", type=str, help="Path to the input dataset to process."
    )

    parser.add_argument(
        "refant",
        type=str,
        default=None,
        nargs="?",
        help="Reference antenna for calibration",
    )

    parser.add_argument(
        "-r",
        "--root-name",
        type=str,
        default=None,
        help="Root name for the products of the pipeline, default is ms_name without extension",
    )

    parser.add_argument(
        "-s",
        "--spw",
        type=str,
        default="all",
        help=f"Select SPWs for processing, {list_input_tooltip('0,1,2')}, default is %(default)s",
    )

    parser.add_argument(
        "-a",
        "--antenna",
        type=str,
        default="all",
        help=f"Select antennas for processing, {list_input_tooltip('ea01,ea02')}, default is %(default)s",
    )

    parser.add_argument(
        "-n",
        "--ncores",
        type=int,
        default=4,
        help="Number of cores to use, default is %(default)d",
    )

    parser.add_argument(
        "-m",
        "--memory-per-core",
        type=str,
        default="10GB",
        help="Memory per core to use, if no unit is given GB is assumed, default is %(default)s",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level to use, default is %(default)s",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if found",
    )

    parser.add_argument(
        "-y", "--assume-yes", action="store_true", help="Assume yes on proceed."
    )

    parser.add_argument(
        "--reimport-asdm",
        action="store_true",
        default=False,
        help="Forcefully re-import the asdm file is the ms already exists (default: %(default)s)",
    )

    # Example of parameter with choice
    parser.add_argument(
        "--starting-stage",
        type=str,
        default=stage_choices[0],
        choices=stage_choices,
        help="Starting stage in which to start processing (default: %(default)s).",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Dots Per Inch for plotting, default is %(default)d",
    )

    return parser


def basic_holography_parser(pipeline_type: str, stage_choices: list):
    parser = create_parser_with_base_options(pipeline_type, stage_choices)

    parser.add_argument(
        "-d",
        "--data-column",
        type=str,
        default="CORRECTED_DATA",
        help="Data column to be extracted from MS, default is %(default)s",
    )

    parser.add_argument(
        "--plot-pointing",
        action="store_true",
        help="Plot antenna pointing, default is %(default)s",
    )

    parser.add_argument(
        "--exclude-bad-antennas",
        default=None,
        type=str,
        help=f"Exclude antennas with bad data, {list_input_tooltip('ea18,ea01')}, default is %(default)s.",
    )

    parser.add_argument(
        "-q",
        "--quack-nchan",
        default=4,
        type=int,
        help="Number of channels to quack at the edge of the spectral window (default is %(default)s)",
    )

    parser.add_argument(
        "-f",
        f"--{pipeline_type}-field",
        default=None,
        type=str,
        help=f"Field Id or name containing {pipeline_type} data (default is to determine it from data)",
    )

    parser.add_argument(
        "--baseline-average-nearest",
        default=1,
        type=int,
        help="Number of baselines to average for each mapping antenna (default is %(default)s)",
    )

    parser.add_argument(
        "--pointing-interpolation-method",
        default="linear",
        type=str,
        choices=["linear", "gaussian"],
        help="Interpolation method to use for matching pointing and visibilities, default is %(default)s",
    )

    return parser


def fetch_ms_metadata_for_holograpy(param_dict, pipeline_type):
    import casatools

    # Fetch metadata from ms
    msmd = casatools.msmetadata()
    msmd.open(param_dict["msname"])
    cal_scans = msmd.scansforintent("*PHASE*")
    mapping_scans = msmd.scansforintent("*MAP*ON_SOURCE")
    mapping_spw_list = msmd.spwsforintent("*MAP*")
    all_spw_list = msmd.spwsforintent("*")
    mapping_fields = np.unique(msmd.fieldsforscans(mapping_scans))
    mapping_spw_nchan = [msmd.nchan(i_spw) for i_spw in mapping_spw_list]
    mapping_spw_names = [msmd.namesforspws(i_spw) for i_spw in mapping_spw_list]
    all_fields = msmd.fieldnames()
    msmd.done()

    base_band0 = []
    base_band1 = []
    for j_spw, spw_name in enumerate(mapping_spw_names):
        if "A0C0" in spw_name[0]:
            base_band0.append(mapping_spw_list[j_spw])
        elif "B0D0" in spw_name[0]:
            base_band1.append(mapping_spw_list[j_spw])
    base_bands = [base_band0, base_band1]
    base_bands = [base_band for base_band in base_bands if len(base_band) > 0]
    print(base_bands)
    del base_band0, base_band1
    base_band_strs = [f"{np.min(base_band)}~{np.max(base_band)}" for base_band in base_bands]

    field_key = f"{pipeline_type}_field"
    if param_dict[field_key] is None:
        if mapping_fields.size > 1:
            raise RuntimeError(
                f"More than 1 {pipeline_type} field, try splitting the ms"
            )
        param_dict[field_key] = mapping_fields[0]
    else:
        try:
            field_id = int(param_dict[field_key])
            if field_id > all_fields.size - 1 or field_id < 0:
                raise RuntimeError(
                    f"Specified {pipeline_type} field ID is out of range"
                )
        except ValueError:
            if param_dict[field_key] not in all_fields:
                raise RuntimeError(f"{param_dict[field_key]} not present in the ms")

    fchan = param_dict["quack_nchan"]
    if np.unique(mapping_spw_nchan).size > 1:
        quacked_list = []
        quacked_base_list = [[] for base_band in base_bands]
        for j_spw, i_nchan in enumerate(mapping_spw_nchan):
            i_spw = mapping_spw_list[j_spw]
            quacked_spw = f"{i_spw}:{fchan}~{i_nchan - fchan}"
            quacked_list.append(quacked_spw)
            for ibase, base_band in enumerate(base_bands):
                if i_spw in base_band:
                    quacked_base_list[ibase].append(quacked_spw)

        param_dict["quacked_spw_selection"] = ",".join(quacked_list)
        quacked_base_bands = []
        for quacked_base_band in quacked_base_list:
            quacked_base_bands.append(",".join(quacked_base_band))
    else:
        lchan = mapping_spw_nchan[0] - param_dict["quack_nchan"]
        minspw = f"{round(np.min(mapping_spw_list)):d}"
        maxspw = f"{round(np.max(mapping_spw_list)):d}"
        spwrange = f"{minspw}~{maxspw}"
        param_dict["quacked_spw_selection"] = f"{spwrange}:{fchan}~{lchan}"
        quacked_base_bands = []
        for base_band_str in base_band_strs:
            quacked_base_bands.append(f"{base_band_str}:{fchan}~{lchan}")

    param_dict["quacked_base_band_selection"] = quacked_base_bands
    delay_spwmap = []
    full_spwmap = []
    for i_spw in all_spw_list:
        if i_spw in base_bands:
            for ibase, base_band in enumerate(base_bands):
                if i_spw in base_band:
                    delay_spwmap.append(np.min(base_band))
        else:
            delay_spwmap.append(np.min(base_bands))
        full_spwmap.append(i_spw)
    param_dict["delay_spwmap"] = delay_spwmap
    param_dict["full_spwmap"] = full_spwmap

    # Convert to comma-separated string
    param_dict["calibration_scans"] = ",".join(map(str, cal_scans))
    param_dict[f"{pipeline_type}_scans"] = ",".join(map(str, mapping_scans))

    return param_dict


def parse_core_memory(user_memory_value):
    memory_units = ["kB", "MB", "GB", "TB", "B"]
    for unit in memory_units:
        if unit in user_memory_value:
            user_numeral = user_memory_value.replace(unit, "")
            try:
                user_numeral = float(user_numeral)
                return user_memory_value
            except ValueError:
                raise ValueError(
                    f"{user_memory_value} is not a valid memory description"
                )

    try:
        user_numeral = float(user_memory_value)
        return f"{user_numeral}GB"
    except ValueError:
        raise ValueError(f"{user_memory_value} is not a valid memory description")


def common_parameter_initialization_for_holography(
    pipeline_type: str,
    stages: list,
    extensions: dict,
    parse_function: Callable,
    msger: MessageBoard,
    for_holograpy: bool = False,
):
    param_dict = parse_function(pipeline_type, stages)

    base_name = base_name_determination(param_dict)
    param_dict = asdm_test_and_import(param_dict, base_name, msger, for_holograpy)

    for identifier, extension in extensions.items():
        param_dict[f"{identifier}_name"] = base_name + extension

    if param_dict["starting_stage"] == "calibration":
        if param_dict["refant"] is None:
            raise RuntimeError(
                "Reference antenna must be specified when starting from calibration"
            )
        param_dict = fetch_ms_metadata_for_holograpy(param_dict, pipeline_type)

    param_dict["antenna"] = parse_list_or_all(param_dict, "antenna")
    param_dict["spw"] = parse_list_or_all(param_dict, "spw", list_type=int)

    if param_dict["exclude_bad_antennas"] is not None:
        param_dict["exclude_bad_antennas"] = parse_list_or_all(
            param_dict, "exclude_bad_antennas"
        )
    param_dict["parallel"] = param_dict["ncores"] >= 2
    param_dict["memory_per_core"] = parse_core_memory(param_dict["memory_per_core"])

    return param_dict


def open_astrohack_file(open_function, file_name):
    mds_obj = open_function(file_name)
    if mds_obj is None:
        raise RuntimeError(f"{file_name} not found")
    return mds_obj


def client_initialization(
    param_dict,
    astrohack_stages,
):
    from toolviper.dask.client import local_client

    log_level = param_dict["log_level"]
    client_log_params = {
        "logger_name": "client",
        "log_to_term": True,
        "log_level": log_level,
        "log_to_file": False,
        "log_file": "client.log",
    }

    worker_log_params = {
        "logger_name": "worker",
        "log_to_term": True,
        "log_level": log_level,
        "log_to_file": False,
        "log_file": "client_worker.log",
    }

    if param_dict["ncores"] > 0 and param_dict["processing_stage"] in astrohack_stages:
        client = local_client(
            cores=param_dict["ncores"],
            memory_limit=param_dict["memory_per_core"],
            worker_log_params=worker_log_params,
            log_params=client_log_params,
        )
    else:
        client = None

    return client
