import datetime
import inspect
import json
import os
import pathlib
import shutil

import toolviper.utils.logger as logger
from toolviper.utils.console import Colorize

from astrohack.utils.algorithms import data_from_version_needs_patch
from astrohack.utils.package_info import get_astrohack_version

DIMENSION_KEY = "_ARRAY_DIMENSIONS"
colorize = Colorize()


def check_ms_exists(ms_name):
    ms_path = pathlib.Path(ms_name)
    if not ms_path.exists():
        raise FileNotFoundError(f"{ms_name} does not exist.")
    if not ms_path.is_dir():
        raise RuntimeError(f"{ms_name} is a file not a directory.")

    ms_tbl_info_path = pathlib.Path(ms_name + "/table.info")
    if not ms_tbl_info_path.is_file():
        raise RuntimeError(
            f"{ms_name} does not contain a table.info file, probably not a proper ms or cal table."
        )


def check_if_file_can_be_opened(filename, file_creator, minimal_version):
    if os.path.exists(filename):
        pass
    else:
        raise FileNotFoundError(f"{filename} cannot be found.")

    try:
        with open(f"{filename}/.zattrs") as root_attrs_file:
            file_metadata = json.load(root_attrs_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} is not a proper astrohack file")
    except json.JSONDecodeError:
        raise ValueError(f"{filename} metadata is not properly formatted")

    try:
        origin_info = file_metadata["origin_info"]
    except KeyError:
        raise ValueError(f"{filename} metadata is missing origin information")

    if origin_info["origin"] != "astrohack":
        raise ValueError(f"{filename} was not created by astrohack")

    if not isinstance(file_creator, list):
        file_creator = [file_creator]
    if origin_info["creator_function"] not in file_creator:
        raise ValueError(
            f"{filename} was created by {origin_info['creator_function']} but {' or '.join(file_creator)} was expected"
        )

    file_version = origin_info["version"]

    if data_from_version_needs_patch(file_version, minimal_version):
        raise ValueError(
            f"{filename} was created by astrohack version {file_version} which has a deprecated file"
            f" format, please rerun astrohack on this dataset from scratch to access the data in it."
        )


def overwrite_file(file, overwrite):
    path = pathlib.Path(file)

    if (path.exists() is True) and (overwrite is False):
        logger.error(
            f"{file} already exists. To overwrite set overwrite to True, or remove current file."
        )

        raise FileExistsError(f"{file} exists.")

    elif (path.exists() is True) and (overwrite is True):
        if file.endswith(".zarr"):
            logger.warning(f"{file} will be overwritten.")
            shutil.rmtree(file)
        else:
            logger.warning(
                f"{file} may not be valid astrohack file. Check the file name again."
            )
            raise TypeError(f"IncorrectFileType: {file}")


def add_caller_and_version_to_dict(in_dict, direct_call=False):
    if direct_call:
        ipos = 1
    else:
        ipos = 2
    curr_time = datetime.datetime.now()
    local_tz = curr_time.astimezone().tzinfo
    time_str = curr_time.strftime(f"%Y-%m-%d %H:%M:%S {local_tz}")

    in_dict["origin_info"] = {
        "origin": "astrohack",
        "version": get_astrohack_version(),
        "creator_function": inspect.stack()[ipos].function,
        "creation_time": time_str,
    }
