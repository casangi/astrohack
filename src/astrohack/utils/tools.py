import os
import glob
import toolviper
import shutil
import sys
import numpy as np
import toolviper.utils.console as console
from contextlib import contextmanager

from casacore import tables
from toolviper.utils import logger as logger

from typing import Union


@contextmanager
def silence_stdout():
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target


@contextmanager
def silence_stderr():
    old_target = sys.stderr
    try:
        with open(os.devnull, "w") as new_target:
            sys.stderr = new_target
            yield new_target
    finally:
        sys.stderr = old_target


def file_search(root: str = "/", file_name=None) -> Union[None, str]:
    colorize = console.Colorize()

    if root == "/":
        toolviper.utils.logger.warning(
            "File search from root could take some time ..."
        )

    toolviper.utils.logger.debug(
        "Searching {root} for {file_name}, please wait ...".format(
            root=colorize.blue(root),
            file_name=colorize.blue(file_name)
        )
    )

    for file in glob.glob("{root}/**".format(root=root), recursive=True):
        if file_name in file:
            basename = os.path.dirname(file)
            return basename

    logger.warning(f"Couldn't locate filename: {colorize.blue(file_name)}")

    return None


def split_pointing_table(ms_name, antennas):
    """ Split pointing table to contain only specified antennas

    :param ms_name: Measurement file
    :type ms_name: str
    :param antennas: List of antennas to sub-select on.
    :type antennas: list (str)
    """

    # Need to get thea antenna-id values for the input antenna names. This is not available in the POINTING table,
    # so we build the values from the ANTENNA table.

    table = "/".join((ms_name, 'ANTENNA'))
    query = 'select NAME from {table}'.format(table=table)

    ant_names = np.array(tables.taql(query).getcol('NAME'))
    ant_id = np.arange(len(ant_names))

    query_ant = np.searchsorted(ant_names, antennas)

    ant_list = " or ".join(["ANTENNA_ID=={ant}".format(ant=ant) for ant in query_ant])

    # Build new POINTING table from the sub-selection of antenna values.
    table = "/".join((ms_name, "POINTING"))

    selection = "select * from {table} where {antennas}".format(table=table, antennas=ant_list)

    reduced = tables.taql(selection)

    # Copy the new table to the source measurement set.
    table = "/".join((ms_name, 'REDUCED'))

    reduced.copy(newtablename='{table}'.format(table=table), deep=True)
    reduced.done()

    # Remove old POINTING table.
    shutil.rmtree("/".join((ms_name, 'POINTING')))

    # Rename REDUCED table to POINTING
    tables.tablerename(
        tablename="/".join((ms_name, 'REDUCED')),
        newtablename="/".join((ms_name, 'POINTING'))
    )


def get_valid_state_ids(
        obs_modes,
        desired_intent="MAP_ANTENNA_SURFACE",
        excluded_intents=('REFERENCE', 'SYSTEM_CONFIGURATION')
):
    """
    Get scan and subscan IDs
    SDM Tables Short Description (https://drive.google.com/file/d/16a3g0GQxgcO7N_ZabfdtexQ8r2jRbYIS/view)
    2.54 ScanIntent (p. 150)
    MAP ANTENNA SURFACE : Holography calibration scan

    2.61 SubscanIntent (p. 152)
    MIXED : Pointing measurement, some antennas are on-source, some off-source
    REFERENCE : reference measurement (used for boresight in holography).
    SYSTEM_CONFIGURATION: dummy scans at the begininng of each row at the VLA.
    Undefined : ?
    """

    valid_state_ids = []
    for i_mode, mode in enumerate(obs_modes):
        if desired_intent in mode:
            bad_words = 0
            for intent in excluded_intents:
                if intent in mode:
                    bad_words += 1
            if bad_words == 0:
                valid_state_ids.append(i_mode)
    return valid_state_ids


def get_telescope_lat_lon_rad(telescope):
    """
    Return array center's latitude, longitude and distance to the center of the earth based on the coordinate reference
    Args:
        telescope: Telescope object

    Returns:
    Array center  latitude, longitude and distance to the center of the Earth in meters
    """
    if telescope.array_center['refer'] == 'ITRF':
        lon = telescope.array_center['m0']['value']
        lat = telescope.array_center['m1']['value']
        rad = telescope.array_center['m2']['value']
    else:

        msg = f'Unsupported telescope position reference :{telescope.array_center["refer"]}'
        logger.error(msg)
        raise Exception(msg)

    return lon, lat, rad
