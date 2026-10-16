import numpy as np
from toolviper.utils import logger as logger


def build_bad_scan_list(bad_scans_parameter):
    """
    Process a user input of bad scans and produce a list of bad scans.
    Args:
        bad_scans_parameter: user bad scan input, either an int, None, or a list of ints.

    Returns:
        an empty list if input is None, or an iterable of ints.
    """
    if bad_scans_parameter is None:
        return []
    elif isinstance(bad_scans_parameter, int):
        return [bad_scans_parameter]
    else:
        # Exclude scans is already iterable
        return bad_scans_parameter


def get_valid_state_ids(
    obs_modes,
    desired_intent="MAP_ANTENNA_SURFACE",
    excluded_intents=("REFERENCE", "SYSTEM_CONFIGURATION"),
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
    if telescope.array_center["refer"] == "ITRF":
        lon = telescope.array_center["m0"]["value"]
        lat = telescope.array_center["m1"]["value"]
        rad = telescope.array_center["m2"]["value"]
    else:
        msg = f"Unsupported telescope position reference :{telescope.array_center['refer']}"
        logger.error(msg)
        raise NotImplementedError(msg)

    return lon, lat, rad


def get_str_idx_in_list(target, array):
    for i_tgt, item in enumerate(array):
        if target == item:
            return i_tgt
    logger.error(f"Target {target} not found in {array}")
    return None


def raise_type_error(var_name, class_name):
    raise TypeError(f"{var_name} is not an object of {class_name}")


def check_is_proper_type(var, class_ref):
    if isinstance(var, class_ref):
        return True
    else:
        raise_type_error(var.__name__, class_ref.__name__)


def check_is_proper_array(array, ndim=None):
    check_is_proper_type(array, np.ndarray)
    if ndim is None:
        pass
    else:
        if isinstance(ndim, int):
            if len(array.shape) != ndim:
                raise ValueError(
                    f"{array.__name__} has a different number of dimensions ({len(array.shape)}) "
                    f"from what is expected ({ndim})."
                )
        else:
            raise TypeError(f"{ndim.__name__} must be set to an integer value.")


def check_is_proper_shape(array, shape):
    check_is_proper_array(array, len(shape))
    if not np.all(np.isclose(array.shape, shape)):
        raise ValueError(
            f"{array.__name__} shape ({array.shape}) is not what was expected({shape})."
        )
