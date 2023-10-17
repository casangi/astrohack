import os
import json
import numpy as np

from casacore import tables

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

from astrohack.mds import AstrohackImageFile
from astrohack.mds import AstrohackHologFile
from astrohack.mds import AstrohackPanelFile
from astrohack.mds import AstrohackPointFile
from astrohack.mds import AstrohackLocitFile
from astrohack.mds import AstrohackPositionFile


from astrohack._utils._dio import _print_array


def open_holog(file):
    """ Open holog file and return instance of the holog data object. Object includes summary function to list\
     available dictionary keys.

    :param file: Path to holog file.
    :type file: str
  
    :return: Holography holog object.
    :rtype: AstrohackHologFile

    .. _Description:
    **AstrohackHologFile**
    Holog object allows the user to access holog data via compound dictionary keys with values, in order of depth, \
    `ddi` -> `map` -> `ant`. The holog object also provides a `summary()` helper function to list available keys for \
    each file. An outline of the holog object structure is show below:

    .. parsed-literal::
        holog_mds =
            {
                ddi_0:{
                    map_0:{
                         ant_0: holog_ds,
                             ⋮
                         ant_n: holog_ds
                    },
                    ⋮
                    map_p: …
                },
            ⋮
            ddi_m: …
            }
    """

    logger = _get_astrohack_logger()

    _data_file = AstrohackHologFile(file=file)

    if _data_file._open():
        return _data_file

    else:
        logger.error(f"Error opening holgraphy file: {file}")


def open_image(file):
    """ Open image file and return instance of the image data object. Object includes summary function to list \
    available dictionary keys.

    :param file: Path to image file.
    :type file: str
  
    :return: Holography image object.
    :rtype: AstrohackImageFile

    .. _Description:
    **AstrohackImageFile**
    Image object allows the user to access image data via compound dictionary keys with values, in order of depth, \
    `ant` -> `ddi`. The image object also provides a `summary()` helper function to list available keys for each file. \
    An outline of the image object structure is show below:

    .. parsed-literal::
       image_mds =
           {
               ant_0:{
                   ddi_0: image_ds,
                   ⋮
                   ddi_m: image_ds
               },
               ⋮
               ant_n: …
           }

    """

    logger = _get_astrohack_logger()

    _data_file = AstrohackImageFile(file=file)

    if _data_file._open():
        return _data_file

    else:
        logger.error(f"Error opening holgraphy image file: {file}")


def open_panel(file):
    """ Open panel file and return instance of the panel data object. Object includes summary function to list \
    available dictionary keys.

    :param file: Path ot panel file.
    :type file: str

    :return: Holography panel object.
    :rtype: AstrohackPanelFile

    .. _Description:
    **AstrohackPanelFile**
    Panel object allows the user to access panel data via compound dictionary keys with values, in order of depth, \
    `ant` -> `ddi`. The panel object also provides a `summary()` helper function to list available keys for each file.\
     An outline of the panel object structure is show below:

    .. parsed-literal::
        panel_mds =
            {
                ant_0:{
                    ddi_0: panel_ds,
                    ⋮
                    ddi_m: panel_ds
                },
                ⋮
                ant_n: …
            }

    """

    logger = _get_astrohack_logger()

    _data_file = AstrohackPanelFile(file=file)

    if _data_file._open():
        return _data_file

    else:
        logger.error(f"Error opening holgraphy panel file: {file}")


def open_locit(file):
    """ Open locit file and return instance of the locit data object. Object includes summary function to list \
    available dictionary keys.

    :param file: Path of locit file.
    :type file: str

    :return: locit object.
    :rtype: AstrohackLocitFile

    .. _Description:
    **AstrohackLocitFile**
    Locit object allows the user to access locit data via compound dictionary keys with values, in order of depth,\
     `ant` -> `ddi`. The locit object also provides a `summary()` helper function to list available keys for each file.\
      An outline of the locit object structure is show below:

    .. parsed-literal::
        locit_mds =
            {
                ant_0:{
                    ddi_0: locit_ds,
                    ⋮
                    ddi_m: locit_ds
                },
                ⋮
                ant_n: …
            }

    """

    logger = _get_astrohack_logger()

    _data_file = AstrohackLocitFile(file=file)

    if _data_file._open():
        return _data_file

    else:
        logger.error(f"Error opening holgraphy locit file: {file}")


def open_position(file):
    """ Open position file and return instance of the position data object. Object includes summary function to list \
    available dictionary keys.

    :param file: Path of position file.
    :type file: str

    :return: position object.
    :rtype: AstrohackPositionFile

    .. _Description:
    **AstrohackPositionFile**
    position object allows the user to access position data via compound dictionary keys with values, in order of \
    depth, `ant` -> `ddi`. The position object also provides a `summary()` helper function to list available keys for\
     each file. An outline of the position object structure is show below:

    .. parsed-literal::
        position_mds =
            {
                ant_0:{
                    ddi_0: position_ds,
                    ⋮
                    ddi_m: position_ds
                },
                ⋮
                ant_n: …
            }

    """

    logger = _get_astrohack_logger()

    _data_file = AstrohackPositionFile(file=file)

    if _data_file._open():
        return _data_file

    else:
        logger.error(f"Error opening holgraphy position file: {file}")


def open_pointing(file):
    """ Open pointing file and return instance of the pointing data object. Object includes summary function to list\
     available dictionary keys.

    :param file: Path to pointing file.
    :type file: str
  
    :return: Holography pointing object.
    :rtype: AstrohackPointFile

    .. _Description:

    **AstrohackPointFile**
    Pointing object allows the user to access pointing data via dictionary key with value based on `ant`. The pointing \
    object also provides a `summary()` helper function to list available keys for each file. An outline of the pointing\
     object structure is show below:

    .. parsed-literal::
        point_mds =
            {
                ant_0: point_ds,
                ⋮
                ant_n: point_ds
            }

    """

    logger = _get_astrohack_logger()

    _data_file = AstrohackPointFile(file=file)

    if _data_file._open():
        return _data_file

    else:
        logger.error(f"Error opening holography pointing file: {file}")


def fix_pointing_table(ms_name, reference_antenna):    
    """ Fix pointing table for a user defined subset of reference antennas.

    :param ms_name: Measurement set name.
    :type file: str

    :param reference_antenna: List of reference antennas.
    :type file: list
  
    .. _Description:

    **Example Usage**
    The `fix_pointing_table` function takes the measurement set name and a list of reference antennas.

    .. parsed-literal::
        import astrohack

        astrohack.dio.fix_pointing_table(
            ms_name="data/ea25_cal_small_before_fixed.split.ms", 
            reference_antenna=["ea15"]
        )


  """

    ms_table = "/".join((ms_name, 'ANTENNA'))

    query = 'select NAME from {table}'.format(table=ms_table)

    ant_names = np.array(tables.taql(query).getcol('NAME'))
    
    ant_id = np.arange(len(ant_names))

    query_ant = np.searchsorted(ant_names, reference_antenna)

    ms_table = "/".join((ms_name, 'POINTING'))

    ant_list = " or ".join(["ANTENNA_ID=={ant}".format(ant=ant) for ant in query_ant])

    update = "update {table} set POINTING_OFFSET=0, TARGET=DIRECTION where {antennas}".format(table=ms_table,
                                                                                              antennas=ant_list)

    tables.taql(update)

    ms_table = "/".join((ms_name, "HISTORY"))
    tb = tables.table(ms_table, readonly=False)

    message = tb.getcol("MESSAGE")

    if "pnt_tbl:fixed" not in message:
        tb.addrows(nrows=1)
        length = len(message)
        tb.putcol(columnname="MESSAGE", value='pnt_tbl:fixed', startrow=length)


def print_json(obj, indent=6, columns=7):
    """ Print formatted JSON dictionary

    :param obj: JSON object
    :type obj: JSON
    :param indent: Indent to be used in JSON dictionary., defaults to 6
    :type indent: int, optional
    :param columns: Columns used to reshape the antenna list., defaults to 7
    :type columns: int, optional
    """
  
    if isinstance(obj, np.ndarray):
        obj = list(obj)

    if isinstance(obj, list):
        if indent > 3:
            list_indent = indent-3
        else:
            list_indent = 0

        print("{open}".format(open="[").rjust(list_indent, ' '))
        _print_array(obj, columns=columns, indent=indent + 1)
        print("{close}".format(close="]").rjust(list_indent, ' '))

    else:
        for key, value in obj.items():
            key_str="{key}{open}".format(key=key, open=":{")
            print("{key}".format(key=key_str).rjust(indent, ' '))
            print_json(value, indent+4, columns=columns)
            print("{close}".format(close="}").rjust(indent-4, ' '))


def inspect_holog_obs_dict(file='.holog_obs_dict.json', style='static', indent=6, columns=7):
    """ Print formatted holography observation dictionary

    :param file: Input file, can be either JSON file or string., defaults to '.holog_obs_dict.json'
    :type file: str | JSON, optional
    :param style: Print style of JSON dictionary. This can be static, formatted generalized print out or dynamic, prints a collapsible formatted dictionary, defaults to static
    :type style: str, optional
    :param indent: Indent to be used in JSON dictionary., defaults to 6
    :type indent: int, optional
    :param columns: Columns used to reshape the antenna list., defaults to 7
    :type columns: int, optional

    .. _Description:

    **Example Usage**
    The `inspect_holog_obs_dict` loads a holography observation dict either from disk or from memory (as an return value from `generate_holog_obs_dict`) and displays it in a more readable way like JSON.stringify() in javascript.

    .. parsed-literal::
        import astrohack

        astrohack.dio.inspect_holog_obs_dict(file=holog_obs_obj)

        >> ddi_0:{
            map_0:{
                scans:{
                        [
                            8,   9,  10,  12,  13,  14,  16
                            17,  18,  23,  24,  25,  27,  28
                            29,  31,  32,  33,  38,  39,  40
                            42,  43,  44,  46,  47,  48,  53
                            54,  55,  57
                        ]
                }
                ant:{
                    ea06:{
                        [
                            ea04, ea25
                        ]
                    }
                }
            }
        } 
    """
    logger = _get_astrohack_logger()

    if not isinstance(file, dict):
        try:
            with open(file) as json_file:
                json_object = json.load(json_file)
            
        except FileNotFoundError:
            logger.error("holog observations dictionary not found: {file}".format(file=file))
    
    else:
        json_object = file

    if style == 'dynamic': 
        from IPython.display import JSON

        return JSON(json_object)
        
    
    else:
        print_json(obj=json_object, indent=indent, columns=columns)
