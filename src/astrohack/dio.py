import json
import pathlib
import toolviper.utils.logger as logger

import numpy as np

from casacore import tables
from rich.console import Console

from astrohack.mds import AstrohackImageFile
from astrohack.mds import AstrohackHologFile
from astrohack.mds import AstrohackPanelFile
from astrohack.mds import AstrohackPointFile
from astrohack.mds import AstrohackLocitFile
from astrohack.mds import AstrohackPositionFile

from astrohack.utils.text import print_array

from typing import Union, List, NewType, Dict, Any, NoReturn

JSON = NewType("JSON", Dict[str, Any])


def open_holog(file: str) -> Union[AstrohackHologFile, None]:
    """ Open holog file and return instance of the holog data object. Object includes summary function to list\
     available dictionary keys.

    :param file: Path to holog file.
    :type file: str
  
    :return: Holography holog object; None if file not found.
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

    _data_file = AstrohackHologFile(file=file)

    if _data_file.open():
        return _data_file

    else:
        return None


def open_image(file: str) -> Union[AstrohackImageFile, None]:
    """ Open image file and return instance of the image data object. Object includes summary function to list \
    available dictionary keys.

    :param file: Path to image file.
    :type file: str
  
    :return: Holography image object; None if file not found.
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

    _data_file = AstrohackImageFile(file=file)

    if _data_file.open():
        return _data_file

    else:
        return None


def open_panel(file: str) -> Union[AstrohackPanelFile, None]:
    """ Open panel file and return instance of the panel data object. Object includes summary function to list \
    available dictionary keys.

    :param file: Path ot panel file.
    :type file: str

    :return: Holography panel object; None if file not found.
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

    _data_file = AstrohackPanelFile(file=file)

    if _data_file.open():
        return _data_file

    else:
        return None


def open_locit(file: str) -> Union[AstrohackLocitFile, None]:
    """ Open locit file and return instance of the locit data object. Object includes summary function to list \
    available dictionary keys.

    :param file: Path of locit file.
    :type file: str

    :return: locit object; None if file not found.
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

    _data_file = AstrohackLocitFile(file=file)

    if _data_file.open():
        return _data_file

    else:
        return


def open_position(file: str) -> Union[AstrohackPositionFile, None]:
    """ Open position file and return instance of the position data object. Object includes summary function to list \
    available dictionary keys.

    :param file: Path of position file.
    :type file: str

    :return: position object; None if file does not exist.
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

    _data_file = AstrohackPositionFile(file=file)

    if _data_file.open():
        return _data_file

    else:
        return None


def open_pointing(file: str) -> Union[AstrohackPointFile, None]:
    """ Open pointing file and return instance of the pointing data object. Object includes summary function to list\
     available dictionary keys.

    :param file: Path to pointing file.
    :type file: str
  
    :return: Holography pointing object; None if file does not exist.
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

    _data_file = AstrohackPointFile(file=file)

    if _data_file.open():
        return _data_file

    else:
        return None


def fix_pointing_table(ms_name: str, reference_antenna: List[str]) -> None:
    """ Fix pointing table for a user defined subset of reference antennas.

    :param ms_name: Measurement set name.
    :type ms_name: str

    :param reference_antenna: List of reference antennas.
    :type reference_antenna: list
  
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

    path = pathlib.Path(ms_name)
    ms_name_fullpath = str(path.absolute().resolve())

    if not path.exists():
        logger.error("Error finding file: {file}".format(file=ms_name_fullpath))

    ms_table = "/".join((ms_name_fullpath, 'ANTENNA'))

    query = 'select NAME from "{table}"'.format(table=ms_table)

    ant_names = np.array(tables.taql(query).getcol('NAME'))

    ant_id = np.arange(len(ant_names))

    query_ant = np.searchsorted(ant_names, reference_antenna)

    ms_table = "/".join((ms_name_fullpath, 'POINTING'))

    ant_list = " or ".join(["ANTENNA_ID=={ant}".format(ant=ant) for ant in query_ant])

    update = "update {table} set POINTING_OFFSET=0, TARGET=DIRECTION where {antennas}".format(table=ms_table,
                                                                                              antennas=ant_list)

    tables.taql(update)

    ms_table = "/".join((ms_name_fullpath, "HISTORY"))
    tb = tables.table(ms_table, readonly=False)

    message = tb.getcol("MESSAGE")

    if "pnt_tbl:fixed" not in message:
        tb.addrows(nrows=1)
        length = len(message)
        tb.putcol(columnname="MESSAGE", value='pnt_tbl:fixed', startrow=length)


def print_json(
        obj: JSON,
        indent: int = 6,
        columns: int = 7
) -> None:
    """ Print formatted JSON dictionary (** Deprecated by Console **)

    :param obj: JSON object
    :type obj: JSON

    :param indent: Indent to be used in JSON dictionary., defaults to 6
    :type indent: int, optional

    :param columns: Columns used to reshape the antenna list., defaults to 7
    :type columns: int, optional
    """
    import toolviper.utils.console as console

    color = console.Colorize()

    if isinstance(obj, np.ndarray):
        obj = list(obj)

    if isinstance(obj, list):
        if indent > 3:
            list_indent = indent - 3
        else:
            list_indent = 0

        print("{open}".format(open="[").rjust(list_indent, ' '))
        print_array(obj, columns=columns, indent=indent + 1)
        print("{close}".format(close="]").rjust(list_indent, ' '))

    else:
        for key, value in obj.items():
            key_str = "{key}{open}".format(key=key, open=":{")
            print("{key}".format(key=key_str).rjust(indent, ' '))
            print_json(value, indent + 4, columns=columns)
            print("{close}".format(close="}").rjust(indent - 4, ' '))


def inspect_holog_obs_dict(
        file: Union[str, JSON] = '.holog_obs_dict.json',
        style: str = 'static'
) -> Union[NoReturn, JSON]:
    """ Print formatted holography observation dictionary

    :param file: Input file, can be either JSON file or string., defaults to '.holog_obs_dict.json'
    :type file: str | JSON, optional

    :param style: Print style of JSON dictionary. This can be static, formatted generalized print out or dynamic, \
    prints a collapsible formatted dictionary, defaults to static
    :type style: str, optional

    .. _Description:

    **Example Usage**
    The `inspect_holog_obs_dict` loads a holography observation dict either from disk or from memory (as an return \
    value from `generate_holog_obs_dict`) and displays it in a more readable way like JSON.stringify() in javascript.

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

    if not isinstance(file, dict):
        try:
            with open(file) as json_file:
                json_object = json.load(json_file)

        except IsADirectoryError:
            try:
                with open(file+'/holog_obs_dict.json') as json_file:
                    json_object = json.load(json_file)
            except FileNotFoundError:
                logger.error("holog observations dictionary not found: {file}".format(file=file))
        except FileNotFoundError:
            logger.error("holog observations dictionary not found: {file}".format(file=file))

    else:
        json_object = file

    if style == 'dynamic':
        from IPython.display import JSON

        return JSON(json_object)

    else:
        console = Console()
        console.log(json_object, log_locals=False)
