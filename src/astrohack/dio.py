import numpy as np

from casacore import tables

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

from astrohack._utils._mds import AstrohackImageFile
from astrohack._utils._mds import AstrohackHologFile
from astrohack._utils._mds import AstrohackPanelFile
from astrohack._utils._mds import AstrohackPointFile


def open_holog(file):
    """ Open holog file and return instance of the holog data object. Object includes summary function to list available dictionary keys.

    :param file: Path to holog file.
    :type file: str
  
    :return: Holography holog object.
    :rtype: AstrohackHologFile

    .. _Description:
    **AstrohackHologFile**
    Holog object allows the user to access holog data via compound dictionary keys with values, in order of depth, `ddi` -> `map` -> `ant`. The holog object also provides a `summary()` helper function to list available keys for each file. An outline of the holog object structure is show below:

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
    """ Open image file and return instance of the image data object. Object includes summary function to list available dictionary keys.

    :param file: Path to image file.
    :type file: str
  
    :return: Holography image object.
    :rtype: AstrohackImageFile

    .. _Description:
    **AstrohackImageFile**
    Image object allows the user to access image data via compound dictionary keys with values, in order of depth, `ant` -> `ddi`. The image object also provides a `summary()` helper function to list available keys for each file. An outline of the image object structure is show below:

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
    """ Open panel file and return instance of the panel data object. Object includes summary function to list available dictionary keys.

    :param file: Path ot panel file.
    :type file: str
  
    :return: Holography panel object.
    :rtype: AstrohackPanelFile

    .. _Description:
    **AstrohackPanelFile**
    Panel object allows the user to access panel data via compound dictionary keys with values, in order of depth, `ant` -> `ddi`. The panel object also provides a `summary()` helper function to list available keys for each file. An outline of the panel object structure is show below:

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


def open_pointing(file):
    """ Open pointing file and return instance of the pointing data object. Object includes summary function to list available dictionary keys.

    :param file: Path to pointing file.
    :type file: str
  
    :return: Holography pointing object.
    :rtype: AstrohackPointFile

    .. _Description:

    **AstrohackPointFile**
    Pointing object allows the user to access pointing data via dictionary key with value based on `ant`. The pointing object also provides a `summary()` helper function to list available keys for each file. An outline of the pointing object structure is show below:

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
        logger.error(f"Error opening holgraphy pointing file: {file}")


def fix_pointing_table(ms_name, reference_antenna):
    """ Fix pointing table for a user defined subset of reference antennas.

  Args:
      ms_name (str): Measurement set.
      reference_antenna (list): List of reference antennas.
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

