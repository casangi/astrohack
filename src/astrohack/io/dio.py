from astrohack.io.beamcut_mds import AstrohackBeamcutFile
from astrohack.io.holog_mds import AstrohackHologFile
from astrohack.io.image_mds import AstrohackImageFile
from astrohack.io.locit_mds import AstrohackLocitFile
from astrohack.io.panel_mds import AstrohackPanelFile
from astrohack.io.point_mds import AstrohackPointFile
from astrohack.io.position_mds import AstrohackPositionFile
from astrohack.utils.constants import minimum_xdtree_version
from astrohack.utils.file import (
    check_if_file_can_be_opened,
)


def open_beamcut(file: str) -> AstrohackBeamcutFile | None:
    """ Open beamcut file and return instance of the beamcut data object. Object includes summary function to list\
     available nodes.

    :param file: Path to beamcut file.
    :type file: str

    :return: beamcut object; None if file not found.
    :rtype: AstrohackBeamcutFile

    .. _Description:
    **AstrohackBeamcutFile**
    Beamcu object allows the user to access beam cut data via a xarray data tree, in order of depth, `ant` -> `ddi` \
    -> `cut`. The beamcut object also provides a `summary()` helper function to list available nodes for each file. \
    An outline of the beam object structure is show below:

    .. parsed-literal::
        beamcut_mds =
            {
                ant_0:{
                    ddi_0:{
                         cut_0: beamcut_ds,
                             ⋮
                         cut_n: beamcut_ds
                    },
                    ⋮
                    ddi_n: …
                },
                ⋮
                ant_n: …
            }
    """
    check_if_file_can_be_opened(file, "beamcut", minimum_xdtree_version)
    _data_file = AstrohackBeamcutFile(file=file)

    if _data_file.open():
        return _data_file
    else:
        return None


def open_holog(file: str) -> AstrohackHologFile | None:
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
                ant_0:{
                    ddi_0:{
                         map_0: holog_ds,
                             ⋮
                         map_n: holog_ds
                    },
                    ⋮
                    ddi_p: …
                },
            ⋮
            ant_m: …
            }
    """
    check_if_file_can_be_opened(file, "extract_holog", minimum_xdtree_version)
    _data_file = AstrohackHologFile(file=file)

    if _data_file.open():
        return _data_file
    else:
        return None


def open_image(file: str) -> AstrohackImageFile | None:
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
    check_if_file_can_be_opened(file, ["holog", "combine"], minimum_xdtree_version)
    _data_file = AstrohackImageFile(file=file)

    if _data_file.open():
        return _data_file
    else:
        return None


def open_panel(file: str) -> AstrohackPanelFile | None:
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
    check_if_file_can_be_opened(file, "panel", minimum_xdtree_version)
    _data_file = AstrohackPanelFile(file=file)

    if _data_file.open():
        return _data_file
    else:
        return None


def open_locit(file: str) -> AstrohackLocitFile | None:
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
    check_if_file_can_be_opened(file, "extract_locit", minimum_xdtree_version)
    _data_file = AstrohackLocitFile(file=file)

    if _data_file.open():
        return _data_file
    else:
        return None


def open_position(file: str) -> AstrohackPositionFile | None:
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
    check_if_file_can_be_opened(
        file, ["locit", "fringefit_locit"], minimum_xdtree_version
    )
    _data_file = AstrohackPositionFile(file=file)

    if _data_file.open():
        return _data_file

    else:
        return None


def open_pointing(file: str) -> AstrohackPointFile | None:
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
    check_if_file_can_be_opened(file, "extract_pointing", minimum_xdtree_version)
    _data_file = AstrohackPointFile(file=file)

    if _data_file.open():
        return _data_file

    else:
        return None
