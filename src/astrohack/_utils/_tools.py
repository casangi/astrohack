import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger


def _well_positioned_colorbar(ax, fig, image, label, location='right', size='5%', pad=0.05):
    """
    Adds a well positioned colorbar to a plot
    Args:
        ax: Axes instance to add the colorbar
        fig: Figure in which the axes are embedded
        image: The plt.imshow instance associated to the colorbar
        label: Colorbar label
        location: Colorbar location
        size: Colorbar size
        pad: Colorbar padding

    Returns: the well positioned colorbar

    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    return fig.colorbar(image, label=label, cax=cax)


def _remove_suffix(input_string, suffix):
    """
    Removes extension suffixes from file names
    Args:
        input_string: filename string
        suffix: The suffix to be removed

    Returns: the input string minus suffix

    """
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def _jsonify(holog_obj):
    """ Convert holog_obs_description dictionay to json format. This just means converting numpy.ndarry
        entries to string lists.

    :param holog_obj: holog_obs_description dictionary.
    :type holog_obj: dict
    :param holog_obj: holog_obs_description dictionary.
    :type holog_obj: dict
    """
    for ddi_key, ddi_value in holog_obj.items():
        for map_key, map_value in holog_obj[ddi_key].items():
            for attr_key, attr_value in holog_obj[ddi_key][map_key].items():
                if "scans" in attr_key:
                    holog_obj[ddi_key][map_key][attr_key] = list(map(str, attr_value))
                
                elif "ant" in attr_key:
                    for ant_key, ant_value in holog_obj[ddi_key][map_key][attr_key].items():
                        holog_obj[ddi_key][map_key][attr_key][ant_key] = list(map(str, ant_value))

                else:
                    pass


def _add_prefix(input_string, prefix):
    """
    Adds a prefix to a string filename, if the filename is a path with /, adds the prefix to the actual filename at the
    end of the path
    Args:
        input_string: filename or file path
        prefix: prefix to be added to the filename

    Returns: filename or path plus prefix added to the filename

    """
    wrds = input_string.split('/')
    wrds[-1] = prefix+'_'+wrds[-1]
    return '/'.join(wrds)


def _print_holog_obs_dict(holog_obj):
    OPEN_DICT  = ":{"
    CLOSE_DICT = "}"
    
    OPEN_LIST  = ":["
    CLOSE_LIST = "]"

    print("\n\n| ********************************************************** |")
    print("|                 HOLOG OBSERVATION DICTIONARY               |")
    print("| ********************************************************** |\n\n")
    
    for ddi_key, ddi_value in holog_obj.items():
        print("{ddi_key} {open_bracket}".format(ddi_key=ddi_key, open_bracket=OPEN_DICT))
        for map_key, map_value in holog_obj[ddi_key].items():
            print("{map_key: >10} {open_bracket}".format(map_key=map_key, open_bracket=OPEN_DICT))
            for attr_key, attr_value in holog_obj[ddi_key][map_key].items():
                if "scans" in attr_key:
                    print("{attr_key: >12}: {open_list}".format(attr_key=attr_key, open_list=OPEN_LIST))
    
                    scan_list = ", ".join(list(map(str, holog_obj[ddi_key][map_key][attr_key])))
                    print("{scan: >18}".format(scan=scan_list))                                   # The print just ification in notebook is weird on this and seems to move according to list length ...
                    print("{close_bracket: >10}".format(close_bracket=CLOSE_LIST))
                
                elif "ant" in attr_key:
                    print("{attr_key: >12} {open_bracket}".format(attr_key=attr_key, open_bracket=OPEN_DICT))
                    for ant_key, ant_value in holog_obj[ddi_key][map_key][attr_key].items():
                        print("{ant_key: >18} {open_list}".format(ant_key=ant_key, open_list=OPEN_LIST))
                        print("{antenna: >25}".format( antenna=", ".join(holog_obj[ddi_key][map_key][attr_key]) ))
                        print("{close_list: >15}".format(close_list=CLOSE_LIST))
                    
                    print("{close_bracket: >10}".format(close_bracket=CLOSE_DICT))

                else:
                    pass
        print("{close_bracket: >5}".format(close_bracket=CLOSE_DICT))
        
    print("{close_bracket}".format(close_bracket=CLOSE_DICT))


def _parm_to_list(parm, path, prefix):
    """
    Transforms a string parameter to a list if parameter is all or a single string
    Args:
        parm: string or list parameter
        path: Path to complete parameter with values if parameter is 'all'

    Returns: parameter value converter to a list

    """
    if parm == 'all':
        tmplist = os.listdir(path)
        oulist = []
        for item in tmplist:
            if item.find(prefix) == 0:
                oulist.append(item)
    elif isinstance(parm, str):
        oulist = [parm]
    else:
        oulist = parm
    return oulist


def _numpy_to_json(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)


def _split_pointing_table(ms_name, antennas):
    """ Split pointing table to contain only specified antennas

    :param ms_name: Measurement file
    :type ms_name: str
    :param antennas: List of antennas to sub-select on.
    :type antennas: list (str)
    """

    # Need to get thea antenna-id values for teh input antenna names. This is not available in the POINTING table
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


def _stokes_axis_to_fits_header(header, iaxis):
    """
    Inserts a dedicated stokes axis in the header at iaxis
    Args:
        header: The header to add the axis description to
        iaxis: The position of the axis in the data

    Returns: The augmented header

    """
    header[f'NAXIS{iaxis}'] = 4
    header[f'CRVAL{iaxis}'] = 1.0
    header[f'CDELT{iaxis}'] = 1.0
    header[f'CRPIX{iaxis}'] = 1.0
    header[f'CROTA{iaxis}'] = 0.
    header[f'CTYPE{iaxis}'] = 'STOKES'
    header[f'CUNIT{iaxis}'] = ''

    return header


def _axis_to_fits_header(header, axis, iaxis, axistype, unit):
    """
    Process an axis to create a FITS compatible linear axis description
    Args:
        header: The header to add the axis description to
        axis: The axis to be described in the header
        iaxis: The position of the axis in the data
        axistype: Axis type to be displayed in the fits header

    Returns: The augmented header

    """
    logger = _get_astrohack_logger()
    naxis = len(axis)
    if naxis == 1:
        inc = axis[0]
    else:
        inc = axis[1] - axis[0]
        if inc == 0:
            logger.error('Axis increment is zero valued')
            raise Exception
        absdiff = abs((axis[-1]-axis[-2])-inc)/inc
        if absdiff > 1e-7:
            logger.error('Axis is not linear!')
            raise Exception

    ref = naxis//2
    val = axis[ref]

    header[f'NAXIS{iaxis}'] = naxis
    header[f'CRVAL{iaxis}'] = val
    header[f'CDELT{iaxis}'] = inc
    header[f'CRPIX{iaxis}'] = ref
    header[f'CROTA{iaxis}'] = 0.
    header[f'CTYPE{iaxis}'] = axistype
    header[f'CUNIT{iaxis}'] = unit
    return header


def _bool_to_string(flag):
    """
    Converts a boolean to a yes or no string
    Args:
        flag: boolean to be converted to string

    Returns: 'yes' or 'no'

    """
    if flag:
        return 'yes'
    else:
        return 'no'


