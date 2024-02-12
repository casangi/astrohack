import os
import zarr
import shutil
import pathlib

import numpy as np
import xarray as xr

import graphviper.utils.logger as logger


def _check_if_file_exists(file):
    if pathlib.Path(file).exists() is False:
        logger.error('File {file} does not exists.')
        raise FileNotFoundError


def _check_if_file_will_be_overwritten(file, overwrite):
    path = pathlib.Path(file)

    if (path.exists() is True) and (overwrite is False):
        logger.error(f'{file} already exists. To overwrite set overwrite to True, or remove current file.')

        raise FileExistsError("{file} exists.".format(file=file))

    elif (path.exists() is True) and (overwrite is True):
        if file.endswith(".zarr"):
            logger.warning(f'{file} will be overwritten.')
            shutil.rmtree(file)
        else:
            logger.warning(f'{file} may not be valid hack file. Check the file name again.')
            raise Exception(f"IncorrectFileType: {file}")


def _create_destination_folder(destination):
    """
    Try to create a folder if it already exists raise a warning
    Args:
        destination: the folder to be created
    """

    try:
        os.mkdir(destination)
    except FileExistsError:
        logger.warning(f'Destination folder already exists, results may be overwritten')


def _aips_holog_to_xds(ampname, devname):
    """
    Read amplitude and deviation FITS files onto a common Xarray dataset
    Args:
        ampname: Name of the amplitude FITS file
        devname: Name of the deviation FITS file

    Returns:
    Xarray dataset
    """
    amphead, ampdata = _read_fits(ampname)
    devhead, devdata = _read_fits(devname)
    ampdata = np.flipud(ampdata)
    devdata = np.flipud(devdata)

    if amphead["NAXIS1"] != devhead["NAXIS1"]:
        raise Exception(ampname + ' and ' + devname + ' have different dimensions')
    if amphead["CRPIX1"] != devhead["CRPIX1"] or amphead["CRVAL1"] != devhead["CRVAL1"] \
            or amphead["CDELT1"] != devhead["CDELT1"]:
        raise Exception(ampname + ' and ' + devname + ' have different axes descriptions')

    npoint, wavelength = _get_aips_headpars(devhead)
    u = np.arange(-amphead["CRPIX1"], amphead["NAXIS1"] - amphead["CRPIX1"]) * amphead["CDELT1"]
    v = np.arange(-amphead["CRPIX2"], amphead["NAXIS2"] - amphead["CRPIX2"]) * amphead["CDELT2"]

    xds = xr.Dataset()
    xds.attrs['npix'] = amphead["NAXIS1"]
    xds.attrs['cell_size'] = amphead["CDELT1"]
    xds.attrs['ref_pixel'] = amphead["CRPIX1"]
    xds.attrs['ref_value'] = amphead["CRVAL1"]
    xds.attrs['npoint'] = npoint
    xds.attrs['wavelength'] = wavelength
    xds.attrs['amp_unit'] = amphead["BUNIT"].strip()
    xds.attrs['AIPS'] = True
    xds.attrs['ant_name'] = amphead["TELESCOP"].strip()
    xds['AMPLITUDE'] = xr.DataArray(ampdata, dims=["u", "v"])
    xds['DEVIATION'] = xr.DataArray(devdata, dims=["u", "v"])
    coords = {"u": u, "v": v}
    xds = xds.assign_coords(coords)
    return xds


def _get_aips_headpars(head):
    """
    Fetch number of points used in holography and wavelength stored by AIPS on a FITS header
    Args:
        head: AIPS FITS header

    Returns:
    npoint, wavelength
    """
    npoint = np.nan
    wavelength = np.nan
    for line in head["HISTORY"]:
        wrds = line.split()
        if wrds[1] == "Visibilities":
            npoint = np.sqrt(int(wrds[-1]))
        elif wrds[1] == "Observing":
            wavelength = float(wrds[-2])
    return npoint, wavelength


def _reshape(array, columns):
    size = len(array)
    rows = int(size / columns)
    if rows <= 0:
        return 1, 0
    else:
        remainder = size - (rows * columns)

        return rows, remainder


def _print_array(array, columns, indent=4):
    rows, remainder = _reshape(array, columns)

    if columns > len(array):
        columns = len(array)

    str_line = ""

    for i in range(rows):
        temp = []
        for j in range(columns):
            k = columns * i + j
            if j == 0:
                temp.append("{:>3}".format(array[k]).rjust(indent, ' '))
            else:
                temp.append("{:>3}".format(array[k]))

        str_line += ", ".join(temp)
        str_line += "\n"

    temp = []
    if remainder > 0:
        for i in range(remainder):
            index = columns * rows + i

            if i == 0:
                temp.append("{:>3}".format(array[index]).rjust(indent, ' '))
            else:
                temp.append("{:>3}".format(array[index]))

        str_line += ", ".join(temp)

    print(str_line)
