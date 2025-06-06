import xarray as xr
from toolviper.utils import logger as logger

import astrohack
import datetime
import numpy as np

from astropy.io import fits

from astrohack.utils.text import add_prefix


def get_stokes_axis_iaxis(header):
    """
    Get which of the axis in the header is the stokes axis
    Args:
        header: FITS header

    Returns:
        None if no stokes axis is found, iaxis if stokes axis is found
    """
    naxis = header["NAXIS"]
    for iaxis in range(naxis):
        axis_type = safe_keyword_fetch(header, f"CTYPE{iaxis+1}")
        if "STOKES" in axis_type:
            return iaxis + 1
    return None


def safe_keyword_fetch(header_dict, keyword):
    """
    Tries to fetch a keyword from a FITS header / dictionary
    Args:
        header_dict: FITS header / Dictionary
        keyword: The intended keyword to fetch

    Returns:
        Keyword value if prensent, None if not present.
    """
    try:
        return header_dict[keyword]
    except KeyError:
        return None


def read_fits(filename, header_as_dict=True):
    """
    Reads a square FITS file and do sanity checks on its dimensionality
    Args:
        filename: a string containing the FITS file name/path

    Returns:
    The FITS header and the associated data array
    """
    hdul = fits.open(filename)
    head = hdul[0].header
    data = hdul[0].data
    hdul.close()
    if head["NAXIS"] != 1:
        if head["NAXIS"] < 1:
            raise Exception(filename + " is not bi-dimensional")
        elif head["NAXIS"] > 1:
            for iax in range(2, head["NAXIS"]):
                if head["NAXIS" + str(iax + 1)] != 1:
                    raise Exception(filename + " is not bi-dimensional")
    if head["NAXIS1"] != head["NAXIS2"]:
        raise Exception(
            filename + " does not have the same amount of pixels in the x and y axes"
        )

    if header_as_dict:
        header_dict = {}
        for key, value in head.items():
            header_dict[key] = value
        return header_dict, data
    else:
        return head, data


def get_axis_from_fits_header(header, iaxis, pixel_offset=True):
    """
    Pull axis information from FITS file and store it in a numpy array, ignores rotation in axes.
    Args:
        header: FITS header
        iaxis: Which axis is to be fetched from the header.
        pixel_offset: apply one pixel offset

    Returns:
        numpy array representation of axis, axis type and axis unit
    """
    n_elem = header[f"NAXIS{iaxis}"]
    ref = header[f"CRPIX{iaxis}"]
    inc = header[f"CDELT{iaxis}"]
    if pixel_offset:
        val = (
            header[f"CRVAL{iaxis}"] + inc
        )  # This makes this routine symmetrical to the put routine.
    else:
        val = header[f"CRVAL{iaxis}"]
    axis = np.arange(n_elem)
    axis = val + (ref - axis) * inc
    axis_unit = safe_keyword_fetch(header, f"CUNIT{iaxis}")
    axis_type = safe_keyword_fetch(header, f"CTYPE{iaxis}")
    return axis, axis_type, axis_unit


def write_fits(header, imagetype, data, filename, unit, origin=None, reorder_axis=True):
    """
    Write a dictionary and a dataset to a FITS file
    Args:
        header: The dictionary containing the header
        imagetype: Type to be added to FITS header
        data: The dataset
        filename: The name of the output file
        unit: to be set to bunit
        origin: Which astrohack mds has created the FITS being written
        reorder_axis: Reorder data axes so that they are compatible with regular FITS ordering
    """

    header["BUNIT"] = unit
    header["TYPE"] = imagetype
    header["ORIGIN"] = f"Astrohack v{astrohack.__version__}: {origin}"
    header["DATE"] = datetime.datetime.now().strftime("%b %d %Y, %H:%M:%S")

    if origin is None:
        header["ORIGIN"] = f"Astrohack v{astrohack.__version__}"
        outfile = filename
    else:
        header["ORIGIN"] = f"Astrohack v{astrohack.__version__}: {origin}"
        outfile = add_prefix(filename, origin)

    if reorder_axis:
        hdu = fits.PrimaryHDU(_reorder_axes_for_fits(data))
    else:
        hdu = fits.PrimaryHDU(data)
    for key in header.keys():
        hdu.header.set(key, header[key])
    hdu.writeto(outfile, overwrite=True)

    return


def _reorder_axes_for_fits(data: np.ndarray):
    carta_dim_order = (
        1,
        0,
        2,
        3,
    )
    shape = data.shape
    n_dim = len(shape)
    if n_dim == 5:
        # Ignore the time dimension and flip polarization and frequency axes
        transpo = np.transpose(data[0, ...], carta_dim_order)
        # Flip vertical axis
        return np.flip(transpo, 2)
    elif n_dim == 2:
        return np.flipud(data)


def put_resolution_in_fits_header(header, resolution):
    """
    Adds resolution information to standard header keywords: BMAJ, BMIN and BPA
    Args:
        header: The dictionary header to be augmented
        resolution: The lenght=2 array with the resolution elements

    Returns: The augmented header dictionary
    """
    if resolution is None:
        return header
    if resolution[0] >= resolution[1]:
        header["BMAJ"] = resolution[0]
        header["BMIN"] = resolution[1]
        header["BPA"] = 0.0
    else:
        header["BMAJ"] = resolution[1]
        header["BMIN"] = resolution[0]
        header["BPA"] = 90.0
    return header


def put_axis_in_fits_header(header: dict, axis, iaxis, axistype, unit, iswcs=True):
    """
    Process an axis to create a FITS compatible linear axis description
    Args:
        header: The header to add the axis description to
        axis: The axis to be described in the header
        iaxis: The position of the axis in the data
        axistype: Axis type to be displayed in the fits header
        unit: Axis unit
        iswcs: Is the axis a part of World Coordinate System for the image?

    Returns: The augmented header

    """
    outheader = header.copy()
    try:
        wcsaxes = outheader["WCSAXES"]
    except KeyError:
        wcsaxes = 0

    naxis = len(axis)
    if naxis == 1:
        inc = axis[0]
    else:
        inc = axis[1] - axis[0]
        if inc == 0:
            logger.error("Axis increment is zero valued")
            raise Exception
        absdiff = abs((axis[-1] - axis[-2]) - inc) / inc
        if absdiff > 1e-7:
            logger.error("Axis is not linear!")
            raise Exception

    ref = naxis // 2
    val = axis[ref]

    if iswcs:
        wcsaxes += 1
    outheader[f"WCSAXES"] = wcsaxes
    outheader[f"NAXIS{iaxis}"] = naxis
    outheader[f"CRVAL{iaxis}"] = val - inc
    outheader[f"CDELT{iaxis}"] = inc
    outheader[f"CRPIX{iaxis}"] = float(ref)
    outheader[f"CROTA{iaxis}"] = 0.0
    outheader[f"CTYPE{iaxis}"] = axistype
    outheader[f"CUNIT{iaxis}"] = unit
    return outheader


def put_stokes_axis_in_fits_header(header, iaxis):
    """
    Inserts a dedicated stokes axis in the header at iaxis
    Args:
        header: The header to add the axis description to
        iaxis: The position of the axis in the data

    Returns: The augmented header

    """
    outheader = header.copy()
    try:
        wcsaxes = outheader["WCSAXES"]
    except KeyError:
        wcsaxes = 0
    wcsaxes += 1
    outheader[f"WCSAXES"] = wcsaxes
    outheader[f"NAXIS{iaxis}"] = 4
    outheader[f"CRVAL{iaxis}"] = 1.0
    outheader[f"CDELT{iaxis}"] = 1.0
    outheader[f"CRPIX{iaxis}"] = 1.0
    outheader[f"CROTA{iaxis}"] = 0.0
    outheader[f"CTYPE{iaxis}"] = "STOKES"
    outheader[f"CUNIT{iaxis}"] = ""

    return outheader


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


def aips_holog_to_xds(ampname, devname):
    """
    Read amplitude and deviation FITS files onto a common Xarray dataset
    Args:
        ampname: Name of the amplitude FITS file
        devname: Name of the deviation FITS file

    Returns:
    Xarray dataset
    """
    amphead, ampdata = read_fits(ampname)
    devhead, devdata = read_fits(devname)
    ampdata = np.flipud(ampdata)
    devdata = np.flipud(devdata)

    if amphead["NAXIS1"] != devhead["NAXIS1"]:
        raise Exception(ampname + " and " + devname + " have different dimensions")
    if (
        amphead["CRPIX1"] != devhead["CRPIX1"]
        or amphead["CRVAL1"] != devhead["CRVAL1"]
        or amphead["CDELT1"] != devhead["CDELT1"]
    ):
        raise Exception(
            ampname + " and " + devname + " have different axes descriptions"
        )

    npoint, wavelength = _get_aips_headpars(devhead)
    u = (
        np.arange(-amphead["CRPIX1"], amphead["NAXIS1"] - amphead["CRPIX1"])
        * amphead["CDELT1"]
    )
    v = (
        np.arange(-amphead["CRPIX2"], amphead["NAXIS2"] - amphead["CRPIX2"])
        * amphead["CDELT2"]
    )

    xds = xr.Dataset()
    xds.attrs["npix"] = amphead["NAXIS1"]
    xds.attrs["cell_size"] = amphead["CDELT1"]
    xds.attrs["ref_pixel"] = amphead["CRPIX1"]
    xds.attrs["ref_value"] = amphead["CRVAL1"]
    xds.attrs["npoint"] = npoint
    xds.attrs["wavelength"] = wavelength
    xds.attrs["amp_unit"] = amphead["BUNIT"].strip()
    xds.attrs["AIPS"] = True
    xds.attrs["ant_name"] = amphead["TELESCOP"].strip()
    xds["AMPLITUDE"] = xr.DataArray(ampdata, dims=["u", "v"])
    xds["DEVIATION"] = xr.DataArray(devdata, dims=["u", "v"])
    coords = {"u": u, "v": v}
    xds = xds.assign_coords(coords)
    return xds
