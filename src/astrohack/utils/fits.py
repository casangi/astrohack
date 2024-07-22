import xarray as xr
from graphviper.utils import logger as logger

import astrohack
import datetime
import numpy as np

from astropy.io import fits

from astrohack.utils import clight, convert_unit
from astrohack.utils.text import add_prefix


def read_fits(filename):
    """
    Reads a square FITS file and do sanity checks on its dimensionality
    Args:
        filename: a string containing the FITS file name/path

    Returns:
    The FITS header and the associated data array
    """
    hdul = fits.open(filename)
    head = hdul[0].header
    data = hdul[0].data[0, 0, :, :]
    hdul.close()
    if head["NAXIS"] != 1:
        if head["NAXIS"] < 1:
            raise Exception(filename + " is not bi-dimensional")
        elif head["NAXIS"] > 1:
            for iax in range(2, head["NAXIS"]):
                if head["NAXIS" + str(iax + 1)] != 1:
                    raise Exception(filename + " is not bi-dimensional")
    if head["NAXIS1"] != head["NAXIS2"]:
        raise Exception(filename + " does not have the same amount of pixels in the x and y axes")
    return head, data


def write_fits(header, imagetype, data, filename, unit, origin):
    """
    Write a dictionary and a dataset to a FITS file
    Args:
        header: The dictionary containing the header
        imagetype: Type to be added to FITS header
        data: The dataset
        filename: The name of the output file
        unit: to be set to bunit
        origin: Which astrohack mds has created the FITS being written
    """

    header['BUNIT'] = unit
    header['TYPE'] = imagetype
    header['ORIGIN'] = f'Astrohack v{astrohack.__version__}: {origin}'
    header['DATE'] = datetime.datetime.now().strftime('%b %d %Y, %H:%M:%S')

    hdu = fits.PrimaryHDU(_reorder_axes_for_fits(data))
    for key in header.keys():
        hdu.header.set(key, header[key])
    hdu.writeto(add_prefix(filename, origin), overwrite=True)

    return


def _reorder_axes_for_fits(data: np.ndarray):
    carta_dim_order = (1, 0, 2, 3,)
    shape = data.shape
    n_dim = len(shape)
    if n_dim == 5:
        # Ignore the time dimension and flip polarization and frequency axes
        transpo = np.transpose(data[0, ...], carta_dim_order)
        # Flip vertical axis
        return np.flip(transpo, 2)
    elif n_dim == 2:
        return np.flipud(data)


def resolution_to_fits_header(header, resolution):
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
        header['BMAJ'] = resolution[0]
        header['BMIN'] = resolution[1]
        header['BPA'] = 0.0
    else:
        header['BMAJ'] = resolution[1]
        header['BMIN'] = resolution[0]
        header['BPA'] = 90.0
    return header


def axis_to_fits_header(header: dict, axis, iaxis, axistype, unit, iswcs=True):
    """
    Process an axis to create a FITS compatible linear axis description
    Args:
        header: The header to add the axis description to
        axis: The axis to be described in the header
        iaxis: The position of the axis in the data
        axistype: Axis type to be displayed in the fits header

    Returns: The augmented header

    """
    outheader = header.copy()
    try:
        wcsaxes = outheader['WCSAXES']
    except KeyError:
        wcsaxes = 0

    naxis = len(axis)
    if naxis == 1:
        inc = axis[0]
    else:
        inc = axis[1] - axis[0]
        if inc == 0:
            logger.error('Axis increment is zero valued')
            raise Exception
        absdiff = abs((axis[-1] - axis[-2]) - inc) / inc
        if absdiff > 1e-7:
            logger.error('Axis is not linear!')
            raise Exception

    ref = naxis // 2
    val = axis[ref]

    if iswcs:
        wcsaxes += 1
    outheader[f'WCSAXES'] = wcsaxes
    outheader[f'NAXIS{iaxis}'] = naxis
    outheader[f'CRVAL{iaxis}'] = val - inc
    outheader[f'CDELT{iaxis}'] = inc
    outheader[f'CRPIX{iaxis}'] = float(ref)
    outheader[f'CROTA{iaxis}'] = 0.
    outheader[f'CTYPE{iaxis}'] = axistype
    outheader[f'CUNIT{iaxis}'] = unit
    return outheader


def stokes_axis_to_fits_header(header, iaxis):
    """
    Inserts a dedicated stokes axis in the header at iaxis
    Args:
        header: The header to add the axis description to
        iaxis: The position of the axis in the data

    Returns: The augmented header

    """
    outheader = header.copy()
    try:
        wcsaxes = outheader['WCSAXES']
    except KeyError:
        wcsaxes = 0
    wcsaxes += 1
    outheader[f'WCSAXES'] = wcsaxes
    outheader[f'NAXIS{iaxis}'] = 4
    outheader[f'CRVAL{iaxis}'] = 1.0
    outheader[f'CDELT{iaxis}'] = 1.0
    outheader[f'CRPIX{iaxis}'] = 1.0
    outheader[f'CROTA{iaxis}'] = 0.
    outheader[f'CTYPE{iaxis}'] = 'STOKES'
    outheader[f'CUNIT{iaxis}'] = ''

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


def export_to_fits_holog_chunk(parm_dict):
    """
    Holog side chunk function for the user facing function export_to_fits
    Args:
        parm_dict: parameter dictionary
    """
    input_xds = parm_dict['xds_data']
    metadata = parm_dict['metadata']
    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    basename = f'{destination}/{antenna}_{ddi}'

    logger.info(f'Exporting image contents of {antenna} {ddi} to FITS files in {destination}')

    try:
        aperture_resolution = input_xds.attrs["aperture_resolution"]

    except KeyError:
        logger.warning("Holog image does not have resolution information")
        logger.warning("Rerun holog with astrohack v>0.1.5 for aperture resolution information")

        aperture_resolution = None

    nchan = len(input_xds.chan)

    if nchan == 1:
        reffreq = input_xds.chan.values[0]

    else:
        reffreq = input_xds.chan.values[nchan // 2]

    telname = input_xds.attrs['telescope_name']

    if telname in ['EVLA', 'VLA', 'JVLA']:
        telname = 'VLA'

    polist = []

    for pol in input_xds.pol:
        polist.append(str(pol.values))

    base_header = {
        'STOKES': ", ".join(polist),
        'WAVELENG': clight / reffreq,
        'FREQUENC': reffreq,
        'TELESCOP': input_xds.attrs['ant_name'],
        'INSTRUME': telname,
        'TIME_CEN': input_xds.attrs['time_centroid'],
        'PADDING': metadata['padding_factor'],
        'GRD_INTR': metadata['grid_interpolation_mode'],
        'CHAN_AVE': "yes" if metadata['chan_average'] is True else "no",
        'CHAN_TOL': metadata['chan_tolerance_factor'],
        'SCAN_AVE': "yes" if metadata['scan_average'] is True else "no",
        'TO_STOKE': "yes" if metadata['to_stokes'] is True else "no",
    }

    ntime = len(input_xds.time)
    if ntime != 1:
        raise Exception("Data with multiple times not supported for FITS export")

    base_header = axis_to_fits_header(base_header, input_xds.chan.values, 3, 'Frequency', 'Hz')
    base_header = stokes_axis_to_fits_header(base_header, 4)
    rad_to_deg = convert_unit('rad', 'deg', 'trigonometric')
    beam_header = axis_to_fits_header(base_header, -input_xds.l.values * rad_to_deg, 1, 'RA---SIN', 'deg')
    beam_header = axis_to_fits_header(beam_header, input_xds.m.values * rad_to_deg, 2, 'DEC--SIN', 'deg')
    beam_header['RADESYSA'] = 'FK5'
    beam = input_xds['BEAM'].values
    if parm_dict['complex_split'] == 'cartesian':
        write_fits(beam_header, 'Complex beam real part', beam.real, add_prefix(basename, 'beam_real') + '.fits',
                   'Normalized', 'image')
        write_fits(beam_header, 'Complex beam imag part', beam.imag, add_prefix(basename, 'beam_imag') + '.fits',
                   'Normalized', 'image')
    else:
        write_fits(beam_header, 'Complex beam amplitude', np.absolute(beam),
                   add_prefix(basename, 'beam_amplitude') + '.fits', 'Normalized', 'image')
        write_fits(beam_header, 'Complex beam phase', np.angle(beam),
                   add_prefix(basename, 'beam_phase') + '.fits', 'Radians', 'image')
    wavelength = clight / input_xds.chan.values[0]
    aperture_header = axis_to_fits_header(base_header, input_xds.u.values * wavelength, 1, 'X----LIN', 'm')
    aperture_header = axis_to_fits_header(aperture_header, input_xds.u.values * wavelength, 2, 'Y----LIN', 'm')
    aperture_header = resolution_to_fits_header(aperture_header, aperture_resolution)
    aperture = input_xds['APERTURE'].values
    if parm_dict['complex_split'] == 'cartesian':
        write_fits(aperture_header, 'Complex aperture real part', aperture.real,
                   add_prefix(basename, 'aperture_real') + '.fits', 'Normalized', 'image')
        write_fits(aperture_header, 'Complex aperture imag part', aperture.imag,
                   add_prefix(basename, 'aperture_imag') + '.fits', 'Normalized', 'image')
    else:
        write_fits(aperture_header, 'Complex aperture amplitude', np.absolute(aperture),
                   add_prefix(basename, 'aperture_amplitude') + '.fits', 'Normalized', 'image')
        write_fits(aperture_header, 'Complex aperture phase', np.angle(aperture),
                   add_prefix(basename, 'aperture_phase') + '.fits', 'rad', 'image')

    phase_amp_header = axis_to_fits_header(base_header, input_xds.u_prime.values * wavelength, 1, 'X----LIN', 'm')
    phase_amp_header = axis_to_fits_header(phase_amp_header, input_xds.v_prime.values * wavelength, 2, 'Y----LIN', 'm')
    phase_amp_header = resolution_to_fits_header(phase_amp_header, aperture_resolution)
    write_fits(phase_amp_header, 'Cropped aperture corrected phase', input_xds['CORRECTED_PHASE'].values,
               add_prefix(basename, 'corrected_phase') + '.fits', 'rad', 'image')
    return
