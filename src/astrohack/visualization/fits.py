import numpy as np
from toolviper.utils import logger as logger
from astrohack.antenna import Telescope, AntennaSurface
from astrohack.utils import clight, convert_unit, add_prefix
from astrohack.utils.fits import axis_to_fits_header, stokes_axis_to_fits_header, write_fits, resolution_to_fits_header


def export_to_fits_panel_chunk(parm_dict):
    """
    Panel side chunk function for the user facing function export_to_fits
    Args:
        parm_dict: parameter dictionary
    """

    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    logger.info(f'Exporting panel contents of {antenna} {ddi} to FITS files in {destination}')
    xds = parm_dict['xds_data']
    telescope = Telescope(xds.attrs['telescope_name'])
    surface = AntennaSurface(xds, telescope, reread=True)
    basename = f'{destination}/{antenna}_{ddi}'
    surface.export_to_fits(basename)
    return


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
