import numpy as np
import xarray as xr

from scipy.interpolate import griddata

from astrohack._utils._panel_classes.telescope import Telescope

from astrohack._utils._dio import _load_holog_file
from astrohack._utils._dio import _read_meta_data, _write_fits

from astrohack._utils._phase_fitting import _phase_fitting_block

from astrohack._utils._algorithms import _chunked_average
from astrohack._utils._algorithms import _find_peak_beam_value
from astrohack._utils._algorithms import _find_nearest
from astrohack._utils._algorithms import _calc_coords

from astrohack._utils._conversion import _to_stokes
from astrohack._utils._constants import clight
from astrohack._utils._tools import _bool_to_string, _axis_to_fits_header, _stokes_axis_to_fits_header, \
    _resolution_to_fits_header, _add_prefix
from astrohack._utils._plot_commons import _well_positioned_colorbar

from astrohack._utils._imaging import _parallactic_derotation
from astrohack._utils._imaging import _mask_circular_disk
from astrohack._utils._imaging import _calculate_aperture_pattern

from astrohack._utils._panel import _get_correct_telescope_from_name
from astrohack._utils._panel_classes.antenna_surface import AntennaSurface
from astrohack._utils._plot_commons import _create_figure_and_axes, _close_figure, _get_proper_color_map
from astrohack._utils._conversion import _convert_unit

import skriba.logger as logger


def _holog_chunk(holog_chunk_params):
    """ Process chunk holography data along the antenna axis. Works with holography file to properly grid , normalize,
        average and correct data and returns the aperture pattern.

    Args:
        holog_chunk_params (dict): Dictionary containing holography parameters.
    """
    holog_file, ant_data_dict = _load_holog_file(
        holog_chunk_params["holog_name"],
        dask_load=False,
        load_pnt_dict=False,
        ant_id=holog_chunk_params["this_ant"],
        ddi_id=holog_chunk_params["this_ddi"]
    )

    meta_data = _read_meta_data(holog_chunk_params["holog_name"]+'/.holog_attr')

    # Calculate lm coordinates
    l, m = _calc_coords(holog_chunk_params["grid_size"], holog_chunk_params["cell_size"])
    
    grid_l, grid_m = list(map(np.transpose, np.meshgrid(l, m)))
        
    to_stokes = holog_chunk_params["to_stokes"]

    ddi = holog_chunk_params["this_ddi"]
    n_holog_map = len(ant_data_dict[ddi].keys())
    
    # For a fixed ddi the frequency axis should not change over holog_maps, consequently we only have to consider the
    # first holog_map.
    map0 = list(ant_data_dict[ddi].keys())[0]
    
    freq_chan = ant_data_dict[ddi][map0].chan.values
    n_chan = ant_data_dict[ddi][map0].dims["chan"]
    n_pol = ant_data_dict[ddi][map0].dims["pol"]
    grid_interpolation_mode = holog_chunk_params["grid_interpolation_mode"]
    
    if holog_chunk_params["chan_average"]:
        reference_scaling_frequency = np.mean(freq_chan)

        avg_chan_map, avg_freq = _create_average_chan_map(freq_chan, holog_chunk_params["chan_tolerance_factor"])
        
        # Only a single channel left after averaging.
        beam_grid = np.zeros((n_holog_map,) + (1, n_pol) + grid_l.shape, dtype=np.complex128)

    else:
        beam_grid = np.zeros((n_holog_map,) + (n_chan, n_pol) + grid_l.shape, dtype=np.complex128)

    time_centroid = []

    for holog_map_index, holog_map in enumerate(ant_data_dict[ddi].keys()):
        ant_xds = ant_data_dict[ddi][holog_map]

        # Todo: Add flagging code

        # Grid the data
        vis = ant_xds.VIS.values
        vis[vis == np.nan] = 0.0
        lm = ant_xds.DIRECTIONAL_COSINES.values
        weight = ant_xds.WEIGHT.values

        if holog_chunk_params["chan_average"]:
            vis_avg, weight_sum = _chunked_average(vis, weight, avg_chan_map, avg_freq)
            lm_freq_scaled = lm[:, :, None] * (avg_freq / reference_scaling_frequency)

            n_chan = avg_freq.shape[0]
            
            # Unavoidable for loop because lm change over frequency.
            for chan_index in range(n_chan):
                # Average scaled beams.
                beam_grid[holog_map_index, 0, :, :, :] = (beam_grid[holog_map_index, 0, :, :, :] +
                                                          np.moveaxis(griddata(lm_freq_scaled[:, :, chan_index],
                                                                               vis_avg[:, chan_index, :],
                                                                               (grid_l, grid_m), method=
                                                                               grid_interpolation_mode,
                                                                               fill_value=0.0), (2), (0)))
            # Averaging now complete
            n_chan = 1
            freq_chan = [np.mean(avg_freq)]
        else:
            beam_grid[holog_map_index, ...] = np.moveaxis(griddata(lm, vis, (grid_l, grid_m),
                                                                   method=grid_interpolation_mode,
                                                                   fill_value=0.0), (0, 1), (2, 3))

        time_centroid_index = ant_data_dict[ddi][holog_map].dims["time"] // 2
        time_centroid.append(ant_data_dict[ddi][holog_map].coords["time"][time_centroid_index].values)

        for chan in range(n_chan):  # Todo: Vectorize holog_map and channel axis
            try:
                xx_peak = _find_peak_beam_value(beam_grid[holog_map_index, chan, 0, ...], scaling=0.25)
                yy_peak = _find_peak_beam_value(beam_grid[holog_map_index, chan, 3, ...], scaling=0.25)
            except:
                center_pixel = np.array(beam_grid.shape[-2:])//2
                xx_peak = beam_grid[holog_map_index, chan, 0, center_pixel[0], center_pixel[1]]
                yy_peak = beam_grid[holog_map_index, chan, 3, center_pixel[0], center_pixel[1]]

            normalization = np.abs(0.5 * (xx_peak + yy_peak))
            
            if normalization == 0:
                logger.warning("Peak of zero found! Setting normalization to unity.")
                normalization = 1
                
            beam_grid[holog_map_index, chan, ...] /= normalization

    beam_grid = _parallactic_derotation(data=beam_grid, parallactic_angle_dict=ant_data_dict[ddi])

    ###############

    pol = ant_data_dict[ddi][holog_map].pol.values
    if to_stokes:
        beam_grid = _to_stokes(beam_grid, ant_data_dict[ddi][holog_map].pol.values)
        pol = ['I', 'Q', 'U', 'V']
    
    ###############
    
    if holog_chunk_params["scan_average"]:
        beam_grid = np.mean(beam_grid,axis=0)[None,...]
        time_centroid = np.mean(np.array(time_centroid))

    logger.info("Calculating aperture pattern ...")
    # Current bottleneck
    aperture_grid, u, v, uv_cell_size = _calculate_aperture_pattern(
        grid=beam_grid,
        delta=holog_chunk_params["cell_size"],
        padding_factor=holog_chunk_params["padding_factor"],
    )
    
    # Get telescope info
    ant_name = ant_data_dict[ddi][holog_map].attrs["antenna_name"]
    
    if ant_name.upper().__contains__('DV'):
        telescope_name = "_".join((meta_data['telescope_name'], 'DV'))

    elif ant_name.upper().__contains__('DA'):
        telescope_name = "_".join((meta_data['telescope_name'], 'DA'))
        
    elif ant_name.upper().__contains__('EA'):
        telescope_name = 'VLA'

    else:
        raise Exception("Antenna type not found: {name}".format(name=meta_data['ant_name']))
    
    telescope = Telescope(telescope_name)

    min_wavelength = clight/freq_chan[0]
    max_aperture_radius = (0.5*telescope.diam)/min_wavelength
    
    image_slice = aperture_grid[0, 0, 0, ...]
    center_pixel = np.array(image_slice.shape[0:2])//2

    # Factor of 1.1: Let's not be too aggressive
    radius_u = int(np.where(np.abs(u) < max_aperture_radius*1.1)[0].max() - center_pixel[0])
    radius_v = int(np.where(np.abs(v) < max_aperture_radius*1.1)[0].max() - center_pixel[1])
        
    if radius_v > radius_u:
        radius = radius_v
    else:
        radius = radius_u

    if holog_chunk_params['apply_mask']:
        # Masking Aperture image
        mask = _mask_circular_disk(
            center=None,
            radius=radius,
            array=aperture_grid,
        )

        aperture_grid = mask*aperture_grid

    start_cut = center_pixel - radius
    end_cut = center_pixel + radius

    amplitude = np.absolute(aperture_grid[..., start_cut[0]:end_cut[0], start_cut[1]:end_cut[1]])
    phase = np.angle(aperture_grid[..., start_cut[0]:end_cut[0], start_cut[1]:end_cut[1]])
    phase_corrected_angle = np.zeros_like(phase)
    u_prime = u[start_cut[0]:end_cut[0]]
    v_prime = v[start_cut[1]:end_cut[1]]

    phase_fit_par = holog_chunk_params["phase_fit"]
    if isinstance(phase_fit_par, bool):
        do_phase_fit = phase_fit_par
        do_pnt_off = True
        do_xy_foc_off = True
        do_z_foc_off = True
        do_cass_off = True
        if telescope.name == 'VLA' or telescope.name == 'VLBA':
            do_sub_til = True
        else:
            do_sub_til = False

    elif isinstance(phase_fit_par, (np.ndarray, list, tuple)):
        if len(phase_fit_par) != 5:
            raise Exception("Phase fit parameter must have 5 elements")

        else:
            if np.sum(phase_fit_par) == 0:
                do_phase_fit = False
            else:
                do_phase_fit = True
                do_pnt_off, do_xy_foc_off, do_z_foc_off, do_sub_til, do_cass_off = phase_fit_par
    
    else:
        raise Exception('Phase fit parameter is neither a boolean nor an array of booleans.')

    if do_phase_fit:
        logger.info('Applying phase correction')
        
        if to_stokes:
            pols = (0,)
        else:
            pols = (0, 3)

        max_wavelength = clight/freq_chan[-1]
        
        results, errors, phase_corrected_angle, _, in_rms, out_rms = _phase_fitting_block(
                    pols=pols,
                    wavelength=max_wavelength,
                    telescope=telescope,
                    cellxy=uv_cell_size[0]*max_wavelength,  # THIS HAS TO BE CHANGES, (X, Y) CELL SIZE ARE NOT THE SAME.
                    amplitude_image=amplitude,
                    phase_image=phase,
                    pointing_offset=do_pnt_off,
                    focus_xy_offsets=do_xy_foc_off,
                    focus_z_offset=do_z_foc_off,
                    subreflector_tilt=do_sub_til,
                    cassegrain_offset=do_cass_off)

    else:
        logger.info('Skipping phase correction')

    # Here we compute the aperture resolution from Equation 7 In EVLA memo 212
    # https://library.nrao.edu/public/memos/evla/EVLAM_212.pdf
    deltal = np.max(l) - np.min(l)
    deltam = np.max(m) - np.min(m)
    aperture_resolution = np.array([1/deltal, 1/deltam])
    aperture_resolution *= 1.27*min_wavelength
    
    # Todo: Add Paralactic angle as a non-dimension coordinate dependant on time.
    xds = xr.Dataset()

    xds["BEAM"] = xr.DataArray(beam_grid, dims=["time", "chan", "pol", "l", "m"])
    xds["APERTURE"] = xr.DataArray(aperture_grid, dims=["time", "chan", "pol", "u", "v"])
    
    xds["AMPLITUDE"] = xr.DataArray(amplitude, dims=["time", "chan", "pol", "u_prime", "v_prime"])
    xds["CORRECTED_PHASE"] = xr.DataArray(phase_corrected_angle, dims=["time", "chan", "pol", "u_prime", "v_prime"])

    xds.attrs["aperture_resolution"] = aperture_resolution
    xds.attrs["ant_id"] = holog_chunk_params["this_ant"]
    xds.attrs["ant_name"] = ant_name
    xds.attrs["telescope_name"] = meta_data['telescope_name']
    xds.attrs["time_centroid"] = np.array(time_centroid)
    xds.attrs["ddi"] = ddi
    
    coords = {
        "ddi": list(ant_data_dict.keys()), 
        "pol": pol, 
        "l": l, 
        "m": m, 
        "u": u, 
        "v": v, 
        "u_prime": u_prime,
        "v_prime": v_prime, 
        "chan": freq_chan
    }
    xds = xds.assign_coords(coords)
    xds.to_zarr("{name}/{ant}/{ddi}".format(name=holog_chunk_params["image_name"], ant=holog_chunk_params["this_ant"],
                                            ddi=ddi), mode="w", compute=True, consolidated=True)


def _create_average_chan_map(freq_chan, chan_tolerance_factor):
    n_chan = len(freq_chan)
    cf_chan_map = np.zeros((n_chan,), dtype=int)

    orig_width = (np.max(freq_chan) - np.min(freq_chan)) / len(freq_chan)

    tol = np.max(freq_chan) * chan_tolerance_factor
    n_pb_chan = int(np.floor((np.max(freq_chan) - np.min(freq_chan)) / tol) + 0.5)

    # Create PB's for each channel
    if n_pb_chan == 0:
        n_pb_chan = 1

    if n_pb_chan >= n_chan:
        cf_chan_map = np.arange(n_chan)
        pb_freq = freq_chan
        return cf_chan_map, pb_freq

    pb_delta_bandwdith = (np.max(freq_chan) - np.min(freq_chan)) / n_pb_chan
    pb_freq = (
        np.arange(n_pb_chan) * pb_delta_bandwdith
        + np.min(freq_chan)
        + pb_delta_bandwdith / 2
    )

    cf_chan_map = np.zeros((n_chan,), dtype=int)
    for i in range(n_chan):
        cf_chan_map[i], _ = _find_nearest(pb_freq, freq_chan[i])

    return cf_chan_map, pb_freq


def _export_to_fits_holog_chunk(parm_dict):
    """
    Holog side chunk function for the user facing function export_to_fits
    Args:
        parm_dict: parameter dictionary
    """
    inputxds = parm_dict['xds_data']
    metadata = parm_dict['metadata']
    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    basename = f'{destination}/{antenna}_{ddi}'
    
    logger.info(f'Exporting image contents of {antenna} {ddi} to FITS files in {destination}')

    try:
        aperture_resolution = inputxds.attrs["aperture_resolution"]

    except KeyError:
        logger.warning("Holog image does not have resolution information")
        logger.warning("Rerun holog with astrohack v>0.1.5 for aperture resolution information")
        
        aperture_resolution = None

    nchan = len(inputxds.chan)
    
    if nchan == 1:
        reffreq = inputxds.chan.values[0]
    
    else:
        reffreq = inputxds.chan.values[nchan//2]
    
    telname = inputxds.attrs['telescope_name']
    
    if telname in ['EVLA', 'VLA', 'JVLA']:
        telname = 'VLA'
    
    polist = []
    
    for pol in inputxds.pol:
        polist.append(str(pol.values))
    
    baseheader = {
        'STOKES'  : ", ".join(polist),
        'WAVELENG': clight/reffreq,
        'FREQUENC': reffreq,
        'TELESCOP': inputxds.attrs['ant_name'],
        'INSTRUME': telname,
        'TIME_CEN': inputxds.attrs['time_centroid'],
        'PADDING' : metadata['padding_factor'],
        'GRD_INTR': metadata['grid_interpolation_mode'],
        'CHAN_AVE': _bool_to_string(metadata['chan_average']),
        'CHAN_TOL': metadata['chan_tolerance_factor'],
        'SCAN_AVE': _bool_to_string(metadata['scan_average']),
        'TO_STOKE': _bool_to_string(metadata['to_stokes']),
    }

    ntime = len(inputxds.time)
    if ntime != 1:
        raise Exception("Data with multiple times not supported for FITS export")

    baseheader = _axis_to_fits_header(baseheader, inputxds.chan.values, 3, 'Frequency', 'Hz')
    baseheader = _stokes_axis_to_fits_header(baseheader, 4)
    rad_to_deg = _convert_unit('rad', 'deg', 'trigonometric')
    beamheader = _axis_to_fits_header(baseheader, -inputxds.l.values*rad_to_deg, 1, 'RA---SIN', 'deg')
    beamheader = _axis_to_fits_header(beamheader, inputxds.m.values*rad_to_deg, 2, 'DEC--SIN', 'deg')
    beamheader['RADESYSA'] = 'FK5'
    beam = inputxds['BEAM'].values
    if parm_dict['complex_split'] == 'cartesian':
        _write_fits(beamheader, 'Complex beam real part', beam.real, _add_prefix(basename, 'beam_real')+'.fits',
                    'Normalized', 'image')
        _write_fits(beamheader, 'Complex beam imag part', beam.imag, _add_prefix(basename, 'beam_imag')+'.fits',
                    'Normalized', 'image')
    else:
        _write_fits(beamheader, 'Complex beam amplitude', np.absolute(beam),
                    _add_prefix(basename, 'beam_amplitude')+'.fits', 'Normalized', 'image')
        _write_fits(beamheader, 'Complex beam phase', np.angle(beam),
                    _add_prefix(basename, 'beam_phase')+'.fits', 'Radians', 'image')
    wavelength = clight / inputxds.chan.values[0]
    apertureheader = _axis_to_fits_header(baseheader, inputxds.u.values*wavelength, 1, 'X----LIN', 'm')
    apertureheader = _axis_to_fits_header(apertureheader, inputxds.u.values*wavelength, 2, 'Y----LIN', 'm')
    apertureheader = _resolution_to_fits_header(apertureheader, aperture_resolution)
    aperture = inputxds['APERTURE'].values
    if parm_dict['complex_split'] == 'cartesian':
        _write_fits(apertureheader, 'Complex aperture real part', aperture.real,
                    _add_prefix(basename, 'aperture_real')+'.fits', 'Normalized', 'image')
        _write_fits(apertureheader, 'Complex aperture imag part', aperture.imag,
                    _add_prefix(basename, 'aperture_imag')+'.fits', 'Normalized', 'image')
    else:
        _write_fits(apertureheader, 'Complex aperture amplitude', np.absolute(aperture),
                    _add_prefix(basename, 'aperture_amplitude')+'.fits', 'Normalized', 'image')
        _write_fits(apertureheader, 'Complex aperture phase', np.angle(aperture),
                    _add_prefix(basename, 'aperture_phase')+'.fits', 'rad', 'image')

    phase_amp_header = _axis_to_fits_header(baseheader, inputxds.u_prime.values*wavelength, 1, 'X----LIN', 'm')
    phase_amp_header = _axis_to_fits_header(phase_amp_header, inputxds.v_prime.values*wavelength, 2, 'Y----LIN', 'm')
    phase_amp_header = _resolution_to_fits_header(phase_amp_header, aperture_resolution)
    _write_fits(phase_amp_header, 'Cropped aperture corrected phase', inputxds['CORRECTED_PHASE'].values,
                _add_prefix(basename, 'corrected_phase')+'.fits', 'rad', 'image')
    return


def _plot_aperture_chunk(parm_dict):
    """
    Chunk function for the user facing function plot_apertures
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    basename = f'{destination}/{antenna}_{ddi}'
    inputxds = parm_dict['xds_data']
    inputxds.attrs['AIPS'] = False
    telescope = _get_correct_telescope_from_name(inputxds)
    surface = AntennaSurface(inputxds, telescope, nan_out_of_bounds=False)

    surface.plot_phase(basename, 'image', parm_dict)
    surface.plot_deviation(basename, 'image', parm_dict)
    surface.plot_amplitude(basename, 'image', parm_dict)


def _plot_beam_chunk(parm_dict):
    """
    Chunk function for the user facing function plot_beams
    Args:
        parm_dict: parameter dictionary
    """
    antenna = parm_dict['this_ant']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    basename = f'{destination}/{antenna}_{ddi}'
    inputxds = parm_dict['xds_data']
    laxis = inputxds.l.values*_convert_unit('rad', parm_dict['angle_unit'], 'trigonometric')
    maxis = inputxds.m.values*_convert_unit('rad', parm_dict['angle_unit'], 'trigonometric')
    if inputxds.dims['chan'] != 1:
        raise Exception("Only single channel holographies supported")

    if inputxds.dims['time'] != 1:
        raise Exception("Only single mapping holographies supported")

    full_beam = inputxds.BEAM.isel(time=0, chan=0).values
    pol_axis = inputxds.pol.values
    if parm_dict['complex_split'] == 'cartesian':
        realpart = full_beam.real
        imagpart = full_beam.imag
        _plot_beam(laxis, maxis, pol_axis, realpart, basename, 'real', 'normalized', parm_dict)
        _plot_beam(laxis, maxis, pol_axis, imagpart, basename, 'imag', 'normalized', parm_dict)
    else:
        ampli = np.absolute(full_beam)
        phase = np.angle(full_beam)*_convert_unit('rad', parm_dict['phase_unit'], 'trigonometric')
        _plot_beam(laxis, maxis, pol_axis, ampli, basename, 'amplitude', 'normalized', parm_dict)
        _plot_beam(laxis, maxis, pol_axis, phase, basename, 'phase', parm_dict['phase_unit'], parm_dict)


def _plot_beam(laxis, maxis, pol_axis, data, basename, label, zunit, parm_dict):
    """
    Plot a beam
    Args:
        laxis: L axis
        maxis: M axis
        pol_axis: Polarization axis
        data: Beam data
        basename: Basename for output file
        label: data label
        zunit: data unit
        parm_dict: dictionary with general and plotting parameters
    """
    colormap = _get_proper_color_map(parm_dict['colormap'])

    n_pol = len(pol_axis)
    
    if n_pol == 4:
        fig, axes = _create_figure_and_axes(parm_dict['figure_size'], [2, 2])
        axes = axes.flat
    elif n_pol == 2:
        fig, axes = _create_figure_and_axes(parm_dict['figure_size'], [2, 1])
    elif n_pol == 1:
        fig, ax = _create_figure_and_axes(parm_dict['figure_size'], [1, 1])
        axes = [ax]
    else:
        msg = f'Do not know how to handle polarization axis with {n_pol} elements'
        logger.error(msg)
        raise Exception(msg)

    extent = [laxis[0], laxis[-1], maxis[0], maxis[-1]]
    for ipol, pol, in enumerate(pol_axis):
        axis = axes[ipol]
        axis.set_title(f'Polarization: {pol}')
        im = axis.imshow(data[ipol, ...], cmap=colormap, interpolation="nearest", extent=extent)
        _well_positioned_colorbar(axis, fig, im, f"Z Scale [{zunit}]")
        axis.set_xlabel(f'L axis [{parm_dict["angle_unit"]}]')
        axis.set_ylabel(f'M axis [{parm_dict["angle_unit"]}]')

    plot_name = _add_prefix(_add_prefix(basename, label), 'image_beam')
    suptitle = f'Beam {label}, Antenna: {parm_dict["this_ant"].split("_")[1]}, DDI: {parm_dict["this_ddi"].split("_")[1]}'
    _close_figure(fig, suptitle, plot_name, parm_dict["dpi"], parm_dict["display"])
    return
