import scipy
import numpy as np
import xarray as xr

from scipy.interpolate import griddata

from astrohack._classes.telescope import Telescope

from astrohack._utils._io import _load_holog_file, _load_image_xds
from astrohack._utils._io import _read_meta_data, _write_meta_data, _write_fits

from astrohack._utils._panel import _phase_fitting_block

from astrohack._utils._algorithms import _chunked_average
from astrohack._utils._algorithms import _find_peak_beam_value
from astrohack._utils._algorithms import _find_nearest
from astrohack._utils._algorithms import _calc_coords

from astrohack._utils._conversion import _to_stokes
from astrohack._utils._constants import clight
from astrohack._utils._tools import _bool_to_string, _axis_to_fits_header, _stokes_axis_to_fits_header

from astrohack._utils._imaging import _parallactic_derotation
from astrohack._utils._imaging import _mask_circular_disk
from astrohack._utils._imaging import _calculate_aperture_pattern

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger


def _holog_chunk(holog_chunk_params):
    """ Process chunk holography data along the antenna axis. Works with holography file to properly grid , normalize, average and correct data
        and returns the aperture pattern.

    Args:
        holog_chunk_params (dict): Dictionary containing holography parameters.
    """
    logger = _get_astrohack_logger()
    
    c = scipy.constants.speed_of_light
    

    holog_file, ant_data_dict = _load_holog_file(
        holog_chunk_params["holog_file"],
        dask_load=False,
        load_pnt_dict=False,
        ant_id=holog_chunk_params["ant_id"],
        ddi_id=holog_chunk_params["ddi_id"]
    )

    meta_data = _read_meta_data(holog_chunk_params["holog_file"], 'holog', 'extract_holog')

    # Calculate lm coordinates
    l, m = _calc_coords(holog_chunk_params["grid_size"], holog_chunk_params["cell_size"])
    grid_l, grid_m = list(map(np.transpose, np.meshgrid(l, m)))
    
    to_stokes = holog_chunk_params["to_stokes"]

    ddi = holog_chunk_params["ddi_id"]
    n_holog_map = len(ant_data_dict[ddi].keys())
    
    # For a fixed ddi the frequency axis should not change over holog_maps, consequently we only have to consider the first holog_map.
    map0 = list(ant_data_dict[ddi].keys())[0]
    
    freq_chan = ant_data_dict[ddi][map0].chan.values
    n_chan = ant_data_dict[ddi][map0].dims["chan"]
    n_pol = ant_data_dict[ddi][map0].dims["pol"]
    
    if holog_chunk_params["chan_average"]:
        reference_scaling_frequency = np.mean(freq_chan)

        avg_chan_map, avg_freq = _create_average_chan_map(freq_chan, holog_chunk_params["chan_tolerance_factor"])
        
        # Only a single channel left after averaging.
        beam_grid = np.zeros((n_holog_map,) + (1, n_pol) + grid_l.shape, dtype=np.complex)

    else:
        beam_grid = np.zeros((n_holog_map,) + (n_chan, n_pol) + grid_l.shape, dtype=np.complex)

    time_centroid = []

    for holog_map_index, holog_map in enumerate(ant_data_dict[ddi].keys()):
        ant_xds = ant_data_dict[ddi][holog_map]
        
        ###To Do: Add flagging code

        # Grid the data
        vis = ant_xds.VIS.values
        vis[vis==np.nan] = 0.0
        lm = ant_xds.DIRECTIONAL_COSINES.values
        weight = ant_xds.WEIGHT.values

        if holog_chunk_params["chan_average"]:
            vis_avg, weight_sum = _chunked_average(vis, weight, avg_chan_map, avg_freq)
            lm_freq_scaled = lm[:, :, None] * (avg_freq / reference_scaling_frequency)

            n_chan = avg_freq.shape[0]
            
            # Unavoidable for loop because lm change over frequency.
            for chan_index in range(n_chan):

                # Average scaled beams.
                beam_grid[holog_map_index, 0, :, :, :] = (beam_grid[holog_map_index, 0, :, :, :] + np.moveaxis(griddata(lm_freq_scaled[:, :, chan_index], vis_avg[:, chan_index, :], (grid_l, grid_m), method=holog_chunk_params["grid_interpolation_mode"],fill_value=0.0),(2),(0)))

            # Avergaing now complete
            n_chan =  1
            
            freq_chan = [np.mean(avg_freq)]
        else:
            beam_grid[holog_map_index, ...] = np.moveaxis(griddata(lm, vis, (grid_l, grid_m), method=holog_chunk_params["grid_interpolation_mode"],fill_value=0.0), (0,1), (2,3))


        time_centroid_index = ant_data_dict[ddi][holog_map].dims["time"] // 2
        time_centroid.append(ant_data_dict[ddi][holog_map].coords["time"][time_centroid_index].values)

        for chan in range(n_chan): ### Todo: Vectorize holog_map and channel axis
            try:
                xx_peak = _find_peak_beam_value(beam_grid[holog_map_index, chan, 0, ...], scaling=0.25)
                yy_peak = _find_peak_beam_value(beam_grid[holog_map_index, chan, 3, ...], scaling=0.25)
            except:
                center_pixel = np.array(beam_grid.shape[-2:])//2
                xx_peak = beam_grid[holog_map_index, chan, 0, center_pixel[0], center_pixel[1]]
                yy_peak = beam_grid[holog_map_index, chan, 3, center_pixel[0], center_pixel[1]]

            normalization = np.abs(0.5 * (xx_peak + yy_peak))
            beam_grid[holog_map_index, chan, ...] /= normalization

    beam_grid = _parallactic_derotation(data=beam_grid, parallactic_angle_dict=ant_data_dict[ddi])
    

    ###############
    pol = beam_grid,ant_data_dict[ddi][holog_map].pol.values
    if to_stokes:
        beam_grid = _to_stokes(beam_grid,ant_data_dict[ddi][holog_map].pol.values)
        pol=['I','Q','U','V']
    ###############
    
    if holog_chunk_params["scan_average"]:
        beam_grid = np.mean(beam_grid,axis=0)[None,...]
        time_centroid = np.mean(np.array(time_centroid))
    
    # Current bottleneck
    aperture_grid, u, v, uv_cell_size = _calculate_aperture_pattern(
        grid=beam_grid,
        delta=holog_chunk_params["cell_size"],
        padding_factor=holog_chunk_params["padding_factor"],
    )
    
    # Get telescope info
    ant_name = ant_data_dict[ddi][holog_map].attrs["antenna_name"]
    
    if  ant_name.upper().__contains__('DV'):
        telescope_name = "_".join((meta_data['telescope_name'], 'DV'))

    elif  ant_name.upper().__contains__('DA'):
        telescope_name = "_".join((meta_data['telescope_name'], 'DA'))
        
    elif  ant_name.upper().__contains__('EA'):
        telescope_name = 'VLA'

    else:
        raise Exception("Antenna type not found: {}".format(meta_data['ant_name']))

    telescope = Telescope(telescope_name)

    min_wavelength = scipy.constants.speed_of_light/freq_chan[0]
    max_aperture_radius = (0.5*telescope.diam)/min_wavelength
    
    image_slice = aperture_grid[0, 0, 0, ...]
    center_pixel = np.array(image_slice.shape[0:2])//2
    radius_u = int(np.where(np.abs(u) < max_aperture_radius*1.1)[0].max() - center_pixel[0]) # Factor of 1.1: Let's not be too aggresive
    radius_v = int(np.where(np.abs(v) < max_aperture_radius*1.1)[0].max() - center_pixel[1]) # Factor of 1.1: Let's not be too aggresive
        
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
            logger.error("Phase fit parameter must have 5 elements")
            raise Exception
        else:
            if np.sum(phase_fit_par) == 0:
                do_phase_fit = False
            else:
                do_phase_fit = True
                do_pnt_off, do_xy_foc_off, do_z_foc_off, do_sub_til, do_cass_off = phase_fit_par
    else:
        logger.error("Phase fit parameter is neither a boolean nor an array of booleans")
        raise Exception

    if do_phase_fit:
        logger.info("Applying phase correction ...")
        
        if to_stokes:
            pols=(0,)
        else:
            pols=(0, 3)
        
        #? Wavelength
        max_wavelength = scipy.constants.speed_of_light/freq_chan[-1]
        
        results, errors, phase_corrected_angle, _, in_rms, out_rms = _phase_fitting_block(
                    pols=pols,
                    wavelength=max_wavelength,
                    telescope=telescope,
                    cellxy=uv_cell_size[0]*max_wavelength, # THIS HAS TO BE CHANGES, (X, Y) CELL SIZE ARE NOT THE SAME.
                    amplitude_image=amplitude,
                    phase_image=phase,
                    pointing_offset=do_pnt_off,
                    focus_xy_offsets=do_xy_foc_off,
                    focus_z_offset=do_z_foc_off,
                    subreflector_tilt=do_sub_til,
                    cassegrain_offset=do_cass_off)
    else:
        logger.info("Skipping phase correction ...")

    
    ###To Do: Add Paralactic angle as a non-dimension coordinate dependant on time.
    xds = xr.Dataset()

    xds["BEAM"] = xr.DataArray(beam_grid, dims=["time", "chan", "pol", "l", "m"])
    xds["APERTURE"] = xr.DataArray(aperture_grid, dims=["time", "chan", "pol", "u", "v"])
    
    xds["AMPLITUDE"] = xr.DataArray(amplitude, dims=["time", "chan", "pol", "u_prime", "v_prime"])
    xds["CORRECTED_PHASE"] = xr.DataArray(phase_corrected_angle, dims=["time", "chan", "pol", "u_prime", "v_prime"])

    xds.attrs["ant_id"] = holog_chunk_params["ant_id"]
    xds.attrs["ant_name"] = ant_name
    xds.attrs["telescope_name"] = meta_data['telescope_name']
    xds.attrs["time_centroid"] = np.array(time_centroid)
    xds.attrs["ddi"] = ddi

    coords = {}
    #coords["time"] = np.array(time_centroid)
    coords["ddi"] = list(ant_data_dict.keys())
    coords["pol"] = pol
    coords["l"] = l
    coords["m"] = m
    coords["u"] = u
    coords["v"] = v
    coords["u_prime"] = u_prime
    coords["v_prime"] = v_prime
    coords["chan"] = freq_chan

    xds = xds.assign_coords(coords)

    xds.to_zarr("{name}/{ant}/{ddi}".format(name= holog_chunk_params["image_file"], ant=holog_chunk_params["ant_id"], ddi=ddi), mode="w", compute=True, consolidated=True)


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


def _create_image_meta_data(image_file, input_params):
    """
    Save image meta data to a json file
    Args:
        image_file: image file
        input_params: holog input parameter dictionaire
    """
    output_attr_file = "{name}/{ext}".format(name=image_file, ext=".image_attr")
    _write_meta_data('holog', output_attr_file, input_params)


def _export_to_fits_holog_chunk(parm_dict):
    """
    Holog side chunk function for the user facing function export_to_fits
    Args:
        parm_dict: parameter dictionary
    """
    logger = _get_astrohack_logger()

    inputxds = _load_image_xds(parm_dict['filename'], parm_dict['this_antenna'], parm_dict['this_ddi'], dask_load=False)
    metadata = parm_dict['holog_mds']._meta_data

    antenna = parm_dict['this_antenna']
    ddi = parm_dict['this_ddi']
    destination = parm_dict['destination']
    basename = f'{destination}/image_{antenna}_{ddi}'

    logger.info(f'Exporting image contents of {antenna} {ddi} to FITS files in {destination}')

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
        logger.error("Data with multiple times not supported for FITS export")
        raise Exception("Data with multiple times not supported for FITS export")

    carta_dim_order = (1, 0, 2, 3, )

    baseheader = _axis_to_fits_header(baseheader, inputxds.chan.values, 3, 'Frequency', 'Hz')
    baseheader = _stokes_axis_to_fits_header(baseheader, 4)

    beamheader = _axis_to_fits_header(baseheader, inputxds.l.values, 1, 'L', 'rad')
    beamheader = _axis_to_fits_header(beamheader, inputxds.m.values, 2, 'M', 'rad')
    transpobeam = np.transpose(inputxds['BEAM'][0, ...].values, carta_dim_order)
    if parm_dict['complex_split'] == 'cartesian':
        _write_fits(beamheader, 'Complex beam real part', transpobeam.real, basename + '_beam_real.fits', 'Normalized',
                    'image')
        _write_fits(beamheader, 'Complex beam imag part', transpobeam.imag, basename + '_beam_imag.fits', 'Normalized',
                    'image')
    else:
        _write_fits(beamheader, 'Complex beam amplitude', np.absolute(transpobeam),
                    basename + '_beam_amplitude.fits', 'Normalized', 'image')
        _write_fits(beamheader, 'Complex beam phase', np.angle(transpobeam), basename + '_beam_phase.fits',
                    'Radians', 'image')

    apertureheader = _axis_to_fits_header(baseheader, inputxds.u.values, 1, 'X', 'm')
    apertureheader = _axis_to_fits_header(apertureheader, inputxds.v.values, 2, 'Y', 'm')
    transpoaper = np.transpose(inputxds['APERTURE'][0, ...].values, carta_dim_order)
    if parm_dict['complex_split'] == 'cartesian':
        _write_fits(apertureheader, 'Complex aperture real part', transpoaper.real,
                    basename + '_aperture_real.fits', 'Normalized', 'image')
        _write_fits(apertureheader, 'Complex aperture imag part', transpoaper.imag,
                    basename + '_aperture_imag.fits', 'Normalized', 'image')
    else:
        _write_fits(apertureheader, 'Complex aperture amplitude', np.absolute(transpoaper),
                    basename + '_aperture_amplitude.fits', 'Normalized', 'image')
        _write_fits(apertureheader, 'Complex aperture phase', np.angle(transpoaper),
                    basename + '_aperture_phase.fits', 'rad', 'image')

    phase_amp_header = _axis_to_fits_header(baseheader, inputxds.u_prime.values, 1, 'X', 'm')
    phase_amp_header = _axis_to_fits_header(phase_amp_header, inputxds.v_prime.values, 2, 'Y', 'm')
    transpoamp = np.transpose(inputxds['AMPLITUDE'][0, ...].values, carta_dim_order)
    _write_fits(phase_amp_header, 'Cropped aperture amplitude', transpoamp, basename + '_amplitude.fits',
                'Normalized', 'image')
    transpopha = np.transpose(inputxds['CORRECTED_PHASE'][0, ...].values, carta_dim_order)
    _write_fits(phase_amp_header, 'Cropped aperture corrected phase', transpopha, basename + '_corrected_phase.fits',
                'rad', 'image')
    return
