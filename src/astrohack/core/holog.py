import astrohack
import numpy as np
import xarray as xr

from scipy.interpolate import griddata

from astrohack.antenna.telescope import Telescope
from astrohack.utils.algorithms import calc_coords
from astrohack.utils.algorithms import chunked_average
from astrohack.utils.algorithms import find_nearest
from astrohack.utils.algorithms import find_peak_beam_value
from astrohack.utils.constants import clight
from astrohack.utils.data import read_meta_data
from astrohack.utils.file import load_holog_file
from astrohack.utils.imaging import calculate_aperture_pattern
from astrohack.utils.imaging import mask_circular_disk
from astrohack.utils.imaging import parallactic_derotation
from astrohack.utils.phase_fitting import phase_fitting_block

import graphviper.utils.logger as logger


def process_holog_chunk(holog_chunk_params):
    """ Process chunk holography data along the antenna axis. Works with holography file to properly grid , normalize,
        average and correct data and returns the aperture pattern.

    Args:
        holog_chunk_params (dict): Dictionary containing holography parameters.
    """
    holog_file, ant_data_dict = load_holog_file(
        holog_chunk_params["holog_name"],
        dask_load=False,
        load_pnt_dict=False,
        ant_id=holog_chunk_params["this_ant"],
        ddi_id=holog_chunk_params["this_ddi"]
    )

    meta_data = read_meta_data(holog_chunk_params["holog_name"] + '/.holog_attr')

    # Calculate lm coordinates
    l, m = calc_coords(holog_chunk_params["grid_size"], holog_chunk_params["cell_size"])

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
            vis_avg, weight_sum = chunked_average(vis, weight, avg_chan_map, avg_freq)
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
                xx_peak = find_peak_beam_value(beam_grid[holog_map_index, chan, 0, ...], scaling=0.25)
                yy_peak = find_peak_beam_value(beam_grid[holog_map_index, chan, 3, ...], scaling=0.25)
            except:
                center_pixel = np.array(beam_grid.shape[-2:]) // 2
                xx_peak = beam_grid[holog_map_index, chan, 0, center_pixel[0], center_pixel[1]]
                yy_peak = beam_grid[holog_map_index, chan, 3, center_pixel[0], center_pixel[1]]

            normalization = np.abs(0.5 * (xx_peak + yy_peak))

            if normalization == 0:
                logger.warning("Peak of zero found! Setting normalization to unity.")
                normalization = 1

            beam_grid[holog_map_index, chan, ...] /= normalization

    beam_grid = parallactic_derotation(data=beam_grid, parallactic_angle_dict=ant_data_dict[ddi])

    ###############

    pol = ant_data_dict[ddi][holog_map].pol.values
    if to_stokes:
        beam_grid = astrohack.utils.conversion.to_stokes(beam_grid, ant_data_dict[ddi][holog_map].pol.values)
        pol = ['I', 'Q', 'U', 'V']

    ###############

    if holog_chunk_params["scan_average"]:
        beam_grid = np.mean(beam_grid, axis=0)[None, ...]
        time_centroid = np.mean(np.array(time_centroid))

    logger.info("Calculating aperture pattern ...")
    # Current bottleneck
    aperture_grid, u, v, uv_cell_size = calculate_aperture_pattern(
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

    min_wavelength = clight / freq_chan[0]
    max_aperture_radius = (0.5 * telescope.diam) / min_wavelength

    image_slice = aperture_grid[0, 0, 0, ...]
    center_pixel = np.array(image_slice.shape[0:2]) // 2

    # Factor of 1.1: Let's not be too aggressive
    radius_u = int(np.where(np.abs(u) < max_aperture_radius * 1.1)[0].max() - center_pixel[0])
    radius_v = int(np.where(np.abs(v) < max_aperture_radius * 1.1)[0].max() - center_pixel[1])

    if radius_v > radius_u:
        radius = radius_v
    else:
        radius = radius_u

    if holog_chunk_params['apply_mask']:
        # Masking Aperture image
        mask = mask_circular_disk(
            center=None,
            radius=radius,
            array=aperture_grid,
        )

        aperture_grid = mask * aperture_grid

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

        max_wavelength = clight / freq_chan[-1]

        results, errors, phase_corrected_angle, _, in_rms, out_rms = phase_fitting_block(
            pols=pols,
            wavelength=max_wavelength,
            telescope=telescope,
            cellxy=uv_cell_size[0] * max_wavelength,  # THIS HAS TO BE CHANGES, (X, Y) CELL SIZE ARE NOT THE SAME.
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
    aperture_resolution = np.array([1 / deltal, 1 / deltam])
    aperture_resolution *= 1.27 * min_wavelength

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
        cf_chan_map[i], _ = find_nearest(pb_freq, freq_chan[i])

    return cf_chan_map, pb_freq
