import astrohack
import numpy as np
import xarray as xr

from astrohack.antenna.telescope import Telescope
from astrohack.utils import create_dataset_label
from astrohack.utils.algorithms import calc_coords
from astrohack.utils.constants import clight
from astrohack.utils.data import read_meta_data
from astrohack.utils.file import load_holog_file
from astrohack.utils.imaging import calculate_far_field_aperture, calculate_near_field_aperture
from astrohack.utils.imaging import mask_circular_disk
from astrohack.utils.gridding import grid_beam
from astrohack.utils.imaging import parallactic_derotation
from astrohack.utils.phase_fitting import execute_phase_fitting

import toolviper.utils.logger as logger


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
    label = create_dataset_label(holog_chunk_params["this_ant"], holog_chunk_params["this_ddi"])
    logger.info(f'Processing {label}')
    meta_data = read_meta_data(holog_chunk_params["holog_name"] + '/.holog_attr')
    ddi = holog_chunk_params["this_ddi"]
    to_stokes = holog_chunk_params["to_stokes"]
    ref_xds = ant_data_dict[ddi]['map_0']
    telescope = _get_correct_telescope(ref_xds.attrs["antenna_name"], meta_data['telescope_name'])
    try:
        is_near_field = ref_xds.attrs['near_field']
    except KeyError:
        is_near_field = False

    beam_grid, time_centroid, freq_axis, \
        pol_axis, l_axis, m_axis, grid_corr = grid_beam(ant_ddi_dict=ant_data_dict[ddi],
                                                        grid_size=holog_chunk_params["grid_size"],
                                                        sky_cell_size=holog_chunk_params["cell_size"],
                                                        avg_chan=holog_chunk_params["chan_average"],
                                                        chan_tol_fac=holog_chunk_params["chan_tolerance_factor"],
                                                        telescope=telescope,
                                                        grid_interpolation_mode=
                                                        holog_chunk_params["grid_interpolation_mode"],
                                                        label=label
                                                        )

    if not is_near_field:
        beam_grid = parallactic_derotation(data=beam_grid, parallactic_angle_dict=ant_data_dict[ddi])

    ###############

    if to_stokes:
        beam_grid = astrohack.utils.conversion.to_stokes(beam_grid, pol_axis)
        pol_axis = ['I', 'Q', 'U', 'V']

    ###############

    if holog_chunk_params["scan_average"]:
        beam_grid = np.mean(beam_grid, axis=0)[None, ...]
        time_centroid = np.mean(np.array(time_centroid))

    # Current bottleneck
    if is_near_field:
        distance, focus_offset = telescope.dist_dict[holog_chunk_params["alma_osf_pad"]]
        aperture_grid, u_axis, v_axis, uv_cell_size, used_wavelength = calculate_near_field_aperture(
            grid=beam_grid,
            sky_cell_size=holog_chunk_params["cell_size"],
            distance=distance,
            freq=freq_axis,
            padding_factor=holog_chunk_params["padding_factor"],
            focus_offset=focus_offset,
            telescope=telescope,
            apply_grid_correction=grid_corr,
            label=label
        )
    else:
        focus_offset = 0
        aperture_grid, u_axis, v_axis, uv_cell_size, used_wavelength = calculate_far_field_aperture(
            grid=beam_grid,
            padding_factor=holog_chunk_params["padding_factor"],
            freq=freq_axis,
            telescope=telescope,
            sky_cell_size=holog_chunk_params["cell_size"],
            apply_grid_correction=grid_corr,
            label=label
        )

    amplitude, phase, u_prime, v_prime = _crop_and_split_aperture(aperture_grid, u_axis, v_axis, telescope,
                                                                  holog_chunk_params['apply_mask'])

    phase_corrected_angle, phase_fit_results = execute_phase_fitting(amplitude, phase, pol_axis, freq_axis, telescope,
                                                                     uv_cell_size, holog_chunk_params["phase_fit"],
                                                                     to_stokes, is_near_field, focus_offset, u_prime,
                                                                     v_prime, label)

    aperture_resolution = _compute_aperture_resolution(l_axis, m_axis, used_wavelength)
    _export_to_xds(beam_grid, aperture_grid, amplitude, phase_corrected_angle, aperture_resolution,
                   holog_chunk_params["this_ant"], ant_data_dict[ddi]['map_0'].attrs["antenna_name"],
                   meta_data['telescope_name'], time_centroid, ddi, phase_fit_results, pol_axis, freq_axis, l_axis,
                   m_axis, u_axis, v_axis, u_prime, v_prime, holog_chunk_params["image_name"])

    logger.info(f'Finished processing {label}')


def _get_correct_telescope(ant_name, telescope_name):
    # Get telescope info
    if ant_name.upper().__contains__('DV'):
        telescope_name = "_".join((telescope_name, 'DV'))

    elif ant_name.upper().__contains__('DA'):
        telescope_name = "_".join((telescope_name, 'DA'))

    elif ant_name.upper().__contains__('EA'):
        telescope_name = 'VLA'

    else:
        raise Exception("Antenna type not found: {name}".format(name=ant_name))

    return Telescope(telescope_name)


def _crop_and_split_aperture(aperture_grid, u_axis, v_axis, telescope, apply_mask, scaling=1.1):
    # Default scaling factor of 1.1: Let's not be too aggressive
    max_aperture_radius = (0.5 * telescope.diam)

    image_slice = aperture_grid[0, 0, 0, ...]
    center_pixel = np.array(image_slice.shape[0:2]) // 2
    radius_u = int(np.where(np.abs(u_axis) < max_aperture_radius * scaling)[0].max() - center_pixel[0])
    radius_v = int(np.where(np.abs(v_axis) < max_aperture_radius * scaling)[0].max() - center_pixel[1])

    if radius_v > radius_u:
        radius = radius_v
    else:
        radius = radius_u

    if apply_mask:
        # Masking Aperture image
        mask = mask_circular_disk(center=None, radius=radius, array=aperture_grid)
        aperture_grid = mask * aperture_grid

    start_cut = center_pixel - radius
    end_cut = center_pixel + radius

    amplitude = np.absolute(aperture_grid[..., start_cut[0]:end_cut[0], start_cut[1]:end_cut[1]])
    phase = np.angle(aperture_grid[..., start_cut[0]:end_cut[0], start_cut[1]:end_cut[1]])
    return amplitude, phase, u_axis[start_cut[0]:end_cut[0]], v_axis[start_cut[1]:end_cut[1]]


def _compute_aperture_resolution(l_axis, m_axis, wavelength):
    # Here we compute the aperture resolution from Equation 7 In EVLA memo 212
    # https://library.nrao.edu/public/memos/evla/EVLAM_212.pdf
    deltal = np.max(l_axis) - np.min(l_axis)
    deltam = np.max(m_axis) - np.min(m_axis)
    aperture_resolution = np.array([1 / deltal, 1 / deltam])
    aperture_resolution *= 1.27 * wavelength
    return aperture_resolution


def _export_to_xds(beam_grid, aperture_grid, amplitude, phase_corrected_angle, aperture_resolution, ant_id, ant_name,
                   telescope_name, time_centroid, ddi, phase_fit_results, pol_axis, freq_axis, l_axis, m_axis, u_axis,
                   v_axis, u_prime, v_prime, image_name):
    # Todo: Add Paralactic angle as a non-dimension coordinate dependant on time.
    xds = xr.Dataset()

    xds["BEAM"] = xr.DataArray(beam_grid, dims=["time", "chan", "pol", "l", "m"])
    xds["APERTURE"] = xr.DataArray(aperture_grid, dims=["time", "chan", "pol", "u", "v"])

    xds["AMPLITUDE"] = xr.DataArray(amplitude, dims=["time", "chan", "pol", "u_prime", "v_prime"])
    xds["CORRECTED_PHASE"] = xr.DataArray(phase_corrected_angle, dims=["time", "chan", "pol", "u_prime", "v_prime"])

    xds.attrs["aperture_resolution"] = aperture_resolution
    xds.attrs["ant_id"] = ant_id
    xds.attrs["ant_name"] = ant_name
    xds.attrs["telescope_name"] = telescope_name
    xds.attrs["time_centroid"] = np.array(time_centroid)
    xds.attrs["ddi"] = ddi
    xds.attrs["phase_fitting"] = phase_fit_results

    coords = {
        "pol": pol_axis,
        "l": l_axis,
        "m": m_axis,
        "u": u_axis,
        "v": v_axis,
        "u_prime": u_prime,
        "v_prime": v_prime,
        "chan": freq_axis
    }
    xds = xds.assign_coords(coords)
    xds.to_zarr(f"{image_name}/{ant_id}/{ddi}", mode="w", compute=True, consolidated=True)

