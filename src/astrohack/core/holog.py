import astrohack
import numpy as np
import xarray as xr

from astrohack.antenna.telescope import Telescope
from astrohack.utils.algorithms import calc_coords
from astrohack.utils.constants import clight
from astrohack.utils.data import read_meta_data
from astrohack.utils.file import load_holog_file
from astrohack.utils.imaging import calculate_far_field_aperture, calculate_near_field_aperture
from astrohack.utils.imaging import mask_circular_disk
from astrohack.utils.gridding import grid_beam
from astrohack.utils.imaging import parallactic_derotation
from astrohack.utils.phase_fitting import execute_phase_fitting

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
    ddi = holog_chunk_params["this_ddi"]
    to_stokes = holog_chunk_params["to_stokes"]
    ref_xds = ant_data_dict[ddi]['map_0']
    telescope = _get_correct_telescope(ref_xds.attrs["antenna_name"], meta_data['telescope_name'])
    try:
        is_near_field = ref_xds.attrs['near_field']
    except KeyError:
        is_near_field = False

    beam_grid, time_centroid, freq_chan, \
        pol_axis, l, m, grid_corr = grid_beam(ant_ddi_dict=ant_data_dict[ddi],
                                              grid_size=holog_chunk_params["grid_size"],
                                              cell_size=holog_chunk_params["cell_size"],
                                              avg_chan=holog_chunk_params["chan_average"],
                                              chan_tol_fac=holog_chunk_params["chan_tolerance_factor"],
                                              telescope=telescope,
                                              grid_interpolation_mode=holog_chunk_params["grid_interpolation_mode"]
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

    logger.info("Calculating aperture pattern ...")
    # Current bottleneck
    if is_near_field:
        focus_offset = ref_xds.attrs["nf_focus_off"]
        aperture_grid, u, v, uv_cell_size, distance = calculate_near_field_aperture(
            grid=beam_grid,
            sky_cell_size=holog_chunk_params["cell_size"],
            distance=holog_chunk_params["distance_to_tower"],
            wavelength=clight / freq_chan[0],
            padding_factor=holog_chunk_params["padding_factor"],
            focus_offset=focus_offset,
            focal_length=telescope.focus,
            diameter=telescope.diam,
            blockage=telescope.inlim,
        )
    else:
        focus_offset = 0
        aperture_grid, u, v, uv_cell_size = calculate_far_field_aperture(
            grid=beam_grid,
            delta=holog_chunk_params["cell_size"],
            padding_factor=holog_chunk_params["padding_factor"],
        )

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
        mask = mask_circular_disk(center=None, radius=radius, array=aperture_grid)
        aperture_grid = mask * aperture_grid

    start_cut = center_pixel - radius
    end_cut = center_pixel + radius

    amplitude = np.absolute(aperture_grid[..., start_cut[0]:end_cut[0], start_cut[1]:end_cut[1]])
    phase = np.angle(aperture_grid[..., start_cut[0]:end_cut[0], start_cut[1]:end_cut[1]])
    u_prime = u[start_cut[0]:end_cut[0]]
    v_prime = v[start_cut[1]:end_cut[1]]

    phase_corrected_angle, phase_fit_results = execute_phase_fitting(amplitude, phase,
                                                                     pol_axis,
                                                                     freq_chan, telescope, uv_cell_size,
                                                                     holog_chunk_params["phase_fit"], to_stokes,
                                                                     is_near_field, focus_offset, u_prime, v_prime)

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
    xds.attrs["ant_name"] = ant_data_dict[ddi]['map_0'].attrs["antenna_name"]
    xds.attrs["telescope_name"] = meta_data['telescope_name']
    xds.attrs["time_centroid"] = np.array(time_centroid)
    xds.attrs["ddi"] = ddi
    xds.attrs["phase_fitting"] = phase_fit_results

    coords = {
        "ddi": list(ant_data_dict.keys()),
        "pol": pol_axis,
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
