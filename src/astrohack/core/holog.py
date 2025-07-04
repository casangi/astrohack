import numpy as np
import xarray as xr

from astrohack.antenna.telescope import Telescope
from astrohack.utils import format_angular_distance
from astrohack.utils.text import create_dataset_label
from astrohack.utils.conversion import convert_5d_grid_to_stokes
from astrohack.utils.algorithms import phase_wrapping
from astrohack.utils.zernike_aperture_fitting import fit_zernike_coefficients
from astrohack.utils.file import load_holog_file
from astrohack.utils.imaging import (
    calculate_far_field_aperture,
    calculate_near_field_aperture,
)
from astrohack.utils.gridding import grid_beam
from astrohack.utils.imaging import parallactic_derotation
from astrohack.utils.phase_fitting import (
    clic_like_phase_fitting,
    skip_phase_fitting,
    aips_like_phase_fitting,
)

import toolviper.utils.logger as logger


def process_holog_chunk(holog_chunk_params):
    """Process chunk holography data along the antenna axis. Works with holography file to properly grid , normalize,
        average and correct data and returns the aperture pattern.

    Args:
        holog_chunk_params (dict): Dictionary containing holography parameters.
    """
    holog_file, ant_data_dict = load_holog_file(
        holog_chunk_params["holog_name"],
        dask_load=False,
        load_pnt_dict=False,
        ant_id=holog_chunk_params["this_ant"],
        ddi_id=holog_chunk_params["this_ddi"],
    )
    label = create_dataset_label(
        holog_chunk_params["this_ant"], holog_chunk_params["this_ddi"], separator=","
    )
    logger.info(f"Processing {label}")
    ddi = holog_chunk_params["this_ddi"]
    convert_to_stokes = holog_chunk_params["to_stokes"]
    ref_xds = ant_data_dict[ddi]["map_0"]
    summary = ref_xds.attrs["summary"]

    user_grid_size = holog_chunk_params["grid_size"]

    if user_grid_size is None:
        grid_size = np.array(summary["beam"]["grid size"])
    elif isinstance(user_grid_size, int):
        grid_size = np.array([user_grid_size, user_grid_size])
    elif isinstance(user_grid_size, (list, np.ndarray)):
        grid_size = np.array(user_grid_size)
    else:
        raise Exception(
            f"Don't know what due with grid size of type {type(user_grid_size)}"
        )

    logger.info(
        f"{label}: Using a grid of {grid_size[0]} by {grid_size[1]} pixels for the beam"
    )

    user_cell_size = holog_chunk_params["cell_size"]
    if user_cell_size is None:
        cell_size = np.array(
            [-summary["beam"]["cell size"], summary["beam"]["cell size"]]
        )
    elif isinstance(user_cell_size, (int, float)):
        cell_size = np.array([-user_cell_size, user_cell_size])
    elif isinstance(user_cell_size, (list, np.ndarray)):
        cell_size = np.array(user_cell_size)
    else:
        raise Exception(
            f"Don't know what due with cell size of type {type(user_cell_size)}"
        )

    logger.info(
        f"{label}: Using a cell size of {format_angular_distance(cell_size[0])} by "
        f"{format_angular_distance(cell_size[1])} for the beam"
    )

    telescope = _get_correct_telescope(
        summary["general"]["antenna name"], summary["general"]["telescope name"]
    )
    try:
        is_near_field = ref_xds.attrs["near_field"]
    except KeyError:
        is_near_field = False

    (
        beam_grid,
        time_centroid,
        freq_axis,
        pol_axis,
        l_axis,
        m_axis,
        grid_corr,
        summary,
    ) = grid_beam(
        ant_ddi_dict=ant_data_dict[ddi],
        grid_size=grid_size,
        sky_cell_size=cell_size,
        avg_chan=holog_chunk_params["chan_average"],
        chan_tol_fac=holog_chunk_params["chan_tolerance_factor"],
        telescope=telescope,
        grid_interpolation_mode=holog_chunk_params["grid_interpolation_mode"],
        observation_summary=summary,
        label=label,
    )

    if not is_near_field:
        beam_grid = parallactic_derotation(
            data=beam_grid, parallactic_angle_dict=ant_data_dict[ddi]
        )

    if holog_chunk_params["scan_average"]:
        beam_grid = np.mean(beam_grid, axis=0)[None, ...]
        time_centroid = np.mean(np.array(time_centroid))

    # Current bottleneck
    if is_near_field:
        distance, focus_offset = telescope.dist_dict[holog_chunk_params["alma_osf_pad"]]
        aperture_grid, u_axis, v_axis, _, used_wavelength = (
            calculate_near_field_aperture(
                grid=beam_grid,
                sky_cell_size=holog_chunk_params["cell_size"],
                distance=distance,
                freq=freq_axis,
                padding_factor=holog_chunk_params["padding_factor"],
                focus_offset=focus_offset,
                telescope=telescope,
                apply_grid_correction=grid_corr,
                label=label,
            )
        )
    else:
        focus_offset = 0
        aperture_grid, u_axis, v_axis, _, used_wavelength = (
            calculate_far_field_aperture(
                grid=beam_grid,
                padding_factor=holog_chunk_params["padding_factor"],
                freq=freq_axis,
                telescope=telescope,
                sky_cell_size=cell_size,
                apply_grid_correction=grid_corr,
                label=label,
            )
        )
    zernike_n_order = holog_chunk_params["zernike_n_order"]
    zernike_coeffs, zernike_model, zernike_rms, osa_coeff_list = (
        fit_zernike_coefficients(
            aperture_grid, u_axis, v_axis, zernike_n_order, telescope
        )
    )

    orig_pol_axis = pol_axis
    if convert_to_stokes:
        beam_grid = convert_5d_grid_to_stokes(beam_grid, pol_axis)
        aperture_grid = convert_5d_grid_to_stokes(aperture_grid, pol_axis)
        pol_axis = ["I", "Q", "U", "V"]

    amplitude, phase, u_prime, v_prime = _crop_and_split_aperture(
        aperture_grid, u_axis, v_axis, telescope
    )

    phase_fit_engine = holog_chunk_params["phase_fit_engine"]

    if phase_fit_engine is None or phase_fit_engine == "none":
        phase_corrected_angle, phase_fit_results = skip_phase_fitting(label, phase)
    else:
        if is_near_field:
            phase_corrected_angle, phase_fit_results = clic_like_phase_fitting(
                phase, freq_axis, telescope, focus_offset, u_prime, v_prime, label
            )
        else:
            if phase_fit_engine == "perturbations":
                phase_corrected_angle, phase_fit_results = aips_like_phase_fitting(
                    amplitude,
                    phase,
                    pol_axis,
                    freq_axis,
                    telescope,
                    u_axis,
                    v_axis,
                    holog_chunk_params["phase_fit_control"],
                    label,
                )
            elif phase_fit_engine == "zernike":
                if zernike_n_order > 4:
                    logger.warning(
                        "Using a Zernike order > 4 for phase fitting may result in overfitting"
                    )

                if convert_to_stokes:
                    zernike_grid = convert_5d_grid_to_stokes(
                        zernike_model, orig_pol_axis
                    )
                else:
                    zernike_grid = zernike_model.copy()

                _, zernike_phase, _, _ = _crop_and_split_aperture(
                    zernike_grid, u_axis, v_axis, telescope
                )

                phase_corrected_angle = phase_wrapping(phase - zernike_phase)
                phase_fit_results = None
            else:
                logger.error(f"Unsupported phase fitting engine: {phase_fit_engine}")
                raise ValueError

    summary["aperture"] = _get_aperture_summary(
        u_axis, v_axis, _compute_aperture_resolution(l_axis, m_axis, used_wavelength)
    )

    _export_to_xds(
        beam_grid,
        aperture_grid,
        amplitude,
        phase_corrected_angle,
        holog_chunk_params["this_ant"],
        time_centroid,
        ddi,
        phase_fit_results,
        pol_axis,
        freq_axis,
        l_axis,
        m_axis,
        u_axis,
        v_axis,
        u_prime,
        v_prime,
        orig_pol_axis,
        osa_coeff_list,
        zernike_coeffs,
        zernike_model,
        zernike_rms,
        zernike_n_order,
        holog_chunk_params["image_name"],
        summary,
    )

    logger.info(f"Finished processing {label}")


def _get_correct_telescope(ant_name, telescope_name):
    # Get telescope info
    if ant_name.upper().__contains__("DV"):
        telescope_name = "_".join((telescope_name, "DV"))

    elif ant_name.upper().__contains__("DA"):
        telescope_name = "_".join((telescope_name, "DA"))

    elif ant_name.upper().__contains__("EA"):
        telescope_name = "VLA"

    else:
        raise Exception("Antenna type not found: {name}".format(name=ant_name))

    return Telescope(telescope_name)


def _crop_and_split_aperture(aperture_grid, u_axis, v_axis, telescope, scaling=1.5):
    # Default scaling factor is now 1.5 to allow for better analysis of the noise around the aperture.
    # This will probably mean no cropping for most apertures, but may be important if dish appears too small in the
    # aperture.
    max_aperture_radius = 0.5 * telescope.diam

    image_slice = aperture_grid[0, 0, 0, ...]
    center_pixel = np.array(image_slice.shape[0:2]) // 2
    radius_u = int(
        np.where(np.abs(u_axis) < max_aperture_radius * scaling)[0].max()
        - center_pixel[0]
    )
    radius_v = int(
        np.where(np.abs(v_axis) < max_aperture_radius * scaling)[0].max()
        - center_pixel[1]
    )

    if radius_v > radius_u:
        radius = radius_v
    else:
        radius = radius_u

    start_cut = center_pixel - radius
    end_cut = center_pixel + radius

    amplitude = np.absolute(
        aperture_grid[..., start_cut[0] : end_cut[0], start_cut[1] : end_cut[1]]
    )
    phase = np.angle(
        aperture_grid[..., start_cut[0] : end_cut[0], start_cut[1] : end_cut[1]]
    )
    return (
        amplitude,
        phase,
        u_axis[start_cut[0] : end_cut[0]],
        v_axis[start_cut[1] : end_cut[1]],
    )


def _compute_aperture_resolution(l_axis, m_axis, wavelength):
    # Here we compute the aperture resolution from Equation 7 In EVLA memo 212
    # https://library.nrao.edu/public/memos/evla/EVLAM_212.pdf
    deltal = np.max(l_axis) - np.min(l_axis)
    deltam = np.max(m_axis) - np.min(m_axis)
    aperture_resolution = np.array([1 / deltal, 1 / deltam])
    aperture_resolution *= 1.27 * wavelength
    return aperture_resolution


def _export_to_xds(
    beam_grid,
    aperture_grid,
    amplitude,
    phase_corrected_angle,
    ant_id,
    time_centroid,
    ddi,
    phase_fit_results,
    pol_axis,
    freq_axis,
    l_axis,
    m_axis,
    u_axis,
    v_axis,
    u_prime,
    v_prime,
    orig_pol_axis,
    osa_coeff_list,
    zernike_coeffs,
    zernike_model,
    zernike_rms,
    zernike_n_order,
    image_name,
    summary,
):
    # Todo: Add Paralactic angle as a non-dimension coordinate dependant on time.
    xds = xr.Dataset()

    xds["BEAM"] = xr.DataArray(beam_grid, dims=["time", "chan", "pol", "l", "m"])
    xds["APERTURE"] = xr.DataArray(
        aperture_grid, dims=["time", "chan", "pol", "u", "v"]
    )

    xds["AMPLITUDE"] = xr.DataArray(
        amplitude, dims=["time", "chan", "pol", "u_prime", "v_prime"]
    )
    xds["CORRECTED_PHASE"] = xr.DataArray(
        phase_corrected_angle, dims=["time", "chan", "pol", "u_prime", "v_prime"]
    )

    xds["ZERNIKE_COEFFICIENTS"] = xr.DataArray(
        zernike_coeffs, dims=["time", "chan", "orig_pol", "osa"]
    )
    xds["ZERNIKE_MODEL"] = xr.DataArray(
        zernike_model, dims=["time", "chan", "orig_pol", "u", "v"]
    )
    xds["ZERNIKE_FIT_RMS"] = xr.DataArray(
        zernike_rms, dims=["time", "chan", "orig_pol"]
    )

    xds.attrs["ant_id"] = ant_id
    xds.attrs["time_centroid"] = np.array(time_centroid)
    xds.attrs["ddi"] = ddi
    xds.attrs["phase_fitting"] = phase_fit_results
    xds.attrs["zernike_N_order"] = zernike_n_order
    xds.attrs["summary"] = summary

    coords = {
        "orig_pol": orig_pol_axis,
        "pol": pol_axis,
        "l": l_axis,
        "m": m_axis,
        "u": u_axis,
        "v": v_axis,
        "u_prime": u_prime,
        "v_prime": v_prime,
        "chan": freq_axis,
        "osa": osa_coeff_list,
    }
    xds = xds.assign_coords(coords)
    xds.to_zarr(
        f"{image_name}/{ant_id}/{ddi}", mode="w", compute=True, consolidated=True
    )


def _get_aperture_summary(u_axis, v_axis, aperture_resolution):
    aperture_dict = {
        "grid size": [u_axis.shape[0], v_axis.shape[0]],
        "cell size": [u_axis[1] - u_axis[0], v_axis[1] - v_axis[0]],
        "resolution": aperture_resolution.tolist(),
    }
    return aperture_dict
