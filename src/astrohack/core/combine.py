from copy import deepcopy

import numpy as np
import toolviper.utils.logger as logger
import xarray as xr
from scipy.interpolate import griddata

from astrohack.io.image_mds import AstrohackImageFile
from astrohack.utils.constants import clight
from astrohack.utils.text import create_dataset_label, param_to_list


def process_combine_chunk(combine_chunk_params: dict, output_mds: AstrohackImageFile):
    """
    Process a combine chunk
    Args:
        combine_chunk_params: Param dictionary for combine chunk
        output_mds: output mds file that contains combined data
    """

    ant_key = combine_chunk_params["this_ant"]
    ant_xdt = combine_chunk_params["xdt_data"]
    user_ddi_sel = combine_chunk_params["ddi"]
    ddi_list = param_to_list(user_ddi_sel, ant_xdt, "ddi")
    dataset_label = create_dataset_label(ant_key, None)

    nddi = len(ddi_list)
    if nddi == 0:
        logger.warning(f"Nothing to process for {ant_key}")
        return
    elif nddi == 1:
        ddi_key = ddi_list[0]
        if ddi_key in list(ant_xdt.keys()):
            logger.info(
                f"{dataset_label} has a single ddi to be combined, data copied from input file"
            )
            output_mds.add_node(ant_xdt[ddi_key].dataset, [ant_key, ddi_key])
        else:
            logger.warning(
                f"{dataset_label} has no {ddi_key}, nothing to process for this antenna"
            )
            return
    else:
        ddi_in_xdt_list = list(ant_xdt.keys())
        ddi_present_list = [ddi_key in ddi_in_xdt_list for ddi_key in ddi_list]
        if np.sum(ddi_present_list) == 0:
            logger.warning(
                f"{dataset_label} has no valid DDI in user selection (ddi = {user_ddi_sel})"
            )
            return

        min_freq = 1e34
        ddi_ref_key = None
        summary_dict = {}
        for i_ddi, ddi_key in enumerate(ddi_list):
            if not ddi_present_list[i_ddi]:
                continue
            summary = ant_xdt[ddi_key].attrs["summary"]
            summary_dict[ddi_key] = summary
            rep_freq = summary["spectral"]["rep. frequency"]
            if rep_freq < min_freq:
                min_freq = rep_freq
                ddi_ref_key = ddi_key

        out_xds = deepcopy(ant_xdt[ddi_ref_key].dataset)
        shape = list(out_xds["CORRECTED_PHASE"].values.shape)
        if out_xds.sizes["chan"] != 1:
            msg = "Only single channel holographies supported"
            logger.error(msg)
            raise RuntimeError(msg)

        nmap = shape[0]
        if nmap != 1:
            msg = "Only single mapping holographies supported"
            logger.error(msg)
            raise RuntimeError(msg)

        npol = shape[2]
        npoints = shape[3] * shape[4]
        amp_sum = np.zeros((npol, npoints))
        pha_sum = np.zeros((npol, npoints))

        u_mesh, v_mesh = np.meshgrid(out_xds.u_prime.values, out_xds.v_prime.values)
        dest_u_axis = u_mesh.ravel()
        dest_v_axis = v_mesh.ravel()

        for i_ddi, ddi_key in enumerate(ddi_list):
            this_dataset_label = create_dataset_label(ant_key, ddi_key)
            if not ddi_present_list[i_ddi]:
                logger.warning(
                    f"{this_dataset_label} does not exist in input mds, skipping"
                )
                continue
            this_xds = ant_xdt[ddi_key].dataset

            u_mesh, v_mesh = np.meshgrid(
                this_xds.u_prime.values,
                this_xds.v_prime.values,
            )
            loca_u_axis = u_mesh.ravel()
            loca_v_axis = v_mesh.ravel()

            if loca_u_axis.shape[0] == dest_u_axis.shape[0]:
                resample_needed = not (
                    np.allclose(loca_u_axis, dest_u_axis, rtol=1e-6)
                    and np.allclose(loca_v_axis, dest_v_axis, rtol=1e-6)
                )
            else:
                resample_needed = True

            if resample_needed:
                logger.info(f"Regridding {this_dataset_label}")
                for ipol in range(npol):
                    thispha = (
                        this_xds["CORRECTED_PHASE"].values[0, 0, ipol, :, :].ravel()
                    )
                    thisamp = this_xds["AMPLITUDE"].values[0, 0, ipol, :, :].ravel()

                    repha = griddata(
                        (loca_u_axis, loca_v_axis),
                        thispha,
                        (dest_u_axis, dest_v_axis),
                        method="linear",
                    )
                    reamp = griddata(
                        (loca_u_axis, loca_v_axis),
                        thisamp,
                        (dest_u_axis, dest_v_axis),
                        method="linear",
                    )
                    amp_sum[ipol, :] += reamp
                    if combine_chunk_params["weighted"]:
                        pha_sum[ipol, :] += repha * reamp
                    else:
                        pha_sum[ipol, :] += repha
            else:
                logger.info(
                    f"{this_dataset_label} already has the proper sampling, simple addition"
                )
                for ipol in range(npol):
                    thispha = (
                        this_xds["CORRECTED_PHASE"].values[0, 0, ipol, :, :].ravel()
                    )
                    thisamp = this_xds["AMPLITUDE"].values[0, 0, ipol, :, :].ravel()
                    amp_sum[ipol, :] += thisamp
                    if combine_chunk_params["weighted"]:
                        pha_sum[ipol, :] += thispha * thisamp
                    else:
                        pha_sum[ipol, :] += thispha

        n_used_ddi = np.sum(ddi_present_list)
        if combine_chunk_params["weighted"]:
            phase = pha_sum / amp_sum
        else:
            phase = pha_sum / n_used_ddi
        amplitude = amp_sum / n_used_ddi

        out_xds["AMPLITUDE"] = xr.DataArray(
            amplitude.reshape(shape), dims=["time", "chan", "pol", "u_prime", "v_prime"]
        )
        out_xds["CORRECTED_PHASE"] = xr.DataArray(
            phase.reshape(shape), dims=["time", "chan", "pol", "u_prime", "v_prime"]
        )

        out_ddi_key = "ddi_99"
        out_xds.attrs["ddi"] = out_ddi_key
        out_xds.attrs["summary"] = _merge_summary_dict(summary_dict, ddi_ref_key)
        output_mds.add_node(out_xds, [ant_key, out_ddi_key])


def _merge_summary_dict(summary_dict, ddi_ref_key):
    out_summary = deepcopy(summary_dict[ddi_ref_key])

    aperture_resolution = out_summary["aperture"]["resolution"]
    frequency_range = out_summary["spectral"]["frequency range"]

    channel_width = 0.0
    rep_freq = 0.0

    n_used_ddi = 0
    for ddi_key, ddi_summary in summary_dict.items():
        # Spectral part
        n_used_ddi += 1
        channel_width += ddi_summary["spectral"]["channel width"]
        rep_freq += ddi_summary["spectral"]["rep. frequency"]
        frequency_range[0] = float(
            np.min([frequency_range[0], ddi_summary["spectral"]["frequency range"][0]])
        )
        frequency_range[1] = float(
            np.max([frequency_range[1], ddi_summary["spectral"]["frequency range"][1]])
        )
        for i_coord in range(2):
            aperture_resolution[i_coord] = float(
                np.max(
                    [
                        aperture_resolution[i_coord],
                        ddi_summary["aperture"]["resolution"][i_coord],
                    ]
                )
            )

    rep_freq /= n_used_ddi

    out_summary["aperture"]["resolution"] = aperture_resolution
    out_summary["spectral"]["frequency range"] = frequency_range
    out_summary["spectral"]["rep. frequency"] = rep_freq
    out_summary["spectral"]["channel width"] = channel_width
    out_summary["spectral"]["rep. wavelength"] = clight / rep_freq

    return out_summary
