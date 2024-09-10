import numpy as np
import xarray as xr

import toolviper.utils.logger as logger

from astrohack.utils.file import load_image_xds
from scipy.interpolate import griddata
from astrohack.utils.constants import clight
from astrohack.utils.text import param_to_list


def process_combine_chunk(combine_chunk_params):
    """
    Process a combine chunk
    Args:
        combine_chunk_params: Param dictionary for combine chunk
    """

    antenna = combine_chunk_params['this_ant']
    ddi_dict = combine_chunk_params['image_mds'][antenna]
    ddi_list = param_to_list(combine_chunk_params['ddi'], ddi_dict, 'ddi')

    nddi = len(ddi_list)
    out_xds_name = '/'.join([combine_chunk_params['combine_name'], antenna, ddi_list[0]])
    if nddi == 0:
        logger.warning(f'Nothing to process for {antenna}')
        return
    elif nddi == 1:
        logger.info(f'{antenna} has a single ddi to be combined, data copied from input file')
        out_xds = ddi_dict[ddi_list[0]]
        out_xds.to_zarr(out_xds_name, mode='w')
    else:
        out_xds = load_image_xds(combine_chunk_params['image_name'], antenna, ddi_list[0], dask_load=False)
        nddi = len(ddi_list)
        shape = list(out_xds['CORRECTED_PHASE'].values.shape)
        if out_xds.sizes['chan'] != 1:
            msg = f'Only single channel holographies supported'
            logger.error(msg)
            raise Exception(msg)
        npol = shape[2]
        npoints = shape[3] * shape[4]
        amp_sum = np.zeros((npol, npoints))
        pha_sum = np.zeros((npol, npoints))
        for ipol in range(npol):
            amp_sum[ipol, :] = out_xds['AMPLITUDE'].values[0, 0, ipol, :, :].ravel()
            if combine_chunk_params['weighted']:
                pha_sum[ipol, :] = out_xds['CORRECTED_PHASE'].values[0, 0, ipol, :, :].ravel() * amp_sum[ipol, :]
            else:
                pha_sum[ipol, :] = out_xds['CORRECTED_PHASE'].values[0, 0, ipol, :, :].ravel()
        wavelength = clight / out_xds.chan.values[0]
        u, v = np.meshgrid(out_xds.u_prime.values * wavelength, out_xds.v_prime.values * wavelength)
        dest_u_axis = u.ravel()
        dest_v_axis = v.ravel()
        for iddi in range(1, nddi):
            logger.info(f'Regridding {antenna} {ddi_list[iddi]}')
            this_xds = load_image_xds(combine_chunk_params['image_name'], antenna, ddi_list[iddi], dask_load=False)
            wavelength = clight / this_xds.chan.values[0]
            u, v = np.meshgrid(this_xds.u_prime.values * wavelength, this_xds.v_prime.values * wavelength)
            loca_u_axis = u.ravel()
            loca_v_axis = v.ravel()
            for ipol in range(npol):
                thispha = this_xds['CORRECTED_PHASE'].values[0, 0, ipol, :, :].ravel()
                thisamp = this_xds['AMPLITUDE'].values[0, 0, ipol, :, :].ravel()
                repha = griddata((loca_u_axis, loca_v_axis), thispha, (dest_u_axis, dest_v_axis), method='linear')
                reamp = griddata((loca_u_axis, loca_v_axis), thisamp, (dest_u_axis, dest_v_axis), method='linear')
                amp_sum[ipol, :] += reamp
                if combine_chunk_params['weighted']:
                    pha_sum[ipol, :] += repha * reamp
                else:
                    pha_sum[ipol, :] += repha

        if combine_chunk_params['weighted']:
            phase = pha_sum / amp_sum
        else:
            phase = pha_sum / nddi
        amplitude = amp_sum / nddi

        out_xds['AMPLITUDE'] = xr.DataArray(amplitude.reshape(shape),
                                            dims=["time", "chan", "pol", "u_prime", "v_prime"])
        out_xds['CORRECTED_PHASE'] = xr.DataArray(phase.reshape(shape),
                                                  dims=["time", "chan", "pol", "u_prime", "v_prime"])

        out_xds.to_zarr(out_xds_name, mode='w')
