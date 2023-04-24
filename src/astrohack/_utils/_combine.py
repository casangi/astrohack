import numpy as np
import os
import xarray as xr

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._io import _load_image_xds


def _combine_chunk(combine_chunk_params):
    """
    Process a combine chunk
    Args:
        combine_chunk_params: Param dictionary for combine chunk
    """
    logger = _get_astrohack_logger()

    data_path = combine_chunk_params['image_file']+'/'+combine_chunk_params['antenna']
    if combine_chunk_params['ddi_list'] == 'all':
        ddi_list = os.listdir(data_path)
    else:
        ddi_list = combine_chunk_params['ddi_list']

    nddi = len(ddi_list)
    out_xds_name = combine_chunk_params['combine_file'] + '/' + combine_chunk_params['antenna'] + '/' + ddi_list[0]
    if nddi == 0:
        logger.warning('Nothing to process for ant_id: '+combine_chunk_params['antenna'])
        return
    elif nddi == 1:
        logger.info(combine_chunk_params['antenna']+' has a single ddi to be combined, data copied from input file')

        out_xds = _load_image_xds(combine_chunk_params['image_file'],
                                  combine_chunk_params['antenna'],
                                  ddi_list[0],
                                  dask_load=False)
        out_xds.to_zarr(out_xds_name, mode='w')
    else:
        out_xds = _load_image_xds(combine_chunk_params['image_file'],
                                  combine_chunk_params['antenna'],
                                  ddi_list[0],
                                  dask_load=False)
        nddi = len(ddi_list)
        shape = out_xds['CORRECTED_PHASE'].values.shape
        shape.append(nddi)
        multi_ddi_phase = np.ndarray(shape)
        multi_ddi_amp = np.zeros_like(multi_ddi_phase)

        for iddi in range(nddi):
            this_xds = _load_image_xds(combine_chunk_params['image_file'],
                                       combine_chunk_params['antenna'],
                                       ddi_list[iddi],
                                       dask_load=False)
            multi_ddi_amp[iddi] = this_xds['AMPLITUDE']
            multi_ddi_phase[iddi] = this_xds['CORRECTED_PHASE']

        amplitude = np.mean(multi_ddi_amp, axis=-1)
        if combine_chunk_params['weighted']:
            phase = np.average(multi_ddi_phase, axis=-1, weights=multi_ddi_amp)
        else:
            phase = np.mean(multi_ddi_phase, axis=-1)

        out_xds['AMPLITUDE'] = xr.DataArray(amplitude, dims=["time", "chan", "pol", "u_prime", "v_prime"])
        out_xds['CORRECTED_PHASE'] = xr.DataArray(phase, dims=["time", "chan", "pol", "u_prime", "v_prime"])

        out_xds.to_zarr(out_xds_name, mode='w')

