import dask
import time
import os

import numpy as np
import xarray as xr
import dask.array as da

from numba import njit
from numba.core import types
from numba.typed import Dict

from ._utils import load_pnt_dict
from astrohack._utils import make_ant_pnt_dict, exstract_holog_chunk

from casacore import tables as ctables


def exstract_holog(ms_name, hack_name, holog_obs_dict, data_col='DATA', subscan_intent='MIXED', parallel=True):
    """subscan_intent: 'MIXED' or 'REFERENCE'

    Args:
        ms_name (string): measurement file name
        holog_obs_dict (dict): nested dictionary ordered by ddi:{ scan: { map:[ant names], ref:[ant names] } } }
        data_col (str, optional): data column from measurement set to acquire. Defaults to 'DATA'.
        subscan_intent (str, optional): subscan intent, can be MIXED or REFERENCE; MIXED refers to a pointing measurement with half ON(OFF) source. Defaults to 'MIXED'.
        parallel (bool, optional): bool for whether to process in parallel. Defaults to True.
    """
    
    pnt_name = os.path.join(hack_name,'pnt.dict')
    make_ant_pnt_dict(ms_name, pnt_name, parallel=parallel)
    
    ######## Get Spectral Windows ########
    
    # nomodify=True when using CASA tables.
    print(os.path.join(ms_name,"DATA_DESCRIPTION"))

    ctb = ctables.table(os.path.join(ms_name,"DATA_DESCRIPTION"),readonly=True, lockoptions={'option': 'usernoread'}, ack=False) 
    ddi_spw = ctb.getcol("SPECTRAL_WINDOW_ID")
    ddi_pol = ctb.getcol("POLARIZATION_ID")
    ddi = np.arange(len(ddi_spw))
    ctb.close()
    
    ######## Get Antenna IDs and Names ########
    ctb = ctables.table(os.path.join(ms_name,"ANTENNA"), readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    ant_name = ctb.getcol("NAME")
    ant_id = np.arange(len(ant_name))
    ctb.close()
    
    ######## Get Scan and Subscan IDs ########
    # SDM Tables Short Description (https://drive.google.com/file/d/16a3g0GQxgcO7N_ZabfdtexQ8r2jRbYIS/view)
    # 2.54 ScanIntent (p. 150)
    #MAP ANTENNA SURFACE : Holography calibration scan
    
    # 2.61 SubscanIntent (p. 152)
    # MIXED : Pointing measurement, some antennas are on -ource, some off-source
    # REFERENCE : reference measurement (used for boresight in holography).
    # Undefined : ?
    
    ctb = ctables.table(os.path.join(ms_name,"STATE"), readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    
    # scan intent (with subscan intent) is stored in the OBS_MODE column of the STATE subtable.
    obs_modes = ctb.getcol("OBS_MODE") 
    ctb.close()
    
    scan_intent = 'MAP_ANTENNA_SURFACE'
    state_ids = []
    for i, mode in enumerate(obs_modes):
        if (scan_intent in mode) and (subscan_intent in mode):
            state_ids.append(i)
            
            
    spw_ctb = ctables.table(os.path.join(ms_name,"SPECTRAL_WINDOW"),readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    pol_ctb = ctables.table(os.path.join(ms_name,"POLARIZATION"),readonly=True, lockoptions={'option': 'usernoread'}, ack=False)

    delayed_list = []
    for ddi in holog_obs_dict:
        
        spw_setup_id = ddi_spw[ddi]
        pol_setup_id = ddi_pol[ddi]
    
        exstract_holog_parms = {'ms_name':ms_name,'hack_name':hack_name,'pnt_name':pnt_name,'ddi':ddi,'data_col':data_col,'chan_setup':{},'pol_setup':{}}
        exstract_holog_parms['chan_setup']['chan_freq'] = spw_ctb.getcol('CHAN_FREQ',startrow=spw_setup_id,nrow=1)[0,:]
        exstract_holog_parms['chan_setup']['chan_width'] = spw_ctb.getcol('CHAN_WIDTH',startrow=spw_setup_id,nrow=1)[0,:]
        exstract_holog_parms['chan_setup']['eff_bw'] = spw_ctb.getcol('EFFECTIVE_BW',startrow=spw_setup_id,nrow=1)[0,:]
        exstract_holog_parms['chan_setup']['ref_freq'] = spw_ctb.getcol('REF_FREQUENCY',startrow=spw_setup_id,nrow=1)[0]
        exstract_holog_parms['chan_setup']['total_bw'] = spw_ctb.getcol('TOTAL_BANDWIDTH',startrow=spw_setup_id,nrow=1)[0]

        exstract_holog_parms['pol_setup']['pol'] = pol_ctb.getcol('CORR_TYPE',startrow=spw_setup_id,nrow=1)[0,:]
        
        for scan in holog_obs_dict[ddi].keys():
            print('Processing ddi: {ddi}, scan: {scan}'.format(ddi=ddi, scan=scan))
            map_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]['map']))[0]
            ref_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]['ref']))[0]

            exstract_holog_parms['map_ant_ids'] = map_ant_ids
            exstract_holog_parms['map_ant_names'] = holog_obs_dict[ddi][scan]['map']
            exstract_holog_parms['ref_ant_ids'] = ref_ant_ids
            exstract_holog_parms['sel_state_ids'] = state_ids
            exstract_holog_parms['scan'] = scan
         
            
            if parallel:
                delayed_list.append(
                    dask.delayed(exstract_holog_chunk)(
                        dask.delayed(exstract_holog_parms)
                    )
                )
            else:
                exstract_holog_chunk(exstract_holog_parms)
                
    spw_ctb.close()
    pol_ctb.close()
    
    if parallel:
        dask.compute(delayed_list)
    



