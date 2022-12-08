import os
import dask

import xarray as xr
import numpy as np

from casacore import tables as ctables

from astrohack._utils._io import _load_pnt_dict, _make_ant_pnt_dict 
from astrohack._utils._io import _extract_holog_chunk, _open_no_dask_zarr
from astrohack._utils._io import _create_hack_meta_data, _read_data_from_hack_meta

def load_hack_file(hack_name, dask_load=True, load_pnt_dict=True, ant_id=None): 
    """ Loads .hack file from disk

    Args:
        hack_name (str): Hack file name

    Returns:
        hackfile (nested-dict): {
                            'pnt.dict':{}, 'ddi':
                                                {'scan':
                                                    {'antenna':
                                                        {
                                                            xarray.DataArray
                                                        }
                                                    }
                                                }
                        }
    """
    
    hack_dict = {}
    
    if load_pnt_dict == True:
        hack_dict['pnt_dict'] = _load_pnt_dict(file=os.path.join(hack_name, 'pnt.dict'), ant_list=None, dask_load=dask_load)

    for ddi in os.listdir(hack_name):
        if ddi.isnumeric():
            hack_dict[int(ddi)] = {}
            for scan in os.listdir(os.path.join(hack_name,ddi)):
                if scan.isnumeric():
                    hack_dict[int(ddi)][int(scan)]={}
                    for ant in os.listdir(os.path.join(hack_name,ddi+'/'+scan)):
                        if ant.isnumeric():
                            mapping_ant_vis_holog_data_name = os.path.join(hack_name,ddi+'/'+scan+'/'+ant)
                            
                            if dask_load:
                                hack_dict[int(ddi)][int(scan)][int(ant)] = xr.open_zarr(mapping_ant_vis_holog_data_name)
                            else:
                                hack_dict[int(ddi)][int(scan)][int(ant)] = _open_no_dask_zarr(mapping_ant_vis_holog_data_name)

    if ant_id == None:
        return hack_dict

    return hack_dict, _read_data_from_hack_meta(hack_name=hack_name, hack_dict=hack_dict, ant_id=ant_id)

def extract_holog(ms_name, hack_name, holog_obs_dict, data_col='DATA', subscan_intent='MIXED', parallel=True):
    """ Extract holography data and create beam maps.
            subscan_intent: 'MIXED' or 'REFERENCE'

    Args:
        ms_name (string): measurement file name
        holog_obs_dict (dict): nested dictionary ordered by ddi:{ scan: { map:[ant names], ref:[ant names] } } }
        data_col (str, optional): data column from measurement set to acquire. Defaults to 'DATA'.
        subscan_intent (str, optional): subscan intent, can be MIXED or REFERENCE; MIXED refers to a pointing measurement with half ON(OFF) source. Defaults to 'MIXED'.
        parallel (bool, optional): Bool for whether to process in parallel. Defaults to True.
    """
    
    pnt_name = os.path.join(hack_name,'pnt.dict')
    _make_ant_pnt_dict(ms_name, pnt_name, parallel=parallel)
    
    ######## Get Spectral Windows ########
    
    # nomodify=True when using CASA tables.
    print(os.path.join(ms_name,"DATA_DESCRIPTION"))

    ctb = ctables.table(os.path.join(ms_name,"DATA_DESCRIPTION"), readonly=True, lockoptions={'option': 'usernoread'}, ack=False) 
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
            
    spw_ctb = ctables.table(os.path.join(ms_name,"SPECTRAL_WINDOW"), readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    pol_ctb = ctables.table(os.path.join(ms_name,"POLARIZATION"), readonly=True, lockoptions={'option': 'usernoread'}, ack=False)

    delayed_list = []
    for ddi in holog_obs_dict:
        
        spw_setup_id = ddi_spw[ddi]
        pol_setup_id = ddi_pol[ddi]
    
        extract_holog_parms = {
            'ms_name':ms_name,
            'hack_name':hack_name,
            'pnt_name':pnt_name,
            'ddi':ddi,
            'data_col':data_col,
            'chan_setup':{},
            'pol_setup':{}
        }

        extract_holog_parms['chan_setup']['chan_freq'] = spw_ctb.getcol('CHAN_FREQ', startrow=spw_setup_id,nrow=1)[0,:]
        extract_holog_parms['chan_setup']['chan_width'] = spw_ctb.getcol('CHAN_WIDTH', startrow=spw_setup_id,nrow=1)[0,:]
        extract_holog_parms['chan_setup']['eff_bw'] = spw_ctb.getcol('EFFECTIVE_BW', startrow=spw_setup_id,nrow=1)[0,:]
        extract_holog_parms['chan_setup']['ref_freq'] = spw_ctb.getcol('REF_FREQUENCY', startrow=spw_setup_id,nrow=1)[0]
        extract_holog_parms['chan_setup']['total_bw'] = spw_ctb.getcol('TOTAL_BANDWIDTH', startrow=spw_setup_id,nrow=1)[0]

        extract_holog_parms['pol_setup']['pol'] = pol_ctb.getcol('CORR_TYPE',startrow=spw_setup_id,nrow=1)[0,:]
        
        for scan in holog_obs_dict[ddi].keys():
            print('Processing ddi: {ddi}, scan: {scan}'.format(ddi=ddi, scan=scan))
            
            map_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]['map']))[0]
            ref_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]['ref']))[0]

            extract_holog_parms['map_ant_ids'] = map_ant_ids
            extract_holog_parms['map_ant_names'] = holog_obs_dict[ddi][scan]['map']
            extract_holog_parms['ref_ant_ids'] = ref_ant_ids
            extract_holog_parms['sel_state_ids'] = state_ids
            extract_holog_parms['scan'] = scan
         
            
            if parallel:
                delayed_list.append(
                    dask.delayed(_extract_holog_chunk)(
                        dask.delayed(extract_holog_parms)
                    )
                )
            else:
                _extract_holog_chunk(extract_holog_parms)
                
    spw_ctb.close()
    pol_ctb.close()
    
    if parallel:
        dask.compute(delayed_list)
    
    print("Finished dask compute ...")
    hack_dict = load_hack_file(hack_name=extract_holog_parms['hack_name'], dask_load=True, load_pnt_dict=False)                            
    _create_hack_meta_data(hack_name=extract_holog_parms['hack_name'], hack_dict=hack_dict)