import os
import dask
import sys

import xarray as xr
import numpy as np

from casacore import tables as ctables

from astrohack._utils._constants import pol_str

from astrohack._utils._conversion import _convert_ant_name_to_id

from astrohack._utils._holog import _create_holog_meta_data
from astrohack._utils._holog import _make_ant_pnt_dict

from astrohack._utils._io import _load_pnt_dict 
from astrohack._utils._io import _extract_holog_chunk 
from astrohack._utils._io import _open_no_dask_zarr
from astrohack._utils._io import _read_data_from_holog_json
from astrohack._utils._io import _read_meta_data
from astrohack._utils._io import _load_holog_file
from astrohack._utils._io import  check_if_file_will_be_overwritten,check_if_file_exists
#from memory_profiler import profile


from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from  astrohack._utils._parm_utils._check_parms import _check_parm
from astrohack._utils._utils import _remove_suffix


#@profile(stream=fp)
def extract_holog(
    ms_name,
    holog_obs_dict,
    holog_name=None,
    point_name=None,
    data_col="DATA",
    parallel=False,
    overwrite=False,
):
    """
    From a measurement set with holography data extract the pointing information (saved into a folder with a .point.zarr extension) and holography data (saved into a folder with an extension .holog.zarr).

    Parameters
    ----------
    ms_name (string):
        Measurement file name.
    holog_obs_dict (dict):
        The holog_obs_dict describes which scans and antennas's data to exstract from the ms.
        For example:
        scans=[8,9,10,12,13,14,16,17,18,23,24,25,27,28,29,31,32,33,38,39,40,42,43,44,46,47,48,53,54,55,57]
        holog_obs_description = {0 :{'scans':scans,'ant':{'ea25':['ea04']}}}
        holog_obs_description['ddi'] = [0]
    holog_obs_dict[holog_mapping_id] (dict):
        A dictionary where each key is a holog_mapping_id. The holog_mapping_ids can be any numbers chosen arbiterily and represent one complete mapping of the beam. The holog_mapping_id is needed since the mapping of a beam can take more than one scan and an ms can have more than one mapping of the beam.
    holog_obs_dict[holog_mapping_id]['scans'] (int np.ndarray/list):
        All the scans in the ms that form part of the holog_mapping_id.
    holog_obs_dict[holog_mapping_id]['ant'] (dict):
        The keys are the mapping antenna names and the values lists of the reference antennas.
    holog_obs_dict[ddi] (int np.ndarray/list):
        All the ddi's in the ms from which data should be exstracted.
    holog_name (string, default= ms name with holog.zarr extension):
        Name of holog.zarr file to create.
    point_name (string, default= ms name with point.zarr extension):
        Name of point.zarr file to create.
    data_col (str, default='DATA'):
        Data column from measurement set to acquire.
    parallel (bool, default=False):
        Boolean for whether to process in parallel. If parallel processing is
    overwrite (bool, optional):
        Boolean for whether to overwrite current holog.zarr and point.zarr files.
    """
    logger = _get_astrohack_logger()
    
    
    extract_holog_parms = _check_extract_holog_parms(ms_name,
                                holog_obs_dict,
                                holog_name,
                                point_name,
                                data_col,
                                parallel,
                                overwrite)
    
    check_if_file_exists(extract_holog_parms['ms_name'])
    check_if_file_will_be_overwritten(extract_holog_parms['holog_name'],extract_holog_parms['overwrite'])
    check_if_file_will_be_overwritten(extract_holog_parms['point_name'],extract_holog_parms['overwrite'])

#    try:
#        pnt_dict = _load_pnt_dict(point_name)
#    except:
#        pnt_dict = _make_ant_pnt_dict(ms_name, point_name, parallel=parallel)
    
    pnt_dict = _make_ant_pnt_dict(extract_holog_parms['ms_name'], extract_holog_parms['point_name'], parallel=extract_holog_parms['parallel'])

    ''' VLA datasets causeing issues.
    if holog_obs_dict is None:
        ant_names_list = []
        #Create mapping antennas
        holog_obs_dict = {}
        for ant_id,pnt_xds in pnt_dict.items():
            ant_name = pnt_xds.attrs['ant_name']
            mapping_scans = pnt_xds.attrs['mapping_scans']
            ant_names_list.append(ant_name)
            for ddi,scans in  mapping_scans.items():
                ddi = int(ddi)
                for s in scans:
                    try:
                        holog_obs_dict[ddi][s]['map'].append(ant_name)
                    except:
                        holog_obs_dict.setdefault(ddi,{})[s] = {'map':[ant_name]}#dict(zip(scans, [ant_name]*len(scans)))
       
       
        #Create reference antennas
        
        ant_names_set = set(ant_names_list)
        for ddi,scan in holog_obs_dict.items():
            for scan_id,ant_types in scan.items():
                holog_obs_dict[ddi][scan_id]['ref'] = ant_names_set - set(holog_obs_dict[ddi][scan_id]['map'])
            
    print(holog_obs_dict)
    '''


    ######## Get Spectral Windows ########
    ctb = ctables.table(
        os.path.join(extract_holog_parms['ms_name'], "DATA_DESCRIPTION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    ddi_spw = ctb.getcol("SPECTRAL_WINDOW_ID")
    ddpol_indexol = ctb.getcol("POLARIZATION_ID")
    ddi = np.arange(len(ddi_spw))
    ctb.close()

    ######## Get Antenna IDs and Names ########
    ctb = ctables.table(
        os.path.join(extract_holog_parms['ms_name'], "ANTENNA"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ant_names = ctb.getcol("NAME")
    ant_id = np.arange(len(ant_names))

    ctb.close()

    ######## Get Scan and Subscan IDs ########
    # SDM Tables Short Description (https://drive.google.com/file/d/16a3g0GQxgcO7N_ZabfdtexQ8r2jRbYIS/view)
    # 2.54 ScanIntent (p. 150)
    # MAP ANTENNA SURFACE : Holography calibration scan

    # 2.61 SubscanIntent (p. 152)
    # MIXED : Pointing measurement, some antennas are on-source, some off-source
    # REFERENCE : reference measurement (used for boresight in holography).
    # Undefined : ?

    ctb = ctables.table(
        os.path.join(extract_holog_parms['ms_name'], "STATE"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    # scan intent (with subscan intent) is stored in the OBS_MODE column of the STATE subtable.
    obs_modes = ctb.getcol("OBS_MODE")
    ctb.close()

    scan_intent = "MAP_ANTENNA_SURFACE"
    state_ids = []

    for i, mode in enumerate(obs_modes):
        if (scan_intent in mode) and ('REFERENCE' not in mode):
            state_ids.append(i)

    spw_ctb = ctables.table(
        os.path.join(extract_holog_parms['ms_name'], "SPECTRAL_WINDOW"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    pol_ctb = ctables.table(
        os.path.join(extract_holog_parms['ms_name'], "POLARIZATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    obs_ctb = ctables.table(
        os.path.join(extract_holog_parms['ms_name'], "OBSERVATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    
    telescope_name = obs_ctb.getcol("TELESCOPE_NAME")[0]

    extract_holog_params = {}

    ## DDI selection
    if holog_obs_dict['ddi'] is None:
        logger.error("No DDIs in holog_obs_dict.")
        raise Exception()

    delayed_list = []
    
    for ddi in holog_obs_dict['ddi']:
        spw_setup_id = ddi_spw[ddi]
        pol_setup_id = ddpol_indexol[ddi]
        
        extract_holog_params["ddi"] = ddi
        extract_holog_params["chan_setup"] = {}
        extract_holog_params["pol_setup"] = {}
        
        extract_holog_params["chan_setup"]["chan_freq"] = spw_ctb.getcol("CHAN_FREQ", startrow=spw_setup_id, nrow=1)[0, :]
        extract_holog_params["chan_setup"]["chan_width"] = spw_ctb.getcol("CHAN_WIDTH", startrow=spw_setup_id, nrow=1)[0, :]
        extract_holog_params["chan_setup"]["eff_bw"] = spw_ctb.getcol("EFFECTIVE_BW", startrow=spw_setup_id, nrow=1)[0, :]
        extract_holog_params["chan_setup"]["ref_freq"] = spw_ctb.getcol("REF_FREQUENCY", startrow=spw_setup_id, nrow=1)[0]
        extract_holog_params["chan_setup"]["total_bw"] = spw_ctb.getcol("TOTAL_BANDWIDTH", startrow=spw_setup_id, nrow=1)[0]

        extract_holog_params["pol_setup"]["pol"] = pol_str[pol_ctb.getcol("CORR_TYPE", startrow=pol_setup_id, nrow=1)[0, :]]
                
        
        extract_holog_params["telescope_name"] = obs_ctb.getcol("TELESCOPE_NAME")[0]
        

        for holog_scan_id in holog_obs_dict.keys(): #loop over all beam_scan_ids, a beam_scan_id can conist out of more than one scan in an ms (this is the case for the VLA pointed mosiacs).
            if isinstance(holog_scan_id,int):
                scans = holog_obs_dict[holog_scan_id]["scans"]
                logger.info("Processing ddi: {ddi}, scans: {scans}".format(ddi=ddi, scans=scans))
            
                map_ant_list = []
                ref_ant_per_map_ant_list = [] #
                for map_ant_str in holog_obs_dict[holog_scan_id]['ant'].keys():
                    ref_ant_ids = np.array(_convert_ant_name_to_id(ant_names,list(holog_obs_dict[holog_scan_id]['ant'][map_ant_str])))
                    map_ant_id = _convert_ant_name_to_id(ant_names,map_ant_str)[0]

                    ref_ant_per_map_ant_list.append(ref_ant_ids)
                    map_ant_list.append(map_ant_id)
                    
                extract_holog_params["ref_ant_per_map_ant_tuple"] = tuple(ref_ant_per_map_ant_list)
                extract_holog_params["map_ant_tuple"] = tuple(map_ant_list)
                extract_holog_params["scans"] = scans
                extract_holog_params["sel_state_ids"] = state_ids
                extract_holog_params["holog_scan_id"] = holog_scan_id
                extract_holog_params["ant_names"] = ant_names
                
                if parallel:
                    delayed_list.append(
                        dask.delayed(_extract_holog_chunk)(
                            dask.delayed(extract_holog_params)
                        )
                    )
                else:
                    _extract_holog_chunk(extract_holog_params)

    spw_ctb.close()
    pol_ctb.close()

    if parallel:
        dask.compute(delayed_list)    

    extract_holog_params["holog_obs_dict"] = {}

    for id in ant_id:
        extract_holog_params["holog_obs_dict"][str(id)] = ant_names[id]

    holog_dict = _load_holog_file(holog_file=holog_name, dask_load=True, load_pnt_dict=False)

    _create_holog_meta_data(holog_file=holog_name, holog_dict=holog_dict, holog_params=extract_holog_params)


def _check_extract_holog_parms(    ms_name,
    holog_obs_dict,
    holog_name,
    point_name,
    data_col,
    parallel,
    overwrite):
    
    extract_holog_params = {}
    extract_holog_params["ms_name"] = ms_name
    extract_holog_params["holog_name"] = holog_name
    extract_holog_params["point_name"] = point_name
    extract_holog_params["parallel"] = parallel
    extract_holog_params["overwrite"] = overwrite

    
    #### Parameter Checking ####
    logger = _get_astrohack_logger()
    parms_passed = True
    
    parms_passed = parms_passed and _check_parms(extract_holog_params, 'ms_name', [str],default=None)

    base_name = _remove_suffix(ms_name,'.ms')
    parms_passed = parms_passed and _check_parms(extract_holog_params,'holog_name', [str],default=base_name+'.holog.zarr')
  
    
    point_base_name = _remove_suffix(holog_name,'.holog.zarr')
    parms_passed = parms_passed and _check_parms(extract_holog_params,'point_name', [str],default=point_base_name+'.point.zarr')
  
    #To Do: special function needed to check holog_obs_dict.
    parm_check = isinstance(holog_obs_dict,dict)
    parms_passed = parms_passed and parm_check
    if not parm_check:
        logger.error('Parameter holog_obs_dict must be of type '+ str(dict))
        
    parms_passed = parms_passed and _check_parms(extract_holog_params,'data_col', [str],default='DATA')

    parms_passed = parms_passed and _check_parms(extract_holog_params, 'parallel', [bool],default=False)

    parms_passed = parms_passed and _check_parms(extract_holog_params, 'overwrite', [bool],default=False)

    if not parms_passed:
        logger.error("extract_holog parameter checking failed.")
        raise Exception("extract_holog parameter checking failed.")
    
    
    return extract_holog_params
