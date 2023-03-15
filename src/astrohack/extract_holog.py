import os
import dask
import sys

import xarray as xr
import numpy as np

from casacore import tables as ctables

from prettytable import PrettyTable

from astrohack._utils import _system_message as console

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
#from memory_profiler import profile


fp=open('extract_holog.log','w+')
#@profile(stream=fp)
def extract_holog(
    ms_name,
    holog_name,
    holog_obs_dict,
    data_col="DATA",
    subscan_intent="MIXED",
    parallel=True,
    overwrite=False,
    sel_ddi=None
):
    """Extract holography data and create beam maps.
            subscan_intent: 'MIXED' or 'REFERENCE'

    Args:
        ms_name (string): Measurement file name
        holog_name (string): Basename of holog file to create.
        holog_obs_dict (dict): Nested dictionary ordered by ddi:{ scan: { map:[ant names], ref:[ant names] } } }
        data_col (str, optional): Data column from measurement set to acquire. Defaults to 'DATA'.
        subscan_intent (str, optional): Subscan intent, can be MIXED or REFERENCE; MIXED refers to a pointing measurement with half ON(OFF) source. Defaults to 'MIXED'.
        parallel (bool, optional): Boolean for whether to process in parallel. Defaults to True.
        overwrite (bool, optional): Boolean for whether to overwrite current holography file.
    """

    holog_file = "{base}.{suffix}".format(base=holog_name, suffix="holog.zarr")

    if os.path.exists(holog_file) is True and overwrite is False:
        console.error(
            "[_create_holog_file] holog file {file} exists. To overwite set the overwrite=True option in extract_holog or remove current file.".format(
                file=holog_file
            )
        )
        raise FileExistsError
    else:
        console.warning(
            "[extract_holog] Warning, current holography files will be overwritten."
        )

    pnt_name = "{base}.{pointing}".format(base=holog_name, pointing="point.zarr")

    try:
        pnt_dict = _load_pnt_dict(pnt_name)
    except:
        pnt_dict = _make_ant_pnt_dict(ms_name, pnt_name, parallel=parallel)
    

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

    # nomodify=True when using CASA tables.
    # print(os.path.join(ms_name,"DATA_DESCRIPTION"))
    console.info(
        "Opening measurement file {ms}".format(
            ms=os.path.join(ms_name, "DATA_DESCRIPTION")
        )
    )

    ctb = ctables.table(
        os.path.join(ms_name, "DATA_DESCRIPTION"),
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
        os.path.join(ms_name, "ANTENNA"),
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
    # MIXED : Pointing measurement, some antennas are on -ource, some off-source
    # REFERENCE : reference measurement (used for boresight in holography).
    # Undefined : ?

    ctb = ctables.table(
        os.path.join(ms_name, "STATE"),
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
        if (scan_intent in mode) and (subscan_intent in mode):
            state_ids.append(i)

    spw_ctb = ctables.table(
        os.path.join(ms_name, "SPECTRAL_WINDOW"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    pol_ctb = ctables.table(
        os.path.join(ms_name, "POLARIZATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    obs_ctb = ctables.table(
        os.path.join(ms_name, "OBSERVATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    
    telescope_name = obs_ctb.getcol("TELESCOPE_NAME")[0]

    extract_holog_params = {}

    delayed_list = []
    #for ddi in [holog_obs_dict['ddi'][0]]: #### NBNBNB: Chnage to all ddi's
    if sel_ddi is None:
        if holog_obs_dict['ddi'] is None:
            console.error("[extract_holog]: No DDI given found.")
            raise Exception()

        console.info("[extract_holog]: Processing DDI: {}".format(holog_obs_dict['ddi']))
        sel_ddi = holog_obs_dict['ddi']
    
    for ddi in sel_ddi:
        console.info("[extract_holog]: Processing select DDI: {}".format(ddi))
        spw_setup_id = ddi_spw[ddi]
        pol_setup_id = ddpol_indexol[ddi]

        extract_holog_params = {
            "ms_name": ms_name,
            "holog_name": holog_name,
            "pnt_name": pnt_name,
            "ddi": ddi,
            "data_col": data_col,
            "chan_setup": {},
            "pol_setup": {},
            "telescope_name": telescope_name,
            "overwrite": overwrite,
        }

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
                console.info("Processing ddi: {ddi}, scans: {scans}".format(ddi=ddi, scans=scans))
            
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

    holog_file = "{base}.{suffix}".format(base=extract_holog_params["holog_name"], suffix="holog.zarr")

    holog_dict = _load_holog_file(holog_file=holog_file, dask_load=True, load_pnt_dict=False)

    _create_holog_meta_data(holog_file=holog_file, holog_dict=holog_dict, holog_params=extract_holog_params)
