import os
import dask
import sys

import xarray as xr
import numpy as np

from casacore import tables as ctables

from astrohack._utils._constants import pol_str

from astrohack._utils._conversion import _convert_ant_name_to_id

from astrohack._utils._holog import _create_holog_meta_data

from astrohack._utils._extract_point import _extract_pointing

from astrohack._utils._io import _load_point_file
from astrohack._utils._io import _open_no_dask_zarr
from astrohack._utils._io import _read_data_from_holog_json
from astrohack._utils._io import _read_meta_data
from astrohack._utils._io import _load_holog_file
from astrohack._utils._io import  check_if_file_will_be_overwritten,check_if_file_exists

from astrohack._utils._extract_holog import _extract_holog_chunk

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._parm_utils._check_parms import _check_parms
from astrohack._utils._utils import _remove_suffix
from astrohack._utils._io import _load_holog_file


from astrohack._utils._dio import AstrohackHologFile
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
    Extract holography and optionally pointing data, from measurement set. Creates holography output file.

    :param ms_name: Name of input measurement file name.
    :type ms_name: str

    :param holog_obs_dict: The *holog_obs_dict* describes which scan and antenna data to extract from the measurement set. As detailed below, this compound dictionary also includes important meta data needed for preprocessing and extraction of the holography data from the measurement set.
    :type holog_obs_dict: dict        

    :param holog_name: Name of *<holog_name>.holog.zarr* file to create. Defaults to measurement set name with *holog.zarr* extension.
    :type holog_name: str, optional

    :param point_name: Name of *<point_name>.point.zarr* file to create. Defaults to measurement set name with *point.zarr* extension.
    :type point_name: str, optional

    :param data_col: Determines the data column to pull from the measurement set. Defaults to "DATA"
    :type data_col: str, optional

    :param parallel: Boolean for whether to process in parallel. Defaults to False
    :type parallel: bool, optional

    :param overwrite: Boolean for whether to overwrite current holog.zarr and point.zarr files., defaults to False
    :type overwrite: bool, optional

    .. _Description:

    **Additional Information**

        This function extracts the holography related information from the given measurement file. The data is restructured into an astrohack file format and saved into a file in the form of *<holog_name>.holog.zarr*. The extension *.holog.zarr* is used for all holography files. In addition, the pointing information is recorded into a holography file of format *<pointing_name>.point.zarr*. The extension *.point.zarr* is used for all holography pointing files. 

        **holog_obs_dict[holog_mapping_id] (dict):**
        *holog_mapping_id* is a unique, arbitrary, user-defined integer assigned to the data that describes a single complete mapping of the beam.
        
        .. rubric:: This is needed for two reasons:
        * A complete mapping of the beam can be done over more than one scan (for example the VLA data). 
        * A measurement set can contain more than one mapping of the beam (for example the ALMA data).
    
        **holog_obs_dict[holog_mapping_id][scans] (int | numpy.ndarray | list):**
        All the scans in the measurement set the *holog_mapping_id*.
    
        **holog_obs_dict[holog_mapping_id][ant] (dict):**
        The dictionary keys are the mapping antenna names and the values a list of the reference antennas. See example below.
    
        **holog_obs_dict[ddi] (int | numpy.ndarray | list):**
        Value(s) of DDI that should be extracted from the measurement set.

        The below example shows how the *holog_obs_description* dictionary should be laid out. For each *holog_mapping_id* the relevant scans 
        and antennas must be provided. For the `ant` key, an entry is required for each mapping antenna and the accompanying reference antenna(s).
    
        .. parsed-literal::
            holog_obs_description = {
                'map_0' :{
                    'scans':[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                    'ant':{
                        'DA44':[
                            'DV02', 'DV03', 'DV04', 
                            'DV11', 'DV12', 'DV13', 
                            'DV14', 'DV15', 'DV16', 
                            'DV17', 'DV18', 'DV19', 
                            'DV20', 'DV21', 'DV22', 
                            'DV23', 'DV24', 'DV25'
                        ]
                    }
                }
            }
            holog_obs_description['ddi'] = [0]

    """
    logger = _get_astrohack_logger()
    
    
    ######### Parameter Checking #########
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
    #######################################
        
    ############# Exstract pointing infromation and save to point.zarr #############
#    try:
#        pnt_dict = _load_point_file(extract_holog_parms['point_name'])
#    except:
#        pnt_dict = _make_ant_pnt_dict(extract_holog_parms['ms_name'], extract_holog_parms['point_name'], parallel=extract_holog_parms['parallel'])
    
    pnt_dict = _extract_pointing(extract_holog_parms['ms_name'], extract_holog_parms['point_name'], parallel=extract_holog_parms['parallel'])

    
    
    #### To DO: automatically create holog_obs_dict
    # from astrohack._utils._extract_holog import _create_holog_obs_dict


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

    ## DDI selection
    if holog_obs_dict['ddi'] is None:
        logger.error("No DDIs in holog_obs_dict.")
        raise Exception()

    delayed_list = []
    
    for ddi in holog_obs_dict['ddi']:
        spw_setup_id = ddi_spw[ddi]
        pol_setup_id = ddpol_indexol[ddi]
        
        extract_holog_parms["ddi"] = ddi
        extract_holog_parms["chan_setup"] = {}
        extract_holog_parms["pol_setup"] = {}
        
        extract_holog_parms["chan_setup"]["chan_freq"] = spw_ctb.getcol("CHAN_FREQ", startrow=spw_setup_id, nrow=1)[0, :]
        extract_holog_parms["chan_setup"]["chan_width"] = spw_ctb.getcol("CHAN_WIDTH", startrow=spw_setup_id, nrow=1)[0, :]
        extract_holog_parms["chan_setup"]["eff_bw"] = spw_ctb.getcol("EFFECTIVE_BW", startrow=spw_setup_id, nrow=1)[0, :]
        extract_holog_parms["chan_setup"]["ref_freq"] = spw_ctb.getcol("REF_FREQUENCY", startrow=spw_setup_id, nrow=1)[0]
        extract_holog_parms["chan_setup"]["total_bw"] = spw_ctb.getcol("TOTAL_BANDWIDTH", startrow=spw_setup_id, nrow=1)[0]

        extract_holog_parms["pol_setup"]["pol"] = pol_str[pol_ctb.getcol("CORR_TYPE", startrow=pol_setup_id, nrow=1)[0, :]]
                
        
        extract_holog_parms["telescope_name"] = obs_ctb.getcol("TELESCOPE_NAME")[0]
        

        for holog_map_key in holog_obs_dict.keys(): #loop over all beam_scan_ids, a beam_scan_id can conist out of more than one scan in an ms (this is the case for the VLA pointed mosiacs).

            if 'map' in holog_map_key:
                scans = holog_obs_dict[holog_map_key]["scans"]
                logger.info("Processing ddi: {ddi}, scans: {scans}".format(ddi=ddi, scans=scans))
            
                map_ant_list = []
                ref_ant_per_map_ant_list = [] #
                
                map_ant_name_list = []
                ref_ant_per_map_ant_name_list = [] #
                for map_ant_str in holog_obs_dict[holog_map_key]['ant'].keys():
                    ref_ant_ids = np.array(_convert_ant_name_to_id(ant_names,list(holog_obs_dict[holog_map_key]['ant'][map_ant_str])))
                    map_ant_id = _convert_ant_name_to_id(ant_names,map_ant_str)[0]

                    ref_ant_per_map_ant_list.append(ref_ant_ids)
                    map_ant_list.append(map_ant_id)
                    
                    ref_ant_per_map_ant_name_list.append(list(holog_obs_dict[holog_map_key]['ant'][map_ant_str]))
                    map_ant_name_list.append(map_ant_str)

                extract_holog_parms["ref_ant_per_map_ant_tuple"] = tuple(ref_ant_per_map_ant_list)
                extract_holog_parms["map_ant_tuple"] = tuple(map_ant_list)
                
                extract_holog_parms["ref_ant_per_map_ant_name_tuple"] = tuple(ref_ant_per_map_ant_name_list)
                extract_holog_parms["map_ant_name_tuple"] = tuple(map_ant_name_list)
                
                extract_holog_parms["scans"] = scans
                extract_holog_parms["sel_state_ids"] = state_ids
                extract_holog_parms["holog_map_key"] = holog_map_key
                extract_holog_parms["ant_names"] = ant_names
                
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

    extract_holog_parms["holog_obs_dict"] = {}

    for id in ant_id:
        extract_holog_parms["holog_obs_dict"][str(id)] = ant_names[id]

    holog_dict = _load_holog_file(holog_file=extract_holog_parms["holog_name"], dask_load=True, load_pnt_dict=False)

    _create_holog_meta_data(holog_file=extract_holog_parms['holog_name'], holog_dict=holog_dict, holog_params=extract_holog_parms)
    
    holog_mds = AstrohackHologFile(extract_holog_parms['holog_name'])
    holog_mds.open()
    
    return holog_mds
    


def _check_extract_holog_parms(    ms_name,
    holog_obs_dict,
    holog_name,
    point_name,
    data_col,
    parallel,
    overwrite):
    
    extract_holog_parms = {}
    extract_holog_parms["ms_name"] = ms_name
    extract_holog_parms["holog_name"] = holog_name
    extract_holog_parms["point_name"] = point_name
    extract_holog_parms["data_col"] = data_col
    extract_holog_parms["parallel"] = parallel
    extract_holog_parms["overwrite"] = overwrite

    
    #### Parameter Checking ####
    logger = _get_astrohack_logger()
    parms_passed = True
    
    parms_passed = parms_passed and _check_parms(extract_holog_parms, 'ms_name', [str],default=None)

    base_name = _remove_suffix(ms_name,'.ms')
    parms_passed = parms_passed and _check_parms(extract_holog_parms,'holog_name', [str],default=base_name+'.holog.zarr')
  
    
    point_base_name = _remove_suffix(extract_holog_parms['holog_name'],'.holog.zarr')
    parms_passed = parms_passed and _check_parms(extract_holog_parms,'point_name', [str],default=point_base_name+'.point.zarr')
  
    #To Do: special function needed to check holog_obs_dict.
    parm_check = isinstance(holog_obs_dict,dict)
    parms_passed = parms_passed and parm_check
    if not parm_check:
        logger.error('Parameter holog_obs_dict must be of type '+ str(dict))
        
    parms_passed = parms_passed and _check_parms(extract_holog_parms,'data_col', [str],default='DATA')

    parms_passed = parms_passed and _check_parms(extract_holog_parms, 'parallel', [bool],default=False)

    parms_passed = parms_passed and _check_parms(extract_holog_parms, 'overwrite', [bool],default=False)

    if not parms_passed:
        logger.error("extract_holog parameter checking failed.")
        raise Exception("extract_holog parameter checking failed.")
    
    
    return extract_holog_parms
