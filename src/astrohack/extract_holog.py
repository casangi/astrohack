import os
import dask
import json
import copy
import inspect 
import numpy as np
import numbers

from astropy.time import Time

from pprint import pformat

from casacore import tables as ctables

from astrohack._utils._tools import NumpyEncoder

from astrohack._utils._constants import pol_str
from astrohack._utils._conversion import _convert_ant_name_to_id
from astrohack._utils._extract_holog import _create_holog_meta_data
from astrohack._utils._extract_point import _extract_pointing
from astrohack._utils._dio import _load_point_file
from astrohack._utils._dio import _check_if_file_will_be_overwritten
from astrohack._utils._dio import _check_if_file_exists
from astrohack._utils._dio import _load_holog_file
from astrohack._utils._dio import _write_meta_data
from astrohack._utils._extract_holog import _extract_holog_chunk
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger
from astrohack._utils._param_utils._check_parms import _check_parms, _parm_check_passed
from astrohack._utils._tools import _remove_suffix

from astrohack._utils._extract_holog import _create_holog_obs_dict

from astrohack.mds import AstrohackHologFile
from astrohack.mds import AstrohackPointFile

from astrohack.extract_pointing import extract_pointing

CURRENT_FUNCTION=0

def extract_holog(
    ms_name,
    point_name,
    holog_name=None,
    holog_obs_dict=None,
    ddi='all',
    baseline_average_distance='all',
    baseline_average_nearest='all',
    data_column="CORRECTED_DATA",
    parallel=False,
    overwrite=False,
):
    """
    Extract holography and optionally pointing data, from measurement set. Creates holography output file.

    :param ms_name: Name of input measurement file name.
    :type ms_name: str

    :param point_name: Name of *<point_name>.point.zarr* file to use. This is must be provided.
    :type holog_name: str

    :param holog_name: Name of *<holog_name>.holog.zarr* file to create. Defaults to measurement set name with *holog.zarr* extension.
    :type holog_name: str, optional

    :param holog_obs_dict: The *holog_obs_dict* describes which scan and antenna data to extract from the measurement set. As detailed below, this compound dictionary also includes important meta data needed for preprocessing and extraction of the holography data from the measurement set. If not specified holog_obs_dict will be generated. For auto generation of the holog_obs_dict the assumtion is made that the same antanna beam is not mapped twice in a row (alternating sets of antennas is fine). If the holog_obs_dict is specified, the ddi input is ignored. The user can self generate this dictionary using `generate_holog_obs_dict`.
    :type holog_obs_dict: dict, optional

    :param ddi:  DDI(s) that should be extracted from the measurement set. Defaults to all DDI's in the ms.
    :type ddi: int numpy.ndarray | int list, optional

    :param baseline_average_distance: To increase the signal to noise for a mapping antenna mutiple reference antennas can be used. The baseline_average_distance is the acceptable distance (in meters) between a mapping antenna and a reference antenna. The baseline_average_distance is only used if the holog_obs_dict is not specified. If no distance is specified all reference antennas will be used. baseline_average_distance and baseline_average_nearest can not be used together.
    :type baseline_average_distance: float, optional

    :param baseline_average_nearest: To increase the signal to noise for a mapping antenna mutiple reference antennas can be used. The baseline_average_nearest is the number of nearest reference antennas to use. The baseline_average_nearest is only used if the holog_obs_dict is not specified.  baseline_average_distance and baseline_average_nearest can not be used together.
    :type baseline_average_nearest: int, optional

    :param data_column: Determines the data column to pull from the measurement set. Defaults to "CORRECTED_DATA".
    :type data_column: str, optional, ex. DATA, CORRECTED_DATA

    :param parallel: Boolean for whether to process in parallel, defaults to False.
    :type parallel: bool, optional

    :param overwrite: Boolean for whether to overwrite current holog.zarr and point.zarr files, defaults to False.
    :type overwrite: bool, optional

    :return: Holography holog object.
    :rtype: AstrohackHologFile

    .. _Description:

    **AstrohackHologFile**

    Holog object allows the user to access holog data via compound dictionary keys with values, in order of depth, `ddi` -> `map` -> `ant`. The holog object also provides a `summary()` helper function to list available keys for each file. An outline of the holog object structure is show below:

    .. parsed-literal::
        holog_mds = 
        {
            ddi_0:{
                map_0:{
                 ant_0: holog_ds,
                          ⋮
                 ant_n: holog_ds
                },
                ⋮
                map_p: …
            },
            ⋮
            ddi_m: …
        }

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

    """
    extract_holog_params = locals()

    logger = _get_astrohack_logger()

    function_name = inspect.stack()[CURRENT_FUNCTION].function

    ######### Parameter Checking #########
    extract_holog_params = _check_extract_holog_params(function_name=function_name, extract_holog_params=extract_holog_params)
    

    _check_if_file_exists(extract_holog_params['ms_name'])
    _check_if_file_will_be_overwritten(extract_holog_params['holog_name'], extract_holog_params['overwrite'])

    if holog_name==None:
        
        logger.debug('[{caller}]: File {file} does not exists. Extracting ...'.format(caller=function_name, file=holog_name))

        holog_name = _remove_suffix(ms_name, '.ms') + '.holog.zarr'
        extract_holog_params['holog_name'] = holog_name
            
        logger.debug('[{caller}]: Extracting holog to {output}'.format(caller=function_name, output=holog_name))
          
    try:
        pnt_dict = _load_point_file(extract_holog_params['point_name'])

    except Exception as error:
        logger.error('[{function_name}]: Error loading {name}. - {error}'.format(function_name=function_name, name=extract_holog_params["point_name"], error=error))
        
        return None

    ######## Get Spectral Windows ########
    ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "DATA_DESCRIPTION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ddi_spw = ctb.getcol("SPECTRAL_WINDOW_ID")
    ddpol_indexol = ctb.getcol("POLARIZATION_ID")
    ms_ddi = np.arange(len(ddi_spw))
    ctb.close()

    ######## Get Antenna IDs and Names ########
    ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "ANTENNA"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ant_names = np.array(ctb.getcol("NAME"))
    ant_id = np.arange(len(ant_names))
    ant_pos = ctb.getcol("POSITION")

    ctb.close()
    
    ######## Get Antenna IDs that are in the main table########
    ctb = ctables.table(
        extract_holog_params['ms_name'],
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ant1 = np.unique(ctb.getcol("ANTENNA1"))
    ant2 = np.unique(ctb.getcol("ANTENNA2"))
    ant_id_main = np.unique(np.append(ant1,ant2))
    
    ant_names_main = ant_names[ant_id_main]
    ctb.close()
    
    # Create holog_obs_dict or modify user supplied holog_obs_dict.
    ddi = extract_holog_params['ddi']

    # Create holog_obs_dict if not specified
    if holog_obs_dict is None:         
        holog_obs_dict = _create_holog_obs_dict(
            pnt_dict, 
            extract_holog_params['baseline_average_distance'],
            extract_holog_params['baseline_average_nearest'], 
            ant_names, 
            ant_pos,
            ant_names_main
        )
        
        #From the generated holog_obs_dict subselect user supplied ddis.
        if ddi != 'all':
            holog_obs_dict_keys = list(holog_obs_dict.keys())
            for ddi_key in holog_obs_dict_keys:
                if 'ddi' in ddi_key:
                    ddi_id = int(ddi_key.replace('ddi_',''))
                    if ddi_id not in ddi:
                        del holog_obs_dict[ddi_key]

            
    logger.info(f"[{function_name}]: holog_obs_dict: \n%s", pformat(list(holog_obs_dict.values())[0], indent=2, width=2))

    encoded_obj = json.dumps(holog_obs_dict, cls=NumpyEncoder)

    with open(".holog_obs_dict.json", "w") as outfile:
        json.dump(encoded_obj, outfile)
        
    ######## Get Scan and Subscan IDs ########
    # SDM Tables Short Description (https://drive.google.com/file/d/16a3g0GQxgcO7N_ZabfdtexQ8r2jRbYIS/view)
    # 2.54 ScanIntent (p. 150)
    # MAP ANTENNA SURFACE : Holography calibration scan

    # 2.61 SubscanIntent (p. 152)
    # MIXED : Pointing measurement, some antennas are on-source, some off-source
    # REFERENCE : reference measurement (used for boresight in holography).
    # Undefined : ?

    ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "STATE"),
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
        os.path.join(extract_holog_params['ms_name'], "SPECTRAL_WINDOW"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    pol_ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "POLARIZATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    obs_ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "OBSERVATION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    
    telescope_name = obs_ctb.getcol("TELESCOPE_NAME")[0]
    start_time_unix = obs_ctb.getcol('TIME_RANGE')[0][0] - 3506716800.0
    time = Time(start_time_unix, format='unix').jyear

    # If we have an EVLA run from before 2023 the pointing table needs to be fixed.
    if telescope_name == "EVLA" and time < 2023:

        # Convert from casa epoch to unix time
        his_ctb = ctables.table(
            os.path.join(extract_holog_params['ms_name'], "HISTORY"),
            readonly=True,
            lockoptions={"option": "usernoread"},
            ack=False,
        )
    
        if "pnt_tbl:fixed" not in his_ctb.getcol("MESSAGE"):
            logger.error("Pointing table not corrected, users should apply function astrohack.dio.fix_pointing_table() to remedy this.")
            
            return None
        
        his_ctb.close()

    count = 0
    delayed_list = []

    for ddi_name in holog_obs_dict.keys():
        ddi = int(ddi_name.replace('ddi_',''))
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

        # Loop over all beam_scan_ids, a beam_scan_id can conist out of more than one scan in an ms (this is the case for the VLA pointed mosiacs).
        for holog_map_key in holog_obs_dict[ddi_name].keys(): 

            if 'map' in holog_map_key:
                scans = holog_obs_dict[ddi_name][holog_map_key]["scans"]
                
                logger.info(f"[{function_name}]: Processing ddi: {ddi}, scans: {scans}")
                
                if len(list(holog_obs_dict[ddi_name][holog_map_key]['ant'].keys())) != 0:
                    map_ant_list = []
                    ref_ant_per_map_ant_list = []
                    
                    map_ant_name_list = []
                    ref_ant_per_map_ant_name_list = []
                    for map_ant_str in holog_obs_dict[ddi_name][holog_map_key]['ant'].keys():
                        ref_ant_ids = np.array(_convert_ant_name_to_id(ant_names, list(holog_obs_dict[ddi_name][holog_map_key]['ant'][map_ant_str])))
                        
                        map_ant_id = _convert_ant_name_to_id(ant_names,map_ant_str)[0]

                        ref_ant_per_map_ant_list.append(ref_ant_ids)
                        map_ant_list.append(map_ant_id)
                        
                        ref_ant_per_map_ant_name_list.append(list(holog_obs_dict[ddi_name][holog_map_key]['ant'][map_ant_str]))
                        map_ant_name_list.append(map_ant_str)

                    extract_holog_params["ref_ant_per_map_ant_tuple"] = tuple(ref_ant_per_map_ant_list)
                    extract_holog_params["map_ant_tuple"] = tuple(map_ant_list)
                    
                    extract_holog_params["ref_ant_per_map_ant_name_tuple"] = tuple(ref_ant_per_map_ant_name_list)
                    extract_holog_params["map_ant_name_tuple"] = tuple(map_ant_name_list)
                    
                    extract_holog_params["scans"] = scans
                    extract_holog_params["sel_state_ids"] = state_ids
                    extract_holog_params["holog_map_key"] = holog_map_key
                    extract_holog_params["ant_names"] = ant_names
                    
                    if parallel:
                        delayed_list.append(
                            dask.delayed(_extract_holog_chunk)(
                                dask.delayed(extract_holog_params)
                            )
                        )
                    else:
                        try:
                            _extract_holog_chunk(extract_holog_params)

                        except Exception as error:
                            print("[{function_name}]: There was an error, see log above for more info :: {error}".format(function_name=function_name, error=error))    

                    count += 1
                    
                else:
                    logger.warning(f'[{function_name}]: DDI ' + str(ddi) + ' has no holography data to extract.')

    spw_ctb.close()
    pol_ctb.close()
    obs_ctb.close()

    if parallel:
        try:
            dask.compute(delayed_list)    
    
        except Exception as error:
            print("[{function_name}]: There was an error, see log above for more info :: {error}".format(function_name=function_name, error=error))

    if count > 0:
        logger.info(f"[{function_name}]: Finished processing")

        holog_dict = _load_holog_file(holog_file=extract_holog_params["holog_name"], dask_load=True, load_pnt_dict=False)
        extract_holog_params['telescope_name'] = telescope_name
        
        holog_attr_file = "{name}/{ext}".format(name=extract_holog_params['holog_name'], ext=".holog_attr")

        meta_data = _create_holog_meta_data(
            holog_file=extract_holog_params['holog_name'], 
            holog_dict=holog_dict,
            input_params=extract_holog_params.copy()
        )

        _write_meta_data(holog_attr_file, meta_data)

        holog_mds = AstrohackHologFile(extract_holog_params['holog_name'])
        holog_mds._open()
        
        return holog_mds
    
    else:
        logger.warning(f"[{function_name}]: No data to process")
        return None

def _check_extract_holog_params(function_name, extract_holog_params):

    #### Parameter Checking ####
    logger = _get_astrohack_logger()
    parms_passed = True
    
    parms_passed = parms_passed and _check_parms(function_name, extract_holog_params, 'ms_name', [str], default=None)

    base_name = _remove_suffix(extract_holog_params['ms_name'], '.ms')
    
    parms_passed = parms_passed and _check_parms(function_name, extract_holog_params, 'holog_name', [str], default=base_name+'.holog.zarr')
    parms_passed = parms_passed and _check_parms(function_name, extract_holog_params, 'point_name', [str], default=None)

    point_base_name = _remove_suffix(extract_holog_params['holog_name'], '.holog.zarr')
    
    parms_passed = parms_passed and _check_parms(function_name, extract_holog_params, 'ddi', [list, int, str], list_acceptable_data_types=[int], default='all')
  
    #To Do: special function needed to check holog_obs_dict.
    parm_check = isinstance(extract_holog_params['holog_obs_dict'], dict) or (extract_holog_params['holog_obs_dict'] is None)
    parms_passed = parms_passed and parm_check

    if not parm_check:
        logger.error(f'[{function_name}]: Parameter holog_obs_dict must be of type {str(dict)}.')
        
    parms_passed = parms_passed and _check_parms(function_name, extract_holog_params, 'baseline_average_distance', [int, float, str], default='all')
    
    parms_passed = parms_passed and _check_parms(function_name, extract_holog_params, 'baseline_average_nearest', [int, str], default='all')
    
    if (extract_holog_params['baseline_average_distance'] != 'all') and (extract_holog_params['baseline_average_nearest'] != 'all'):
        logger.error(f'[{function_name}]: baseline_average_distance: {str(baseline_average_distance)} and 'f'baseline_average_nearest: {str(baseline_average_distance)} can not both be specified.')

        parms_passed = False
 
    parms_passed = parms_passed and _check_parms(function_name, extract_holog_params, 'data_column', [str], default='CORRECTED_DATA')

    parms_passed = parms_passed and _check_parms(function_name, extract_holog_params, 'parallel', [bool], default=False)

    parms_passed = parms_passed and _check_parms(function_name, extract_holog_params, 'overwrite', [bool],default=False)

    _parm_check_passed(function_name, parms_passed)

    return extract_holog_params

def generate_holog_obs_dict(
    ms_name,
    point_name,
    ddi='all',
    baseline_average_distance='all',
    baseline_average_nearest='all',
    parallel=False
):
    """
    Generate holography observation dictionary, from measurement set..

    :param ms_name: Name of input measurement file name.
    :type ms_name: str
    
    :param ddi:  DDI(s) that should be extracted from the measurement set. Defaults to all DDI's in the ms.
    :type ddi: int numpy.ndarray | int list, optional

    :param baseline_average_distance: To increase the signal to noise for a mapping antenna mutiple reference antennas can be used. The baseline_average_distance is the acceptable distance between a mapping antenna and a reference antenna. The baseline_average_distance is only used if the holog_obs_dict is not specified. If no distance is specified all reference antennas will be used. baseline_average_distance and baseline_average_nearest can not be used together.
    :type holog_obs_dict: float, optional

    :param baseline_average_nearest: To increase the signal to noise for a mapping antenna mutiple reference antennas can be used. The baseline_average_nearest is the number of nearest reference antennas to use. The baseline_average_nearest is only used if the holog_obs_dict is not specified.  baseline_average_distance and baseline_average_nearest can not be used together.
    :type holog_obs_dict: int, optional

    :param point_name: Name of *<point_name>.point.zarr* file to use. 
    :type point_name: str, optional

    :param parallel: Boolean for whether to process in parallel. Defaults to False
    :type parallel: bool, optional

    :return: holog observation dictionary
    :rtype: json

    .. _Description:

    **AstrohackHologFile**

    Holog object allows the user to access holog data via compound dictionary keys with values, in order of depth, `ddi` -> `map` -> `ant`. The holog object also provides a `summary()` helper function to list available keys for each file. An outline of the holog object structure is show below:

    .. parsed-literal::
        holog_mds = 
        {
            ddi_0:{
                map_0:{
                 ant_0: holog_ds,
                          ⋮
                 ant_n: holog_ds
                },
                ⋮
                map_p: …
            },
            ⋮
            ddi_m: …
        }

    **Additional Information**

        **holog_obs_dict[holog_mapping_id] (dict):**
        *holog_mapping_id* is a unique, arbitrary, user-defined integer assigned to the data that describes a single complete mapping of the beam.
        
        .. rubric:: This is needed for two reasons:
        * A complete mapping of the beam can be done over more than one scan (for example the VLA data). 
        * A measurement set can contain more than one mapping of the beam (for example the ALMA data).
    
        **holog_obs_dict[holog_mapping_id][scans] (int | numpy.ndarray | list):**
        All the scans in the measurement set the *holog_mapping_id*.
    
        **holog_obs_dict[holog_mapping_id][ant] (dict):**
        The dictionary keys are the mapping antenna names and the values a list of the reference antennas. See example below.

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

    """
    extract_holog_params = locals()

    logger = _get_astrohack_logger()
    
    function_name = inspect.stack()[CURRENT_FUNCTION].function
    
    _check_if_file_exists(ms_name)
    _check_if_file_exists(point_name)

    ######## Get Spectral Windows ########
    ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "DATA_DESCRIPTION"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )
    
    ddi_spw = ctb.getcol("SPECTRAL_WINDOW_ID")
    ddpol_indexol = ctb.getcol("POLARIZATION_ID")
    ms_ddi = np.arange(len(ddi_spw))
    
    ctb.close()

    ######## Get Antenna IDs and Names ########
    ctb = ctables.table(
        os.path.join(extract_holog_params['ms_name'], "ANTENNA"),
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ant_names = np.array(ctb.getcol("NAME"))
    ant_id = np.arange(len(ant_names))
    ant_pos = ctb.getcol("POSITION")

    ctb.close()
    
    ######## Get Antenna IDs that are in the main table########
    ctb = ctables.table(
        extract_holog_params['ms_name'],
        readonly=True,
        lockoptions={"option": "usernoread"},
        ack=False,
    )

    ant1 = np.unique(ctb.getcol("ANTENNA1"))
    ant2 = np.unique(ctb.getcol("ANTENNA2"))
    ant_id_main = np.unique(np.append(ant1,ant2))
    
    ant_names_main = ant_names[ant_id_main]
    ctb.close()
    
    # Create holog_obs_dict or modify user supplied holog_obs_dict.
    ddi = extract_holog_params['ddi']
    
    pnt_mds = AstrohackPointFile(extract_holog_params['point_name'])
    pnt_mds._open()

    holog_obs_dict = _create_holog_obs_dict(
            pnt_mds, 
            extract_holog_params['baseline_average_distance'],
            extract_holog_params['baseline_average_nearest'], 
            ant_names, 
            ant_pos,
            ant_names_main
    )
        
    #From the generated holog_obs_dict subselect user supplied ddis.
    if ddi != 'all':
        holog_obs_dict_keys = list(holog_obs_dict.keys())
        for ddi_key in holog_obs_dict_keys:
            if 'ddi' in ddi_key:
                ddi_id = int(ddi_key.replace('ddi_',''))
                if ddi_id not in ddi:
                    del holog_obs_dict[ddi_key]

    encoded_obj = json.dumps(holog_obs_dict, cls=NumpyEncoder)

    with open(".holog_obs_dict.json", "w") as outfile:
        json.dump(encoded_obj, outfile)

    return json.loads(encoded_obj)