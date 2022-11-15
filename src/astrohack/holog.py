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

from casacore import tables as ctables

def _get_nearest_index(value:float, array: np.ndarray):
    """ Simple utility function to find the index of the nearest entry to value.

    Args:
        value (float): _description_
        array (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    
    return np.abs(array - value).argmin()

def extract_pointing_chunk(map_ant_ids, time_vis, pnt_ant_dict):
    """ Extract nearest MAIN table time indexed pointing map

    Args:
        map_ant_ids (dict): list of antenna ids
        time_vis (numpy.ndarray): sorted, unique list of visibility times
        pnt_ant_dict (dict): map of pointing directional cosines with a map key based on the antenna id and indexed by the MAIN table visibility time. 

    Returns:
        _type_: _description_
    """
    # pnt_map_dict = extract_pointing_chunk(map_ant_ids, time_vis, pnt_ant_dict)
    # >> map_ant_ids: list of antenna ids
    # >> time_vis: sorted, unique list of visibility times
    # >> pnt_ant_dict: pointing directions mapped on antenna id [ant_id]->xarray.DIRECTIONAL_COSINES(time, direction) for instance

    n_time_vis = time_vis.shape[0]

    pnt_map_dict = {}

    for antenna in map_ant_ids:
        print(antenna)
        pnt_map_dict[antenna] = np.zeros((n_time_vis, 2)) 
        for time_index, time in enumerate(time_vis):
        
            # find nearest pnt_ant_dict time value and add to dictionary
            index = _get_nearest_index(time, pnt_ant_dict[antenna].coords['time'].data)

            # l-value of directional cosines
            pnt_map_dict[antenna][time_index] = pnt_ant_dict[antenna].DIRECTIONAL_COSINES[index][0]  

            # m-value of directional cosines
            pnt_map_dict[antenna][time_index] = pnt_ant_dict[antenna].DIRECTIONAL_COSINES[index][1]

    return pnt_map_dict

@njit(cache=False)
def extract_holog_chunk_jit(vis_data, weight, ant1, ant2, time_vis_row, time_vis, flag, flag_row, map_ant_ids, ref_ant_ids):
    """ JIT copiled function to extract relevant visibilty data from chunk after flagging and applying weights.

    Args:
        vis_data (numpy.ndarray): Visibility data (row, channel, polarization)
        weight (numpy.ndarray): Data weight values (row, polarization)
        ant1 (numpy.ndarray): List of antenna_ids for antenna1
        ant2 (numpy.ndarray): List of antenna_ids for antenna2
        time_vis_row (numpy.ndarray): Array of full time talues by row
        time_vis (numpy.ndarray): Array of selected time values
        flag (numpy.ndarray): Array of data quality flags to apply to data
        flag_row (numpy.ndarray): Array indicating when a full row of data should be flagged/
        map_ant_ids (numpy.ndarray): Array of antenna_ids for mapping data
        ref_ant_ids (numpy.ndarray): Array of antenna_ids for reference data

    Returns:
        dict: Antenna_id referenced (key) dictionary containing the visibility data selected by (time, channel, polarization)
    """

    '''
    1. Should we do this in double precision?
    2. ~Add flag_row and flags~
    3. ~Do weighted sum of data~
    4. Channel averaging
    5. ? Calculate a time_vis as an average from time_vis_centroid
    '''

    n_row, n_chan, n_pol = vis_data.shape
    n_time = len(time_vis)
    
    vis_map_dict = {}
    #sum_map_dict = {}
    #weight_map_dict = {}
    sum_weight_map_dict = {}

    for antenna_id in map_ant_ids:
        vis_map_dict[antenna_id] = np.zeros((n_time, n_chan, n_pol), dtype=types.complex64)
        #weight_map_dict[antenna_id] = np.zeros((n_time, n_chan, n_pol), dtype=types.complex64)
        #sum_map_dict[antenna_id] = np.zeros((n_time, n_chan, n_pol), dtype=types.complex64)
        sum_weight_map_dict[antenna_id] = np.zeros((n_time, n_chan, n_pol), dtype=types.complex64)

        
    #Create sum of weight dict
    
    print(vis_data.dtype)
    
    for row in range(n_row):

        if flag_row is False:
            continue

        ant1_index = ant1[row]
        ant2_index = ant2[row]
        
        if (ant1_index in map_ant_ids) and (ant2_index in ref_ant_ids):
            vis_baseline = vis_data[row, :, :] # n_chan x n_pol
            mapping_ant_index = ant1_index # mapping antenna index

        elif (ant2_index in map_ant_ids) and (ant1_index not in ref_ant_ids): #conjugate
            vis_baseline = np.conjugate(vis_data[row, :, :])
            mapping_ant_index = ant2_index

        else:
            continue
            
        #Need to do weights and flags
        time_index = np.searchsorted(time_vis, time_vis_row[row])
        
        #Should we unroll this assignment for numba?
        #vis_map_dict[mapping_ant_index][time_index, :, :] = vis_map_dict[mapping_ant_index][time_index, :, :] + vis_baseline

        for chan in range(n_chan):
            for pol in range(n_pol):
                if ~(flag[row, chan, pol]):
                    # Calculate running weighted sum of visibilities
                    vis_map_dict[mapping_ant_index][time_index, chan, pol] = vis_map_dict[mapping_ant_index][time_index, chan, pol] + vis_baseline[chan, pol]*weight[row, pol]

                    # Build map dictionary of weights for later use
                    #weight_map_dict[mapping_ant_index][time_index, chan, pol] = weight[row, pol]

                    # Calculate running sum of weights
                    sum_weight_map_dict[mapping_ant_index][time_index, chan, pol] = sum_weight_map_dict[mapping_ant_index][time_index, chan, pol] + weight[row, pol]
                                
                    # Calculate running weighted average
                    #vis_map_dict[mapping_ant_index][time_index, chan, pol] = sum_map_dict[mapping_ant_index][time_index, chan, pol]/total_weight_map_dict[mapping_ant_index][time_index, chan, pol]

    

    for mapping_antenna_index in vis_map_dict.keys():
        print(mapping_antenna_index)
        for time_index in range(n_time):
            for chan in range(n_chan):
                for pol in range(n_pol):
                    if sum_weight_map_dict[mapping_antenna_index][time_index, chan, pol] == 0:
                        vis_map_dict[mapping_antenna_index][time_index, chan, pol] = 0.
                    else:
                        vis_map_dict[mapping_antenna_index][time_index, chan, pol] = vis_map_dict[mapping_antenna_index][time_index, chan, pol]/sum_weight_map_dict[mapping_antenna_index][time_index, chan, pol]

    return vis_map_dict, sum_weight_map_dict            
            
def holog_chunk(ms_name, data_col, ddi, scan, map_ant_ids, ref_ant_ids, sel_state_ids):
    """ Perform data query on holography data chunk and get unique time and state_ids/

    Args:
        ms_name (str): Measurementset name
        data_col (str): Data column to extract.
        ddi (int): Data description id
        scan (int): Scan number
        map_ant_ids (numpy.narray): Array of antenna_id values corresponding to mapping data.
        ref_ant_ids (numpy.narray): Arry of antenna_id values corresponding to reference data.
        sel_state_ids (list): List pf state_ids corresponding to holography data/
    """
    
    start = time.time()
    ctb = ctables.taql('select %s, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG, STATE_ID from %s WHERE DATA_DESC_ID == %s AND SCAN_NUMBER == %s AND STATE_ID in %s' % (data_col, ms_name, ddi, scan, sel_state_ids))
    
    vis_data = ctb.getcol('DATA')
    weight = ctb.getcol('WEIGHT')
    ant1 = ctb.getcol('ANTENNA1')
    ant2 = ctb.getcol('ANTENNA2')
    time_vis_row = ctb.getcol('TIME')
    time_vis_row_centroid = ctb.getcol('TIME_CENTROID')
    flag = ctb.getcol('FLAG')
    flag_row = ctb.getcol('FLAG_ROW')
    state_ids_row = ctb.getcol('STATE_ID')

    #n_end = int(1599066/8) #/8
    #vis_data = ctb.getcol('DATA',0,n_end)
    #weight = ctb.getcol('WEIGHT',0,n_end)
    #ant1 = ctb.getcol('ANTENNA1',0,n_end)
    #ant2 = ctb.getcol('ANTENNA2',0,n_end)
    #time_vis_row = ctb.getcol('TIME',0,n_end)
    #time_vis_centroid_row = ctb.getcol('TIME_CENTROID',0,n_end)
    #flag = ctb.getcol('FLAG',0,n_end)
    #flag_row = ctb.getcol('FLAG_ROW',0,n_end)
    #state_ids_row = ctb.getcol('STATE_ID',0,n_end)
    
    ctb.close()

    print(time.time()-start)
    #print(vis_data.shape, weight.shape, ant1.shape, ant2.shape, time_vis_row.shape, time_vis_centroid_row.shape, flag.shape, flag_row.shape)
    
    start = time.time()
    time_vis, unique_index = np.unique(time_vis_row, return_index=True) # Note that values are sorted.
    state_ids = state_ids_row[unique_index]
    
    print('Time to unique ',time.time()-start)

    start = time.time()

    # vis_map_dict: [mapping_antenna](time_index, chan, polarization)
    # weight_map_dict: [mapping_antenna](time_index, chan, polarization)
    vis_map_dict, weight_map_dict = extract_holog_chunk_jit(vis_data, weight, ant1, ant2, time_vis_row, time_vis, flag, flag_row, map_ant_ids, ref_ant_ids)


    
    print('Time to jit ',time.time()-start)

    #pnt_ant_dict = load_pnt_dict(...)
    pnt_ant_dict = load_pnt_dict('.'.join((ms_name, 'pnt.dict')))
    
    # pnt_map_dict = extract_pointing_chunk(map_ant_ids, time_vis, pnt_ant_dict)
    # >> map_ant_ids: list of antenna ids
    # >> time_vis: sorted, unique list of visibility times: MAIN table values
    # >> pnt_ant_dict: pointing directions mapped on antenna id [ant_id]->xarray.DIRECTIONAL_COSINES(time, direction) for instance : POINTING tables values

    pnt_map_dict = extract_pointing_chunk(map_ant_ids, time_vis, pnt_ant_dict)

    return pnt_map_dict

    #grid all subscans onto a single grid
    #bm_map_dict = create_beam_maps(vis_map_dict, pnt_map_dict, map_ant_ids, state_ids, time_vis) # each mapping antenna has an image cube of dims: n_state_ids (time) x nchan x pol x l x m, n_state_ids = len(np.unique(state_ids))
 
    '''
    start = time.time()
    #ctb = ctables.table(ms_name, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    ctb = ctables.taql('select DATA, ANTENNA1, ANTENNA2, TIME from %s WHERE DATA_DESC_ID == %s AND SCAN_NUMBER == %s' % (ms_name,ddi,scan))
    #data = ctb.getcol('DATA',0,1000000)
    data = ctb.getcol('DATA')
    print('getcol',time.time()-start, data.shape)
   
    start = time.time()
    
    for i_s in state_ids:
        for i_m in map_ant_ids:
            for i_r in ref_ant_ids:
                if i_m < i_r:
                    ctb = ctables.taql('select DATA, ANTENNA1, ANTENNA2, TIME from %s WHERE DATA_DESC_ID == %s AND SCAN_NUMBER == %s AND ANTENNA1==%s AND ANTENNA2==%s AND STATE_ID==%s' % (ms_name,ddi,scan,i_m,i_r))
                else:
                    ctb = ctables.taql('select DATA, ANTENNA1, ANTENNA2, TIME from %s WHERE DATA_DESC_ID == %s AND SCAN_NUMBER == %s AND ANTENNA1==%s AND ANTENNA2==%s AND STATE_ID==%s' % (ms_name,ddi,scan,i_r,i_m,i_s))
                data = ctb.getcol('DATA')
                print(data.shape)
    
    print('getcol',time.time()-start, data.shape)

    
    ctb.close()
    #2022-10-31 21:53:56 INFO listobs                  20:40:51.9 - 20:52:32.8     2      0 J1924-2914             3230766  [0]  [0.096] [CALIBRATE_WVR#MIXED,CALIBRATE_WVR#REFERENCE,MAP_ANTENNA_SURFACE#MIXED,MAP_ANTENNA_SURFACE#REFERENCE]
    

    ### NB: Add check if directions refrence frame is Azemuth Elevation (AZELGEO)
    
    direction = np.swapaxes(pt_ant_table.getcol('DIRECTION')[:,0,:],0,1)
    target = np.swapaxes(pt_ant_table.getcol('TARGET')[:,0,:],0,1)
    encoder = np.swapaxes(pt_ant_table.getcol('ENCODER'),0,1)
    direction_time = pt_ant_table.getcol('TIME')
    pointing_offset = np.swapaxes(pt_ant_table.getcol('POINTING_OFFSET')[:,0,:],0,1)
    '''
   



def holog(ms_name, holog_obs_dict, data_col='DATA', subscan_intent='MIXED', parallel=True):
    """subscan_intent: 'MIXED' or 'REFERENCE'

    Args:
        ms_name (string): measurement file name
        holog_obs_dict (dict): nested dictionary ordered by ddi:{ scan: { map:[ant names], ref:[ant names] } } }
        data_col (str, optional): data column from measurement set to acquire. Defaults to 'DATA'.
        subscan_intent (str, optional): subscan intent, can be MIXED or REFERENCE; MIXED refers to a pointing measurement with half ON(OFF) source. Defaults to 'MIXED'.
        parallel (bool, optional): bool for whether to process in parallel. Defaults to True.
    """
    
    ######## Get Spectral Windows ########
    
    # nomodify=True when using CASA tables.
    print(os.path.join(ms_name,"DATA_DESCRIPTION"))

    ctb = ctables.table(os.path.join(ms_name,"DATA_DESCRIPTION"),readonly=True, lockoptions={'option': 'usernoread'}, ack=False) 
    spectral_window_id = ctb.getcol("SPECTRAL_WINDOW_ID")
    ddi = np.arange(len(spectral_window_id))
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

    '''
    for ddi in holog_obs_dict:
        for scan in holog_obs_dict[ddi].keys(): 
            print('Processing ddi: {ddi}, scan: {scan}'.format(ddi=ddi, scan=scan))

            # Determine antenna ids associated with mapping (reference) in list
            map_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]['map']))[0]
            ref_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]['ref']))[0]
            
            holog_chunk(ms_name, data_col, ddi, scan, map_ant_ids, ref_ant_ids, state_ids)

    '''
    if parallel:
        delayed_list = []
        for ddi in holog_obs_dict:
            for scan in holog_obs_dict[ddi].keys(): 
                print('Processing ddi: {ddi}, scan: {scan}'.format(ddi=ddi, scan=scan))

                map_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]['map']))[0]
                ref_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]['ref']))[0]

                delayed_list.append(
                    dask.delayed(holog_chunk)(
                        dask.delayed(ms_name ), 
                        dask.delayed(data_col), 
                        dask.delayed(ddi),
                        dask.delayed(scan),
                        dask.delayed(map_ant_ids),
                        dask.delayed(ref_ant_ids),
                        dask.delayed(state_ids)
                    )
                )
        
        dask.compute(delayed_list)
    else:
        for ddi in holog_obs_dict:
            for scan in holog_obs_dict[ddi].keys(): 
                print('Processing ddi: {ddi}, scan: {scan}'.format(ddi=ddi, scan=scan))

                map_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]['map']))[0]
                ref_ant_ids = np.nonzero(np.in1d(ant_name, holog_obs_dict[ddi][scan]['ref']))[0]        
                                
                return holog_chunk(ms_name, data_col, ddi, scan, map_ant_ids, ref_ant_ids, state_ids)
    '''
    if parallel:
        delayed_list = []
        #for i_d in ddi:
        #    delayed_pnt_list.append(dask.delayed(make_ant_pnt_xds)(dask.delayed(ms_name ),dask.delayed(i_a),dask.delayed(pnt_name)))
        #dask.compute(delayed_pnt_list)
    else:
        for i_d in ddi:
            for i_s in scan_id:
                extract_holog(ms_name,data_col,i_d)
    '''
    #return load_pnt_dict(pnt_name)






    #(4, 64, 3230766)
    #        3230766
    '''
    start = time.time()
    tb = table()
    tb.open(ms_name, nomodify=True, lockoptions={'option': 'usernoread'})
    #main_table = tb.taql('select DATA, ANTENNA1, ANTENNA2, TIME from %s WHERE DATA_DESC_ID == %s AND SCAN_NUMBER == %s' % (ms_name,ddi,scan))
    
    print('TAQL',time.time()-start)
    
    
    start = time.time()
    #data = main_table.getcol('ANTENNA1')
    #data = main_table.getcol('DATA',startrow=0,nrow=int(3230766/4))
    data = tb.getcol('DATA',startrow=0,nrow=1000000)
    print('getcol 1',time.time()-start, data.shape)
    
    print(data.shape)
    tb.close()
    
    ####################
    
    start = time.time()
    ctb = ctables.table(ms_name, readonly=True, lockoptions={'option': 'usernoread'}, ack=False)
    data = ctb.getcol('DATA',0,1000000)
    print('getcol 0',time.time()-start, data.shape)

    print('*******'*20)
    '''
