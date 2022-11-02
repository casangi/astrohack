import numpy as np
import xarray as xr
import dask.array as da
import dask
import time
#from casatools import table
from casacore import tables as ctables
import os
from numba import njit
from numba.core import types
from numba.typed import Dict

@njit(cache=False)
def extract_holog_chunk_jit(vis_data, weight, ant1, ant2, time_vis_row, time_vis, flag, flag_row, map_ant_ids,ref_ant_ids):
    '''
    1. Should we do this in double precision?
    2. Add flag_row and flags
    3. Do weighted sum of data
    4. Channel averaging
    5. ? Calculate a time_vis as an average from time_vis_centroid
    '''
    n_row,n_chan,n_pol = vis_data.shape
    n_time = len(time_vis)
    print('jit function',n_row,n_chan,n_pol)

    vis_map_dict = {map_ant_ids[0]:np.zeros((n_time,n_chan,n_pol),dtype=types.complex64)}
    for i_m in map_ant_ids[1:]:
        vis_map_dict[i_m] = np.zeros((n_time,n_chan,n_pol),dtype=types.complex64)
        
    #Create sum of weight dict
    
    print(vis_data.dtype)
    
    for i_r in range(n_row):
        i_a1 = ant1[i_r]
        i_a2 = ant2[i_r]
        
        if (i_a1 in map_ant_ids) and (i_a2 in ref_ant_ids):
            vis_bl = vis_data[i_r,:,:] # n_chan x n_pol
            i_m = i_a1 #mapping antenna index
        elif (i_a2 in map_ant_ids) and (i_a1 not in ref_ant_ids): #conjugate
            vis_bl = np.conjugate(vis_data[i_r,:,:])
            i_m = i_a2
        else:
            continue
            
        #Need to do weights and flags
        i_t = np.searchsorted(time_vis,time_vis_row[i_r])
        
        #Should we unroll this assignment for numba?
        vis_map_dict[i_m][i_t,:,:] = vis_map_dict[i_m][i_t,:,:] + vis_bl

    return vis_map_dict
        
            
        
            
            
        

def holog_chunk(ms_name,data_col,ddi,scan,map_ant_ids,ref_ant_ids,sel_state_ids):
    
    start = time.time()
    ctb = ctables.taql('select %s, ANTENNA1, ANTENNA2, TIME, TIME_CENTROID, WEIGHT, FLAG_ROW, FLAG, STATE_ID from %s WHERE DATA_DESC_ID == %s AND SCAN_NUMBER == %s AND STATE_ID in %s' % (data_col,ms_name,ddi,scan,sel_state_ids))
    #    vis_data = ctb.getcol('DATA')
    #    weight = ctb.getcol('WEIGHT')
    #    ant1 = ctb.getcol('ANTENNA1')
    #    ant2 = ctb.getcol('ANTENNA2')
    #    time = ctb.getcol('TIME')
    #    time_vis_row_centroid = ctb.getcol('TIME_CENTROID')
    #    flag = ctb.getcol('FLAG')
    #    flag_row = ctb.getcol('FLAG_ROW')
    n_end = int(1599066/8) #/8
    vis_data = ctb.getcol('DATA',0,n_end)
    weight = ctb.getcol('WEIGHT',0,n_end)
    ant1 = ctb.getcol('ANTENNA1',0,n_end)
    ant2 = ctb.getcol('ANTENNA2',0,n_end)
    time_vis_row = ctb.getcol('TIME',0,n_end)
    time_vis_centroid_row = ctb.getcol('TIME_CENTROID',0,n_end)
    flag = ctb.getcol('FLAG',0,n_end)
    flag_row = ctb.getcol('FLAG_ROW',0,n_end)
    state_ids_row = ctb.getcol('STATE_ID',0,n_end)
    ctb.close()
    print(time.time()-start)
    print(vis_data.shape, weight.shape, ant1.shape, ant2.shape, time_vis_row.shape, time_vis_centroid_row.shape, flag.shape, flag_row.shape)
    
    start = time.time()
    time_vis, unique_indx = np.unique(time_vis_row,return_index=True) #Note that values are sorted.
    state_ids = state_ids_row[unique_indx]
    
    print('Time to unique ',time.time()-start)

    start = time.time()
    vis_map_dict = extract_holog_chunk_jit(vis_data, weight, ant1, ant2, time_vis_row, time_vis, flag, flag_row, map_ant_ids,ref_ant_ids)
    print('Time to jit ',time.time()-start)
    
    #pnt_ant_dict = load(...)
    #pnt_map_dict = extract_pointing_chunk(map_ant_ids,time_vis,pnt_ant_dict)
    #grid all subscans onto a single grid
    #bm_map_dict = create_beam_maps(vis_map_dict,pnt_map_dict,map_ant_ids,state_ids,time_vis) # each mapping antenna has an image cube of dims: n_state_ids (time) x nchan x pol x l x m, n_state_ids = len(np.unique(state_ids))
 
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
   



def holog(ms_name,holog_obs_dict,data_col='DATA',subscan_intent='MIXED',parallel=True):
    '''
    subscan_intent: 'MIXED' or 'REFERENCE'
    '''
    
    ######## Get Spectral Windows ########
    ctb = ctables.table(os.path.join(ms_name,"DATA_DESCRIPTION"),readonly=True, lockoptions={'option': 'usernoread'}, ack=False) #nomodify=True when using CASA tables.
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
    obs_modes = ctb.getcol("OBS_MODE") #scan intent (with subscan intent) is stored in the OBS_MODE column of the STATE subtable.
    ctb.close()
    
    scan_intent = 'MAP_ANTENNA_SURFACE'
    state_ids = []
    for i,ob in enumerate(obs_modes):
        if (scan_intent in ob) and (subscan_intent in ob):
            state_ids.append(i)


    for ddi in holog_obs_dict:
        for scan in [2]: #holog_obs_dict[ddi]
            map_ant_ids = np.nonzero(np.in1d(ant_name,holog_obs_dict[ddi][scan]['map']))[0]
            ref_ant_ids = np.nonzero(np.in1d(ant_name,holog_obs_dict[ddi][scan]['ref']))[0]
            print(map_ant_ids,ref_ant_ids)
            holog_chunk(ms_name,data_col,ddi,scan,map_ant_ids,ref_ant_ids,state_ids)

    
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
