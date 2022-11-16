import dask
import time
import os

import numpy as np
import xarray as xr
import dask.array as da

from casacore import tables
from numba import njit
from numba.core import types
from numba.typed import Dict
from casacore import tables as ctables

jit_cache =  False
#To do
# Check if weight spectrum is present and use that.
# If antenna is flagged print out antenna name and id.
# Add logging (but not in numba code).

# Remove all trace of casa table tool


#### Pointing Table Conversion ####
def load_pnt_dict(file):
    """_summary_

    Args:
        file (zarr): Input zarr file containing pointing dictionary

    Returns:
        dict: Pointing dictionary
    """
    pnt_dict = {}
    for f in os.listdir(file):
        
        if f.isnumeric():
            pnt_dict[int(f)] =  xr.open_zarr(os.path.join(file, f))
    return pnt_dict


def make_ant_pnt_xds(ms_name, ant_id, pnt_name):
    """_summary_

    Args:
        ms_name (str): Measurement file
        ant_id (int): Antenna id
        pnt_name (str): _description_
    """

    tb = tables.taql('select DIRECTION,TIME,TARGET,ENCODER,ANTENNA_ID,POINTING_OFFSET from %s WHERE ANTENNA_ID == %s' % (os.path.join(ms_name, "POINTING"), ant_id))

    ### NB: Add check if directions refrence frame is Azemuth Elevation (AZELGEO)
    direction = tb.getcol('DIRECTION')[:,0,:]
    target = tb.getcol('TARGET')[:,0,:]
    encoder = tb.getcol('ENCODER')
    direction_time = tb.getcol('TIME')
    pointing_offset = tb.getcol('POINTING_OFFSET')[:,0,:]
    
    #print(direction.shape, target.shape, encoder.shape, direction_time.shape, pointing_offset.shape)
    tb.close()

    '''Using CASA table tool
    tb = table()
    tb.open(os.path.join(ms_name,"POINTING"), nomodify=True, lockoptions={'option': 'usernoread'})
    pt_ant_table = tb.taql('select DIRECTION,TIME,TARGET,ENCODER,ANTENNA_ID,POINTING_OFFSET from %s WHERE ANTENNA_ID == %s' % (os.path.join(ms_name,"POINTING"),ant_id))
    
    ### NB: Add check if directions refrence frame is Azemuth Elevation (AZELGEO)
    
    direction = np.swapaxes(pt_ant_table.getcol('DIRECTION')[:,0,:],0,1)
    target = np.swapaxes(pt_ant_table.getcol('TARGET')[:,0,:],0,1)
    encoder = np.swapaxes(pt_ant_table.getcol('ENCODER'),0,1)
    direction_time = pt_ant_table.getcol('TIME')
    pointing_offset = np.swapaxes(pt_ant_table.getcol('POINTING_OFFSET')[:,0,:],0,1)
    tb.close()
    '''
    
    pnt_xds = xr.Dataset()
    coords = {'time':direction_time}
    pnt_xds = pnt_xds.assign_coords(coords)
    # Measurement set v2 definition: https://drive.google.com/file/d/1IapBTsFYnUT1qPu_UK09DIFGM81EIZQr/view?usp=sharing
    #DIRECTION: Antenna pointing direction
    pnt_xds['DIRECTION'] = xr.DataArray(direction, dims=('time','az_el'))

    # ENCODER: The current encoder values on the primary axes of the mount type for the antenna, expressed as a Direction 
    # Measure.
    pnt_xds['ENCODER'] = xr.DataArray(encoder, dims=('time','az_el'))

    # TARGET: This is the true expected position of the source, including all coordinate corrections such as precession, 
    # nutation etc.
    pnt_xds['TARGET'] = xr.DataArray(target, dims=('time','az_el'))

    # POINTING_OFFSET: The a priori pointing corrections applied by the telescope in pointing to the DIRECTION position, 
    # optionally expressed as polynomial coefficients.
    pnt_xds['POINTING_OFFSET'] = xr.DataArray(pointing_offset, dims=('time','az_el'))
    
    #Calculate directional cosines (l,m) which are used as the gridding locations.
    # See equations 8,9 in https://library.nrao.edu/public/memos/evla/EVLAM_212.pdf.
    # TARGET: A_s, E_s (target source position)
    # DIRECTION: A_a, E_a (Antenna's pointing direction)
    
    ### NB: Is VLA's definition of Azimuth the same for ALMA, MeerKAT, etc.? (positive for a clockwise rotation from north, viewed from above)
    ### NB: Compare with calulation using WCS in astropy.
    l = np.cos(target[:,1])*np.sin(target[:,0]-direction[:,0])
    m = np.sin(target[:,1])*np.cos(direction[:,1]) - np.cos(target[:,1])*np.sin(direction[:,1])*np.cos(target[:,0]-direction[:,0])
    
    pnt_xds['DIRECTIONAL_COSINES'] = xr.DataArray(np.array([l,m]).T, dims=('time','ra_dec'))
    #time.sleep(30)
    pnt_xds.to_zarr(os.path.join(pnt_name, str(ant_id)), mode='w', compute=True, consolidated=True)


def make_ant_pnt_dict(ms_name, pnt_name, parallel=True):
    """_summary_

    Args:
        ms_name (str): Measurement file name
        pnt_name (str): _description_
        parallel (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    from casatools import table
    tb = table()
    tb.open(os.path.join(ms_name,"ANTENNA"), nomodify=True, lockoptions={'option': 'usernoread'})
    ant_name = tb.getcol("NAME")
    ant_id = np.arange(len(ant_name))
    tb.close()
    
    if parallel:
        delayed_pnt_list = []
        for i_a in ant_id:
            delayed_pnt_list.append(dask.delayed(make_ant_pnt_xds)(dask.delayed(ms_name ), dask.delayed(i_a), dask.delayed(pnt_name)))
        dask.compute(delayed_pnt_list)
    else:
        for i_a in ant_id:
            make_ant_pnt_xds(ms_name, i_a, pnt_name)

    return load_pnt_dict(pnt_name)





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
        pnt_map_dict[antenna] = np.zeros((n_time_vis, 2))
        pnt_map_dict[antenna] = pnt_ant_dict[antenna].interp(time=time_vis, method='nearest').DIRECTIONAL_COSINES.values

    return pnt_map_dict

@njit(cache=jit_cache)
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
    sum_weight_map_dict = {}

    for antenna_id in map_ant_ids:
        vis_map_dict[antenna_id] = np.zeros((n_time, n_chan, n_pol), dtype=types.complex64)
        sum_weight_map_dict[antenna_id] = np.zeros((n_time, n_chan, n_pol), dtype=types.complex64)

        
    # Create sum of weight dict
    
    print(vis_data.dtype)
    
    for row in range(n_row):

        if flag_row is False:
            continue

        ant1_index = ant1[row]
        ant2_index = ant2[row]
        
        if (ant1_index in map_ant_ids) and (ant2_index in ref_ant_ids):
            vis_baseline = vis_data[row, :, :] # n_chan x n_pol
            map_ant_indx = ant1_index # mapping antenna index

        elif (ant2_index in map_ant_ids) and (ant1_index not in ref_ant_ids): #conjugate
            vis_baseline = np.conjugate(vis_data[row, :, :])
            map_ant_indx = ant2_index

        else:
            continue
        
        time_index = np.searchsorted(time_vis, time_vis_row[row])
        
        for chan in range(n_chan):
            for pol in range(n_pol):
                if ~(flag[row, chan, pol]):
                    # Calculate running weighted sum of visibilities
                    vis_map_dict[map_ant_indx][time_index, chan, pol] = vis_map_dict[map_ant_indx][time_index, chan, pol] + vis_baseline[chan, pol]*weight[row, pol]

                    # Calculate running sum of weights
                    sum_weight_map_dict[map_ant_indx][time_index, chan, pol] = sum_weight_map_dict[map_ant_indx][time_index, chan, pol] + weight[row, pol]       

    flagged_mapping_antennas = []

    for map_ant_indx in vis_map_dict.keys():
        sum_of_sum_weight = 0
        
        for time_index in range(n_time):
            for chan in range(n_chan):
                for pol in range(n_pol):
                    sum_weight = sum_weight_map_dict[map_ant_indx][time_index, chan, pol]
                    sum_of_sum_weight = sum_of_sum_weight + sum_weight
                    if sum_weight == 0:
                        vis_map_dict[map_ant_indx][time_index, chan, pol] = 0.
                    else:
                        vis_map_dict[map_ant_indx][time_index, chan, pol] = vis_map_dict[map_ant_indx][time_index, chan, pol]/sum_weight
                        
        if sum_of_sum_weight == 0:
            flagged_mapping_antennas.append(map_ant_indx)

    return vis_map_dict, sum_weight_map_dict, flagged_mapping_antennas
    
def create_map_hack_xds_dict(hack_name,vis_map_dict, weight_map_dict, pnt_map_dict, time, chan, pol, flagged_mapping_antennas, scan,ddi):
    
    map_hack_dict = {}
    print(time.shape,chan.shape,pol.shape)
    coords = {'time':time,'chan':chan,'pol':pol}
    
    for map_ant_indx in vis_map_dict.keys():
        if map_ant_indx not in flagged_mapping_antennas:
            xds = xr.Dataset()
            xds = xds.assign_coords(coords)
            xds['VIS'] = xr.DataArray(vis_map_dict[map_ant_indx],dims=['time','chan','pol'])
            xds['WEIGHT'] = xr.DataArray(weight_map_dict[map_ant_indx],dims=['time','chan','pol'])
            xds['DIRECTIONAL_COSINES'] = xr.DataArray(pnt_map_dict[map_ant_indx],dims=['time','lm'])
            xds.attrs['scan'] = scan
            xds.attrs['ant_id'] = map_ant_indx
            xds.attrs['ddi'] = ddi
            xds.to_zarr(os.path.join(hack_name, str(ddi) + '/' + str(scan) + '/' + str(map_ant_indx)), mode='w', compute=True, consolidated=True)
            #print(xds)
        else:
            print('In scan ', scan, ' antenna ', map_ant_indx, ' is flagged')
        
        
    
            
def extract_holog_chunk(extract_holog_parms):
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
    
    ms_name = extract_holog_parms['ms_name']
    pnt_name = extract_holog_parms['pnt_name']
    data_col = extract_holog_parms['data_col']
    ddi = extract_holog_parms['ddi']
    scan = extract_holog_parms['scan']
    map_ant_ids = extract_holog_parms['map_ant_ids']
    ref_ant_ids = extract_holog_parms['ref_ant_ids']
    sel_state_ids = extract_holog_parms['sel_state_ids']
    hack_name = extract_holog_parms['hack_name']
    
    chan_freq = extract_holog_parms['chan_setup']['chan_freq']
    pol = extract_holog_parms['pol_setup']['pol']
    
    print(extract_holog_parms.keys())
    
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

    '''
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
    '''
    ctb.close()
    
    ###################################
    
    print(time.time()-start)
    #print(vis_data.shape, weight.shape, ant1.shape, ant2.shape, time_vis_row.shape, time_vis_centroid_row.shape, flag.shape, flag_row.shape)
    
    start = time.time()
    time_vis, unique_index = np.unique(time_vis_row, return_index=True) # Note that values are sorted.
    state_ids = state_ids_row[unique_index]
    
    print('Time to unique ',time.time()-start)

    start = time.time()
    # vis_map_dict: [mapping_antenna](time_index, chan, polarization)
    # weight_map_dict: [mapping_antenna](time_index, chan, polarization)
    vis_map_dict, weight_map_dict, flagged_mapping_antennas = extract_holog_chunk_jit(vis_data, weight, ant1, ant2, time_vis_row, time_vis, flag, flag_row, map_ant_ids, ref_ant_ids)
    print('Time to extract_holog_chunk_jit ',time.time()-start)


    pnt_ant_dict = load_pnt_dict(pnt_name)
    
    start = time.time()
    pnt_map_dict = extract_pointing_chunk(map_ant_ids, time_vis, pnt_ant_dict)
    print('Time to extract_pointing_chunk ',time.time()-start)
    
    start = time.time()
    hack_dict  = create_map_hack_xds_dict(hack_name,vis_map_dict, weight_map_dict, pnt_map_dict, time_vis, chan_freq, pol, flagged_mapping_antennas, scan, ddi)
    print('create_map_hack_xds_dict ',time.time()-start)
    
    print('Done')

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
   
