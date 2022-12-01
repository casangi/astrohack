import dask
import time
import os

import numpy as np
import xarray as xr
import dask.array as da
import zarr
import copy

import astropy
import astropy.units as u
import astropy.coordinates as coord

from numba import njit
from numba.core import types
from numba.typed import Dict
from casacore import tables as ctables

DIMENSION_KEY = "_ARRAY_DIMENSIONS"

jit_cache =  False
# To do
#   - Check if weight spectrum is present and use that.
#   - If antenna is flagged print out antenna name and id.
#   - Add logging (but not in numba code).

# Remove all trace of casa table tool

def _calculate_parallactic_angle_chunk(time_samples, observing_location, direction, indicies, dir_frame='FK5', zenith_frame='FK5'):
    """
    Converts a direction and zenith (frame FK5) to a topocentric Altitude-Azimuth (https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html) 
    frame centered at the observing_location (frame ITRF) for a UTC time. The parallactic angles is calculated as the position angle of the Altitude-Azimuth 
    direction and zenith.
    
    Parameters
    ----------
    time_samples: str np.array, [n_time], 'YYYY-MM-DDTHH:MM:SS.SSS'
        UTC time series. Example '2019-10-03T19:00:00.000'.
    observing_location: int np.array, [3], [x,y,z], meters
        ITRF geocentric coordinates.
    direction: float np.array, [n_time,2], [time,ra,dec], radians
        The pointing direction.
    Returns
    -------
    parallactic_angles: float np.array, [n_time], radians
        An array of parallactic angles.
    """
    
    observing_location = coord.EarthLocation.from_geocentric(x=observing_location[0]*u.m, y=observing_location[1]*u.m, z=observing_location[2]*u.m)
    
    direction = np.take(direction, indicies, axis=0)

    direction = coord.SkyCoord(ra=direction[:,0]*u.rad, dec=direction[:,1]*u.rad, frame=dir_frame.lower())
    zenith = coord.SkyCoord(0, 90, unit=u.deg, frame=zenith_frame.lower())
    
    altaz_frame = coord.AltAz(location=observing_location, obstime=time_samples)
    zenith_altaz = zenith.transform_to(altaz_frame)
    direction_altaz = direction.transform_to(altaz_frame)
    
    return direction_altaz.position_angle(zenith_altaz)

def _get_attrs(zarr_obj):
    '''
    get attributes of zarr obj (groups or arrays)
    '''
    return {
        k: v
        for k, v in zarr_obj.attrs.asdict().items()
        if not k.startswith("_NC")
    }

def _open_no_dask_zarr(zarr_name,slice_dict={}):
    '''
        Alternative to xarray open_zarr where the arrays are not Dask Arrays.
        
        slice_dict: A dictionary of slice objects for which values to read form a dimension.
                    For example silce_dict={'time':slice(0,10)} would select the first 10 elements in the time dimension.
                    If a dim is not specified all values are retruned.
        return:
            xarray.Dataset()
    '''
    
    zarr_group = zarr.open_group(store=zarr_name,mode='r')
    group_attrs = _get_attrs(zarr_group)
    
    slice_dict_complete = copy.deepcopy(slice_dict)
    coords = {}
    xds = xr.Dataset()
    for var_name, var in zarr_group.arrays():
        var_attrs = _get_attrs(var)
        
        for dim in var_attrs[DIMENSION_KEY]:
            if dim not in slice_dict_complete:
                slice_dict_complete[dim] = slice(None) #No slicing.
                
        if (var_attrs[DIMENSION_KEY][0] == var_name) and (len(var_attrs[DIMENSION_KEY]) == 1):
            coords[var_name] = var[slice_dict_complete[var_attrs[DIMENSION_KEY][0]]] #Dimension coordinates.
        else:
            #Construct slicing
            slicing_list = []
            for dim in var_attrs[DIMENSION_KEY]:
                slicing_list.append(slice_dict_complete[dim])
            slicing_tuple = tuple(slicing_list)
            xds[var_name] = xr.DataArray(var[slicing_tuple],dims=var_attrs[DIMENSION_KEY])
            
    xds = xds.assign_coords(coords)
    
    xds.attrs = group_attrs
    return xds

#### Pointing Table Conversion ####
def _load_pnt_dict(file, ant_list=None, dask_load=True):
    """ Load pointing dictionary from disk.

    Args:
        file (zarr): Input zarr file containing pointing dictionary.

    Returns:
        dict: Pointing dictionary
    """
    pnt_dict = {}

    for f in os.listdir(file):
        if f.isnumeric():
            if (ant_list is None) or (int(f) in ant_list):
                if dask_load:
                    pnt_dict[int(f)] = xr.open_zarr(os.path.join(file, f))
                else:
                    pnt_dict[int(f)] = _open_no_dask_zarr(os.path.join(file, f))
    

    return pnt_dict


def _make_ant_pnt_xds_chunk(ms_name, ant_id, pnt_name):
    """ Extract subset of pointing table data into a dictionary of xarray dataarrays. This is written to disk as a zarr file.
            This function processes a chunk the overalll data and is managed by Dask.

    Args:
        ms_name (str): Measurement file name.
        ant_id (int): Antenna id
        pnt_name (str): Name of output poitning dictinary file name.
    """

    tb = ctables.taql('select DIRECTION, TIME, TARGET, ENCODER, ANTENNA_ID, POINTING_OFFSET from %s WHERE ANTENNA_ID == %s' % (os.path.join(ms_name, "POINTING"), ant_id))

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
    
    # Calculate directional cosines (l,m) which are used as the gridding locations.
    # See equations 8,9 in https://library.nrao.edu/public/memos/evla/EVLAM_212.pdf.
    # TARGET: A_s, E_s (target source position)
    # DIRECTION: A_a, E_a (Antenna's pointing direction)
    
    ### NB: Is VLA's definition of Azimuth the same for ALMA, MeerKAT, etc.? (positive for a clockwise rotation from north, viewed from above)
    ### NB: Compare with calulation using WCS in astropy.
    l = np.cos(target[:,1])*np.sin(target[:,0]-direction[:,0])
    m = np.sin(target[:,1])*np.cos(direction[:,1]) - np.cos(target[:,1])*np.sin(direction[:,1])*np.cos(target[:,0]-direction[:,0])
    
    pnt_xds['DIRECTIONAL_COSINES'] = xr.DataArray(np.array([l,m]).T, dims=('time','ra_dec'))
    
    pnt_xds.to_zarr(os.path.join(pnt_name, str(ant_id)), mode='w', compute=True, consolidated=True)


def _make_ant_pnt_dict(ms_name, pnt_name, parallel=True):
    """ Top level function to extract subset of pointing table data into a dictionary of xarray dataarrays.

    Args:
        ms_name (str): Measurement file name.
        pnt_name (str): Output pointing dictionary file name.
        parallel (bool, optional): Process in parallel. Defaults to True.

    Returns:
        dict: pointing dictionary of xarray dataarrays
    """
    
    ctb = ctables.table(os.path.join(ms_name,"ANTENNA"), readonly=True, lockoptions={'option': 'usernoread'})
    
    antenna_name = ctb.getcol("NAME")
    antenna_id = np.arange(len(antenna_name))
    
    ctb.close()
    
    if parallel:
        delayed_pnt_list = []
        for id in antenna_id:
            delayed_pnt_list.append(dask.delayed(_make_ant_pnt_xds_chunk)(dask.delayed(ms_name ), dask.delayed(id), dask.delayed(pnt_name)))
        dask.compute(delayed_pnt_list)
    else:
        for id in antenna_id:
            _make_ant_pnt_xds_chunk(ms_name, id, pnt_name)

    return _load_pnt_dict(pnt_name)

def _extract_pointing_chunk(map_ant_ids, time_vis, pnt_ant_dict):
    """ Extract nearest MAIN table time indexed pointing map

    Args:
        map_ant_ids (dict): list of antenna ids
        time_vis (numpy.ndarray): sorted, unique list of visibility times
        pnt_ant_dict (dict): map of pointing directional cosines with a map key based on the antenna id and indexed by the MAIN table visibility time.

    Returns:
        dict:  Dictionary of directional cosine data mapped to nearest MAIN table sample times.
    """

    n_time_vis = time_vis.shape[0]

    pnt_map_dict = {}

    for antenna in map_ant_ids:
        pnt_map_dict[antenna] = np.zeros((n_time_vis, 2))
        pnt_map_dict[antenna] = pnt_ant_dict[antenna].interp(time=time_vis, method='nearest').DIRECTIONAL_COSINES.values

    return pnt_map_dict

@njit(cache=jit_cache, nogil=True)
def _extract_holog_chunk_jit(vis_data, weight, ant1, ant2, time_vis_row, time_vis, flag, flag_row, map_ant_ids, ref_ant_ids):
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
    
    for row in range(n_row):

        if flag_row is False:
            continue

        ant1_index = ant1[row]
        ant2_index = ant2[row]
        
        if (ant1_index in map_ant_ids) and (ant2_index in ref_ant_ids):
            vis_baseline = vis_data[row, :, :] # n_chan x n_pol
            map_ant_index = ant1_index # mapping antenna index

        elif (ant2_index in map_ant_ids) and (ant1_index not in ref_ant_ids): #conjugate
            vis_baseline = np.conjugate(vis_data[row, :, :])
            map_ant_index = ant2_index

        else:
            continue
        
        time_index = np.searchsorted(time_vis, time_vis_row[row])
        
        for chan in range(n_chan):
            for pol in range(n_pol):
                if ~(flag[row, chan, pol]):
                    # Calculate running weighted sum of visibilities
                    vis_map_dict[map_ant_index][time_index, chan, pol] = vis_map_dict[map_ant_index][time_index, chan, pol] + vis_baseline[chan, pol]*weight[row, pol]

                    # Calculate running sum of weights
                    sum_weight_map_dict[map_ant_index][time_index, chan, pol] = sum_weight_map_dict[map_ant_index][time_index, chan, pol] + weight[row, pol]       

    flagged_mapping_antennas = []

    for map_ant_index in vis_map_dict.keys():
        sum_of_sum_weight = 0
        
        for time_index in range(n_time):
            for chan in range(n_chan):
                for pol in range(n_pol):
                    sum_weight = sum_weight_map_dict[map_ant_index][time_index, chan, pol]
                    sum_of_sum_weight = sum_of_sum_weight + sum_weight
                    if sum_weight == 0:
                        vis_map_dict[map_ant_index][time_index, chan, pol] = 0.
                    else:
                        vis_map_dict[map_ant_index][time_index, chan, pol] = vis_map_dict[map_ant_index][time_index, chan, pol]/sum_weight
                        
        if sum_of_sum_weight == 0:
            flagged_mapping_antennas.append(map_ant_index)

    return vis_map_dict, sum_weight_map_dict, flagged_mapping_antennas

def _get_time_samples(time_vis):
    """_summary_

    Args:
        time_vis (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_time_vis = time_vis.shape[0]

    middle = int(n_time_vis*0.5)-1
    indicies = [0, middle, n_time_vis - 1]

    return np.take(time_vis, indicies), indicies

    
def _create_hack_file(hack_name, vis_map_dict, weight_map_dict, pnt_map_dict, time, chan, pol, flagged_mapping_antennas, scan, ddi, ms_name):
    """ Create hack-structured, formatted output file and save to zarr.

    Args:
        hack_name (str): Hack file name.
        vis_map_dict (dict): _description_
        weight_map_dict (dict): _description_
        pnt_map_dict (dict): _description_
        time (numpy.ndarray): _description_
        chan (numpy.ndarray): _description_
        pol (numpy.ndarray): _description_
        flagged_mapping_antennas (numpy.ndarray): _description_
        scan (numpy.ndarray): _description_
        ddi (numpy.ndarray): _description_
    """

    ctb = ctables.table("/".join((ms_name, "ANTENNA")))
    observing_location = ctb.getcol("POSITION")
    ctb.close()

    time_vis_days = time/(3600*24)
    astro_time_vis = astropy.time.Time(time_vis_days, format='mjd')
    time_samples, indicies = _get_time_samples(astro_time_vis)

    coords = {'time':time, 'chan':chan, 'pol':pol}
    
    for map_ant_index in vis_map_dict.keys():
        if map_ant_index not in flagged_mapping_antennas:
            
            parallactic_samples = _calculate_parallactic_angle_chunk(
                time_samples=time_samples, 
                observing_location=observing_location[map_ant_index], 
                direction=pnt_map_dict[map_ant_index], 
                indicies=indicies
            )

            
            xds = xr.Dataset()
            xds = xds.assign_coords(coords)
            xds['VIS'] = xr.DataArray(vis_map_dict[map_ant_index], dims=['time','chan','pol'])
            xds['WEIGHT'] = xr.DataArray(weight_map_dict[map_ant_index], dims=['time','chan','pol'])
            xds['DIRECTIONAL_COSINES'] = xr.DataArray(pnt_map_dict[map_ant_index], dims=['time','lm'])
            xds.attrs['scan'] = scan
            xds.attrs['ant_id'] = map_ant_index
            xds.attrs['ddi'] = ddi
            xds.attrs['parallactic_samples'] = parallactic_samples.to_string()
            xds.to_zarr(os.path.join(hack_name, str(ddi) + '/' + str(scan) + '/' + str(map_ant_index)), mode='w', compute=True, consolidated=True)
            
        else:
            print('In scan ', scan, ' antenna ', map_ant_index, ' is flagged')
        
        
    
            
def _extract_holog_chunk(extract_holog_parms):
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
    print('vis data type ',vis_data.dtype)
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
    
      
    time_vis, unique_index = np.unique(time_vis_row, return_index=True) # Note that values are sorted.
    state_ids = state_ids_row[unique_index]

    vis_map_dict, weight_map_dict, flagged_mapping_antennas = _extract_holog_chunk_jit(vis_data, weight, ant1, ant2, time_vis_row, time_vis, flag, flag_row, map_ant_ids, ref_ant_ids)
    
    del vis_data, weight, ant1, ant2, time_vis_row, flag, flag_row 

    pnt_ant_dict = _load_pnt_dict(pnt_name,map_ant_ids, dask_load=False)
    
    pnt_map_dict = _extract_pointing_chunk(map_ant_ids, time_vis, pnt_ant_dict)    
    
    hack_dict  = _create_hack_file(hack_name, vis_map_dict, weight_map_dict, pnt_map_dict, time_vis, chan_freq, pol, flagged_mapping_antennas, scan, ddi, ms_name)
    
    print('Done')

    # Grid all subscans onto a single grid
    # bm_map_dict = create_beam_maps(vis_map_dict, pnt_map_dict, map_ant_ids, state_ids, time_vis) # each mapping antenna has an image cube of dims: n_state_ids (time) x nchan x pol x l x m, n_state_ids = len(np.unique(state_ids))
 

