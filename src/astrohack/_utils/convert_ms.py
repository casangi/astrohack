import dask
import time
import os

import numpy as np
import xarray as xr
import dask.array as da

from casacore import tables


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
    pnt_xds['DIRECTION'] = xr.DataArray(direction,dims=('time','az_el'))

    # ENCODER: The current encoder values on the primary axes of the mount type for the antenna, expressed as a Direction 
    # Measure.
    pnt_xds['ENCODER'] = xr.DataArray(encoder,dims=('time','az_el'))

    # TARGET: This is the true expected position of the source, including all coordinate corrections such as precession, 
    # nutation etc.
    pnt_xds['TARGET'] = xr.DataArray(target,dims=('time','az_el'))

    # POINTING_OFFSET: The a priori pointing corrections applied by the telescope in pointing to the DIRECTION position, 
    # optionally expressed as polynomial coefficients.
    pnt_xds['POINTING_OFFSET'] = xr.DataArray(pointing_offset,dims=('time','az_el'))
    
    #Calculate directional cosines (l,m) which are used as the gridding locations.
    # See equations 8,9 in https://library.nrao.edu/public/memos/evla/EVLAM_212.pdf.
    # TARGET: A_s, E_s (target source position)
    # DIRECTION: A_a, E_a (Antenna's pointing direction)
    
    ### NB: Is VLA's definition of Azimuth the same for ALMA, MeerKAT, etc.? (positive for a clockwise rotation from north, viewed from above)
    ### NB: Compare with calulation using WCS in astropy.
    l = np.cos(target[:,1])*np.sin(target[:,0]-direction[:,0])
    m = np.sin(target[:,1])*np.cos(direction[:,1]) - np.cos(target[:,1])*np.sin(direction[:,1])*np.cos(target[:,0]-direction[:,0])
    
    pnt_xds['DIRECTIONAL_COSINES'] = xr.DataArray(np.array([l,m]).T,dims=('time','ra_dec'))
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


