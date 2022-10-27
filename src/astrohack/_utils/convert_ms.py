import numpy as np
import xarray as xr
import dask.array as da
import dask
import time
from casatools import table
import os

#### Pointing Table Conversion ####
def load_pnt_dict(file):
    pnt_dict = {}
    for f in os.listdir(file):
        #print(os.path.join(file,f))
        if f.isnumeric():
            pnt_dict[f] =  xr.open_zarr(os.path.join(file,f))
    return pnt_dict


def make_ant_pnt_xds(ms_name,ant_id,pnt_name):
    tb = table()
    tb.open(os.path.join(ms_name,"POINTING"), nomodify=True, lockoptions={'option': 'usernoread'})
    pt_ant_table = tb.taql('select DIRECTION,TIME,ANTENNA_ID from %s WHERE ANTENNA_ID == %s' % (os.path.join(ms_name,"POINTING"),ant_id))
    
    dir = np.swapaxes(pt_ant_table.getcol('DIRECTION')[:,0,:],0,1)
    dir_time = pt_ant_table.getcol('TIME')
    tb.close()
    
    #print(dir_time.shape)
    #print(dir.shape)
    
    #Maybe we should do the conversion to l,m now?
    pnt_xds = xr.Dataset()
    coords = {'time':dir_time}
    pnt_xds = pnt_xds.assign_coords(coords)
    pnt_xds['DIRECTION'] = xr.DataArray(dir,dims=('time','ra_dec'))

    #time.sleep(30)
    pnt_xds.to_zarr(os.path.join(pnt_name,str(ant_id)),mode='w',compute=True,consolidated=True)


def make_ant_pnt_dict(ms_name,pnt_name,parallel=True):
    tb = table()
    tb.open(os.path.join(ms_name,"ANTENNA"), nomodify=True, lockoptions={'option': 'usernoread'})
    ant_name = tb.getcol("NAME")
    ant_id = np.arange(len(ant_name))
    tb.close()
    
    if parallel:
        delayed_pnt_list = []
        for i_a in ant_id:
            delayed_pnt_list.append(dask.delayed(make_ant_pnt_xds)(dask.delayed(ms_name ),dask.delayed(i_a),dask.delayed(pnt_name)))
        dask.compute(delayed_pnt_list)
    else:
        for i_a in ant_id:
            make_ant_pnt_xds(ms_name,i_a,pnt_name)

    return load_pnt_dict(pnt_name)
