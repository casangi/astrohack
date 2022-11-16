
from astrohack._utils.convert_ms import load_pnt_dict
import os
import xarray as xr

def load_hack_dict(hack_name):
    hack_dict = {}
    pnt_dict = load_pnt_dict(os.path.join(hack_name,'pnt.dict'))
    #print(pnt_dict)
    hack_dict['pnt_dict'] = pnt_dict


    pnt_dict = {}
    for ddi in os.listdir(hack_name):
        if ddi.isnumeric():
            hack_dict[int(ddi)] = {}
            for scan in os.listdir(os.path.join(hack_name,ddi)):
                if scan.isnumeric():
                    hack_dict[int(ddi)][int(scan)]={}
                    for ant_id in os.listdir(os.path.join(hack_name,ddi+'/'+scan)):
                        if ant_id.isnumeric():
                            mapping_ant_vis_holog_data_name = os.path.join(hack_name,ddi+'/'+scan+'/'+ant_id)
                            hack_dict[int(ddi)][int(scan)][int(ant_id)] = xr.open_zarr(mapping_ant_vis_holog_data_name)
                            #print(ddi,scan,ant_id)
                            #hack_dict[int(f)] =  xr.open_zarr(os.path.join(file, f))


    return hack_dict

