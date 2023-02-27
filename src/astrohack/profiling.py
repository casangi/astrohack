import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from astrohack import holog
from astrohack.dio import extract_holog

ms_name = '/users/jhoskins/fornax/Development/alma_band3.calibrated.DA44.ms'

DA = [
    'DA41', 'DA42', 'DA43', 
    'DA44', 'DA45', 'DA46', 
    'DA48', 'DA49', 'DA50', 
    'DA51', 'DA52', 'DA53', 
    'DA54', 'DA55', 'DA56', 
    'DA57', 'DA58', 'DA59'
]
DV = [
    'DV02', 'DV03', 'DV04', 
    'DV11', 'DV12', 'DV13', 
    'DV14', 'DV15', 'DV16', 
    'DV17', 'DV18', 'DV19', 
    'DV20', 'DV21', 'DV22', 
    'DV23', 'DV24', 'DV25'
] 

holog_obs_description = {
    0:{
        4:{
            'map':DA,
            'ref':DV
        }
    }
}

start = time.time()
extract_holog(
    ms_name=ms_name, 
    holog_name='hack_file', 
    holog_obs_dict=holog_obs_description,
    data_col='DATA',
    subscan_intent='MIXED',
    parallel=True,
    overwrite=True
)

print("Extract: ", time.time() - start)

start = time.time()
holog(
    holog_file='hack_file.holog.zarr', 
    padding_factor=50, 
    parallel=True
)

print("Holog: ", time.time() - start)
