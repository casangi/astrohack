import pytest

import shutil
import os
import json
import copy

import numpy as np

from astrohack.gdown_utils import gdown_data

from astrohack._utils._tools import _jsonify

from astrohack.extract_holog import extract_holog
from astrohack.holog import holog
from astrohack.panel import panel

from astrohack.astrohack_client import astrohack_local_client

base_name = 'ea25_cal_small_'

client = astrohack_local_client(cores=2, memory_limit='8GB', log_parms={'log_level':'DEBUG'})

@pytest.fixture(scope='session')
def set_data(tmp_path_factory):
  data_dir = tmp_path_factory.mktemp("data")

  gdown_data(ms_name='ea25_cal_small_before_fixed.split.ms', download_folder=str(data_dir))
  gdown_data(ms_name='ea25_cal_small_after_fixed.split.ms', download_folder=str(data_dir))

  return data_dir

from astrohack.dio import open_panel

def verify_panel_shifts(
  data_dir="",
  panel_list=['3-4', '5-27', '5-37', '5-38'], 
  expected_shift=[np.array([-100, 75, 0, 150])],
  ref_mean_shift = np.array([-112.2325, 73.225, -1.455, 139.04  ]),
  antenna='ant_ea25',
  ddi='ddi_0'
):
    
    M_TO_MILS = 39370.1
    
    before_mds = open_panel('{data}/before.split.panel.zarr'.format(data=data_dir))
    after_mds = open_panel('{data}/after.split.panel.zarr'.format(data=data_dir))
    
    before_shift = before_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values*M_TO_MILS
    after_shift = after_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values*M_TO_MILS
    
    difference = after_shift - before_shift
    
    mean_shift = np.mean(difference, axis=1)
    
    delta_mean_shift = np.abs(mean_shift - expected_shift)
    delta_ref_shift = np.abs(ref_mean_shift - expected_shift)
        
    delta_shift = delta_mean_shift - delta_ref_shift  # New corrections - old corrections --> delta if delta < 0 ==> we improved.
    
    if np.any(np.abs(delta_shift) > 0.0039): # This is approximately a micron shift
        print("There were changes!")
        for i, delta in enumerate(delta_shift[0]):
            if delta < 0:
                print("{panel}, improved by {delta} mils".format(panel=panel_list[i], delta=delta))
            else:
                print("{panel}, got worse by {delta} mils".format(panel=panel_list[i], delta=delta))

def verify_holog_obs_dictionary(holog_obs_dict):

    ref_holog_obj = {}
    ref_holog_obj = copy.deepcopy(holog_obs_dict)

    _jsonify(ref_holog_obj)

    with open(".holog_obs_dict.json") as json_file:
        holog_obj = json.load(json_file)
                          
    assert holog_obj == ref_holog_obj, "Error: holog_obs_descrition dictionary has changes unexpectedly."
    
    os.remove(".holog_obs_dict.json")   

def verify_holog_diagnostics(cell_size, grid_size, number_of_digits=7):
    
    with open(".holog_diagnostic.json") as json_file:
        json_data = json.load(json_file)
        
    json_data['cell_size'] = [round(x, number_of_digits) for x in json_data['cell_size']]
        
    assert (json_data['cell_size'] == cell_size).all(), "Unexpected change in cell_size occured."
    assert (json_data['grid_size'] == grid_size).all(), "Unexpected change in grid_size occured."

def test_holog_obs_dict(set_data):
    before_ms = str(set_data/"".join((base_name,"before_fixed.split.ms")))
    before_holog = str(set_data/"before.split.holog.zarr")
    after_ms =  str(set_data/"".join((base_name, "after_fixed.split.ms")))
    after_holog = str(set_data/"after.split.holog.zarr")

    holog_obs_dict = {
      'ddi_0': {
        'map_0': {
            'ant': {
                'ea06': np.array(['ea04']),
                'ea25': np.array(['ea04'])
            },
            'scans': np.array([
                8, 9, 10, 12, 13, 14, 16, 17, 18, 23, 24, 
                25, 27, 28, 29, 31, 32, 33, 38, 39, 40, 
                42, 43, 44, 46, 47, 48, 53, 54, 55, 57
            ])
          }
        }
    }


    extract_holog(
      ms_name=before_ms, 
      holog_name=before_holog, 
      ddi_sel=[0],
      data_col='CORRECTED_DATA',
      parallel=False,
      overwrite=True,
      reuse_point_zarr=False
    )

    verify_holog_obs_dictionary(holog_obs_dict)

    holog_obs_dict = {
      'ddi_0': {
          'map_0': {
              'ant': {
                  'ea06': np.array(['ea04']),
                  'ea25': np.array(['ea04'])
              },
              'scans': np.array([
                8,  9, 10, 12, 13, 14, 16, 17, 18, 
                23, 24, 25, 27, 28, 29, 31, 32, 33, 
                38, 39, 40, 42, 43, 44, 46, 47, 48, 
                53, 54, 55, 57
              ])
            }
        },
        'ddi_1': {
          'map_0': {
              'ant': {
                'ea06': np.array(['ea04']),
                'ea25': np.array(['ea04'])
              },
              'scans': np.array([
                8,  9, 10, 12, 13, 14, 16, 17, 18, 
                23, 24, 25, 27, 28, 29, 31, 32, 33, 
                38, 39, 40, 42, 43, 44, 46, 47, 48, 
                53, 54, 55, 57
              ])
            }
        }
    }


    holog_mds_after, _ = extract_holog(
        ms_name=after_ms, 
        holog_name=after_holog,
        data_col='CORRECTED_DATA',
        parallel=False,
        overwrite=True,
        reuse_point_zarr=False
    )

    verify_holog_obs_dictionary(holog_obs_dict)

def test_holog(set_data):

  before_holog = str(set_data/"before.split.holog.zarr")
  after_holog = str(set_data/"after.split.holog.zarr")
  
  holog(
    holog_name=before_holog, 
    padding_factor=50, 
    grid_interpolation_mode='linear',
    chan_average = True,
    reference_scaling_frequency=None,
    scan_average = True,
    overwrite=True,
    phase_fit=True,
    apply_mask=True,
    to_stokes=True,
    parallel=True
  )

  verify_holog_diagnostics(
    cell_size = np.array([-0.0006442, 0.0006442]),
    grid_size = np.array([31, 31]),
    number_of_digits=7
  )

  holog(
    holog_name=after_holog, 
    padding_factor=50, 
    grid_interpolation_mode='linear',
    chan_average = True,
    reference_scaling_frequency=None,
    scan_average = True,
    overwrite=True,
    phase_fit=True,
    apply_mask=True,
    to_stokes=True,
    parallel=True
  )

  verify_holog_diagnostics(
    cell_size = np.array([-0.0006442, 0.0006442]),
    grid_size = np.array([31, 31]),
    number_of_digits=7
  )

def test_screw_adjustments(set_data):
  before_image = str(set_data/"before.split.image.zarr")
  after_image = str(set_data/"after.split.image.zarr")

  before_panel = panel(
    image_name=before_image, 
    panel_model='rigid',
    parallel=True,
    overwrite=True
  )
  after_panel = panel(
    image_name=after_image, 
    panel_model='rigid',
    parallel=True,
    overwrite=True
  )

  verify_panel_shifts(data_dir=str(set_data))
