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
  expected_shift=np.array([-100, 75, 0, 150]),
  ref_mean_shift = np.array([-112.17789033, 73.22619286, -1.53666468, 138.99617087]),
  antenna='ant_ea25',
  ddi='ddi_0'
):
    def relative_difference(mean, expected):  
      return 2*np.abs(mean - expected)/(abs(mean) + abs(expected))
    
    M_TO_MILS = 39370.1
    
    before_mds = open_panel('{data}/before.split.panel.zarr'.format(data=data_dir))
    after_mds = open_panel('{data}/after.split.panel.zarr'.format(data=data_dir))
    
    before_shift = before_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values*M_TO_MILS
    after_shift = after_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values*M_TO_MILS
    
    difference = after_shift - before_shift
    
    mean_shift = np.mean(difference, axis=1)

    delta_mean_shift = np.abs(mean_shift - expected_shift)
    delta_ref_shift = np.abs(ref_mean_shift - expected_shift)
        
    delta_shift = delta_mean_shift - delta_ref_shift   # New corrections - old corrections --> delta if delta < 0 ==> we improved.
    relative_shift = relative_difference(delta_mean_shift, delta_ref_shift)
    
    return np.all(relative_shift <= 1e-6)

def verify_center_pixels(file, reference_center_pixels, number_of_digits=7):
    from astrohack.dio import open_image
    
    mds = open_image(file)['ant_ea25']['ddi_0']
    
    aperture_shape = mds.APERTURE.values.shape[-2], mds.APERTURE.values.shape[-1]
    beam_shape = mds.BEAM.values.shape[-2], mds.BEAM.values.shape[-1]    
    
    aperture_center_pixels = mds.APERTURE.values[..., aperture_shape[0]//2, aperture_shape[1]//2]
    beam_center_pixels = mds.BEAM.values[..., beam_shape[0]//2, beam_shape[1]//2]
    
    aperture_check = np.all(np.round(reference_center_pixels['aperture'], number_of_digits) == np.round(aperture_center_pixels, number_of_digits))
    beam_check = np.all(np.round(reference_center_pixels['beam'], number_of_digits) == np.round(beam_center_pixels, number_of_digits))

    return aperture_check and beam_check
    
    
                
def verify_holog_obs_dictionary(holog_obs_dict):

    ref_holog_obj = {}
    ref_holog_obj = copy.deepcopy(holog_obs_dict)

    _jsonify(ref_holog_obj)

    with open(".holog_obs_dict.json") as json_file:
        holog_obj = json.load(json_file)
                          
    return  holog_obj == ref_holog_obj

def verify_holog_diagnostics(cell_size, grid_size, number_of_digits=7):
    
    with open(".holog_diagnostic.json") as json_file:
        json_data = json.load(json_file)
        
    json_data['cell_size'] = np.array([round(x, number_of_digits) for x in json_data['cell_size']])
        
    return np.all(json_data['cell_size'] == cell_size) and np.all(json_data['grid_size'] == grid_size)

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
      data_column='CORRECTED_DATA',
      parallel=False,
      overwrite=True,
      reuse_point_zarr=False
    )

    assert verify_holog_obs_dictionary(holog_obs_dict)

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
        data_column='CORRECTED_DATA',
        parallel=False,
        overwrite=True,
        reuse_point_zarr=False
    )

    assert verify_holog_obs_dictionary(holog_obs_dict)

def test_holog(set_data):
  reference_before_dict = {
    'aperture': np.array([
        [
            [ 
                0.225725131776915+0.12054326352090801j, 
                0.016838151657196862-0.011171227001989446j, 
                0.048663223170791664-0.024197325651061738j, 
                -0.09340400975688747-0.045923630603716084j
            ]
        ]
    ]),
    'beam': np.array([
        [
            [ 
                0.993988815582417+0.10948166283475885j,
                0.005367471552903289+0.01583907120584613j,
                0.014425807274417096+0.0059018780911659395j,
                -0.004021223235578464+0.00040825140567029433j
            ]
        ]
    ])
  }

  reference_after_dict = {
    'aperture': np.array([
        [
            [ 
                0.12300077664655817-0.06757802469840296j,
                -0.062129865145786264+0.15643752202705213j,
                0.012563562045357843+0.1010680231104541j,
                0.016046659386512153-0.008623951102220646j
            ]
        ]
    ]),
    'beam': np.array([
        [
            [ 
                0.9979274427989393+0.06434919509030099j,
                0.00894231045479574+0.022065573193509103j,
                0.02131330222465877+0.0051265326714060675j,
                0.00271048218206843-0.005807650531399505j
            ]
        ]
    ])
  }

  before_holog = str(set_data/"before.split.holog.zarr")
  after_holog = str(set_data/"after.split.holog.zarr")

  before_image = str(set_data/"before.split.image.zarr")
  after_image = str(set_data/"after.split.image.zarr")
  
  holog(
    holog_name=before_holog, 
    padding_factor=50, 
    grid_interpolation_mode='linear',
    chan_average = True,
    scan_average = True,
    overwrite=True,
    phase_fit=True,
    apply_mask=True,
    to_stokes=True,
    parallel=True
  )
  
  assert verify_center_pixels(file=before_image, reference_center_pixels=reference_before_dict)

  assert verify_holog_diagnostics(
    cell_size = np.array([-0.0006442, 0.0006442]),
    grid_size = np.array([31, 31]),
    number_of_digits=7
  )

  holog(
    holog_name=after_holog, 
    padding_factor=50, 
    grid_interpolation_mode='linear',
    chan_average = True,
    scan_average = True,
    overwrite=True,
    phase_fit=True,
    apply_mask=True,
    to_stokes=True,
    parallel=True
  )

  assert verify_center_pixels(file=after_image, reference_center_pixels=reference_after_dict)

  assert verify_holog_diagnostics(
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

  assert verify_panel_shifts(data_dir=str(set_data))
