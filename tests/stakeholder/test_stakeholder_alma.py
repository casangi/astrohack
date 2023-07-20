import pytest

import shutil
import os
import json
import copy
import astrohack

import numpy as np

from astrohack.gdown_utils import gdown_data

from astrohack._utils._tools import _jsonify

from astrohack.extract_pointing import extract_pointing
from astrohack.extract_holog import extract_holog
from astrohack.holog import holog
from astrohack.panel import panel

from astrohack.dio import open_panel

from astrohack.astrohack_client import astrohack_local_client

# Can use this for parallel testing; turning of for now.
# client = astrohack_local_client(cores=2, memory_limit='8GB', log_parms={'log_level':'DEBUG'})

@pytest.fixture(scope='session')
def set_data(tmp_path_factory):
  data_dir = tmp_path_factory.mktemp("data")
    
  # Data files
  astrohack.gdown_utils.download('J1924-2914.ms.calibrated.split.SPW3', folder=str(data_dir), unpack=True)

  # Verification json information
  astrohack.gdown_utils.download(file='extract_holog_verification.json', folder=str(data_dir))
  astrohack.gdown_utils.download(file='holog_numerical_verification.json', folder=str(data_dir))

  return data_dir

def relative_difference(result, expected):  
      return 2*np.abs(result - expected)/(abs(result) + abs(expected))

def verify_panel_positions(
    data_dir="",
    panel_list=['3-11', '5-31', '7-52', '11-62'], 
    reference_position = np.array([-2.16823971, -0.94590908,  0.84834425, 0.76463105]),
    antenna='ant_DV13',
    ddi='ddi_0'
):
    
  M_TO_MILS = 39370.1
    
  panel_mds = open_panel('{data}/alma.split.panel.zarr'.format(data=data_dir))

    
  panel_position = np.mean(panel_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values*M_TO_MILS, axis=1)

  relative_position = relative_difference(panel_position, reference_position)
    
  return np.any(relative_position < 1e-6)

def verify_center_pixels(file, antenna, ddi, reference_center_pixels, tolerance=1e-6):
    from astrohack.dio import open_image
    
    mds = open_image(file)[antenna][ddi]
    
    aperture_shape = mds.APERTURE.values.shape[-2], mds.APERTURE.values.shape[-1]
    beam_shape = mds.BEAM.values.shape[-2], mds.BEAM.values.shape[-1]    
    
    aperture_center_pixels = np.squeeze(mds.APERTURE.values[..., aperture_shape[0]//2, aperture_shape[1]//2])
    beam_center_pixels = np.squeeze(mds.BEAM.values[..., beam_shape[0]//2, beam_shape[1]//2])
    
    aperture_ref = list(map(complex, reference_center_pixels['aperture']))
    beam_ref = list(map(complex, reference_center_pixels['beam']))
    
    for i in range(len(aperture_ref)):

        aperture_check = relative_difference(
            aperture_ref[i].real, 
            aperture_center_pixels[i].real
        ) < tolerance
        
        beam_check = relative_difference(
            beam_ref[i].real, 
            beam_center_pixels[i].real
        ) < tolerance

        real_check = aperture_check and beam_check
                
        aperture_check = relative_difference(
            aperture_ref[i].imag, 
            aperture_center_pixels[i].imag
        ) < tolerance

        beam_check = relative_difference(
            beam_ref[i].imag, 
            beam_center_pixels[i].imag
        ) < tolerance

        imag_check = aperture_check and beam_check
        
        return real_check and imag_check   
                
def verify_holog_obs_dictionary(holog_obs_dict):

    ref_holog_obj = {}
    ref_holog_obj = copy.deepcopy(holog_obs_dict)

    _jsonify(ref_holog_obj)

    with open(".holog_obs_dict.json") as json_file:
        holog_obj = json.load(json_file)
                          
    return  holog_obj == ref_holog_obj

def verify_holog_diagnostics(json_data, truth_json, tolerance=1e-7):
    
    with open(truth_json) as file:
        reference_dict = json.load(file)
        
    cell_size = reference_dict["alma"]['cell_size'][1]
    grid_size = float(reference_dict["alma"]['grid_size'][1])
    
    json_data['cell_size'] = np.abs(float(json_data['cell_size']))
    
    cell_size = np.abs(float(cell_size))

    return (relative_difference(json_data['cell_size'], cell_size) < tolerance) and (relative_difference(np.sqrt(int(json_data['n_pix'])), grid_size) < tolerance)

def test_holography_pipeline(set_data):
    alma_ms = str(set_data/"J1924-2914.ms.calibrated.split.SPW3")
    alma_point = str(set_data/"alma.split.point.zarr")
    alma_holog = str(set_data/"alma.split.holog.zarr")

    with open(str(set_data/"extract_holog_verification.json")) as file:
      holog_obs_dict = json_dict = json.load(file)

    extract_pointing(
      ms_name=alma_ms,
      point_name=alma_point,
      parallel=False,
      overwrite=True
    )

    extract_holog(
      ms_name=alma_ms,
      holog_name=alma_holog,
      point_name=alma_point,
      data_column='DATA',
      parallel=False,
      overwrite=True
    )

    verify_holog_obs_dictionary(holog_obs_dict["alma"])
    
    with open(str(set_data/"holog_numerical_verification.json")) as file:
      reference_dict = json.load(file)

    before_image = str(set_data/"vla.before.split.image.zarr")
    after_image = str(set_data/"vla.after.split.image.zarr")
    
    holog(
      holog_name=alma_holog, 
      padding_factor=50, 
      grid_interpolation_mode="linear",
      chan_average = True,
      scan_average = True,
      overwrite=True,
      phase_fit=True,
      apply_mask=True,
      to_stokes=True,
      parallel=False
    )

    alma_image = str(set_data/"alma.split.image.zarr")

    assert verify_center_pixels(
      file=alma_image, 
      antenna="ant_DV13", 
      ddi="ddi_0", 
      reference_center_pixels=reference_dict["alma"]['pixels'],
      tolerance=1e-6
    )

    alma_panel = panel(
      image_name=alma_image, 
      panel_model='rigid',
      parallel=False,
      overwrite=True
    )

    assert verify_panel_positions(data_dir=str(set_data))
    