import pytest

import os
import json
import shutil
import astrohack

import numpy as np

from astrohack.holog import holog
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing
from astrohack.extract_holog import generate_holog_obs_dict

def relative_difference(result, expected):
        return 2*np.abs(result - expected)/(abs(result) + abs(expected))

class TestHolog():
    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given test class
        such as fetching test data """
        astrohack.data.datasets.download(file="ea25_cal_small_after_fixed.split.ms", folder="data/", unpack=True)
        
        astrohack.data.datasets.download(file='extract_holog_verification.json')
        astrohack.data.datasets.download(file='holog_numerical_verification.json')

        extract_pointing(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            overwrite=True,
            parallel=False
        )

        # Extract holography data using holog_obd_dict
        holog_mds = extract_holog(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True
        )

        extract_holog(
            ms_name="data/ea25_cal_small_after_fixed.split.ms",
            point_name="data/ea25_cal_small_after_fixed.split.point.zarr",
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            data_column='CORRECTED_DATA',
            parallel=False,
            overwrite=True
        )

        holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_after_fixed.split.image.zarr/.image_attr') as json_attr:
            cls.json_file = json.load(json_attr)

    @classmethod
    def teardown_class(cls):
        """ teardown any state that was previously setup with a call to setup_class
        such as deleting test data """
        shutil.rmtree("data")

    def setup_method(self):
        """ setup any state specific to all methods of the given class """
        pass

    def teardown_method(self):
        """ teardown any state that was previously setup for all methods of the given class """
        pass


    def test_holog_grid_cell_size(self):

        tolerance = 2.e-5

        with open("holog_numerical_verification.json") as file:
            reference_dict = json.load(file)

        with open('data/ea25_cal_small_after_fixed.split.image.zarr/.image_attr') as attr_file:
            image_attr = json.load(attr_file)
    
        for i, _ in enumerate(image_attr['cell_size']):
            assert relative_difference(
                image_attr['cell_size'][i], 
                reference_dict["vla"]['cell_size'][i]
            ) < tolerance

            assert relative_difference(
                image_attr['grid_size'][i], 
                reference_dict["vla"]['grid_size'][i]
            ) < tolerance

    
    def test_holog_image_name(self):

        assert os.path.exists('data/ea25_cal_small_after_fixed.split.image.zarr')

    def test_holog_ant_id(self):

        image_mds = holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            grid_interpolation_mode='linear',
            chan_average=True,
            scan_average=True,
            ant_id=['ea25'],
            overwrite=True,
            phase_fit=True,
            apply_mask=True,
            to_stokes=True,
            parallel=False
        )

        assert list(image_mds.keys()) == ['ant_ea25']

    def test_holog_ddi(self):

        image_mds = holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            padding_factor=50, 
            grid_interpolation_mode='linear',
            chan_average=True,
            scan_average=True,
            overwrite=True,
            ddi=[0],
            phase_fit=True,
            apply_mask=True,
            to_stokes=True,
            parallel=False
        )

        for ant in image_mds.keys():
            for ddi in image_mds[ant].keys():
                assert ddi == "ddi_0"

    

    def test_holog_padding_factor(self):

        image_mds = holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            padding_factor=50, 
            grid_interpolation_mode='linear',
            chan_average=True,
            scan_average=True,
            overwrite=True,
            phase_fit=True,
            apply_mask=True,
            to_stokes=True,
            parallel=False
        )

        for ant in image_mds.keys():
            for ddi in image_mds[ant].keys():
                assert image_mds[ant][ddi].APERTURE.shape == (1, 1, 4, 529, 529)

    def test_holog_chan_average(self):
        image_mds = holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            chan_average=True,
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_after_fixed.split.image.zarr/.image_attr') as json_attr:
            json_file = json.load(json_attr)     
        
        assert json_file['chan_average'] == True

    def test_holog_scan_average(self):
        image_mds = holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            scan_average=False,
            overwrite=True,
            parallel=False
        )     

        with open('data/ea25_cal_small_after_fixed.split.image.zarr/.image_attr') as json_attr:
            json_file = json.load(json_attr)

        assert json_file['scan_average'] == False

    def test_holog_grid_interpolation(self):
        image_mds = holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            grid_interpolation_mode='nearest',
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_after_fixed.split.image.zarr/.image_attr') as json_attr:
            json_file = json.load(json_attr)        

        assert json_file['grid_interpolation_mode'] == 'nearest'

    def test_holog_chan_tolerance(self):
        image_mds = holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            chan_tolerance_factor=0.0049,
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_after_fixed.split.image.zarr/.image_attr') as json_attr:
            json_file = json.load(json_attr)     

        assert json_file['chan_tolerance_factor'] == 0.0049

    def test_holog_to_stokes(self):
        image_mds = holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            to_stokes=False,
            overwrite=True,
            parallel=False
        )

        with open('data/ea25_cal_small_after_fixed.split.image.zarr/.image_attr') as json_attr:
            json_file = json.load(json_attr)     

        assert json_file['to_stokes'] == False

    def test_holog_overwrite(self):
        initial_time = os.path.getctime('data/ea25_cal_small_after_fixed.split.image.zarr')
        
        image_mds = holog(
            holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
            image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
            overwrite=True,
            parallel=False
        )

        modified_time = os.path.getctime('data/ea25_cal_small_after_fixed.split.image.zarr')

        assert initial_time != modified_time

    def test_holog_not_overwrite(self):
        initial_time = os.path.getctime('data/ea25_cal_small_after_fixed.split.image.zarr')
        
        try:
            holog(
                holog_name='data/ea25_cal_small_after_fixed.split.holog.zarr',
                image_name='data/ea25_cal_small_after_fixed.split.image.zarr', 
                overwrite=False,
                parallel=False
            )

        except FileExistsError:
            pass

        finally:
            modified_time = os.path.getctime('data/ea25_cal_small_after_fixed.split.image.zarr')

            assert initial_time == modified_time

        


    