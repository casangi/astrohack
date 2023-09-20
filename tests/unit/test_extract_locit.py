import pytest

import os
import json
import shutil
import astrohack

from astrohack.extract_locit import extract_locit

class TestExtractHolog():
    cal_table = './data/locit-input-pha.cal'
    locit_name = './data/locit-input-pha.locit.zarr'
    #position_name = './data/locit-input-pha.position.zarr'

    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given test class
        such as fetching test data """
        astrohack.data.datasets.download(file="locit-input-pha.cal", folder="data")
        #astrohack.data.datasets.download(file="locit-input-pha.locit.zarr'", folder="data")
        #astrohack.data.datasets.download(file="locit-input-pha.position.zarr", folder="data")
        
    @classmethod
    def teardown_class(cls):
        """ teardown any state that was previously setup with a call to setup_class
        such as deleting test data """
        shutil.rmtree("data")
        
    def teardown_method(self):
        shutil.rmtree(self.locit_name)

    def test_extract_locit_simple(self):
    
        locit_mds = extract_locit(self.cal_table, locit_name=self.locit_name)
        
        expected_keys = ['obs_info', 'ant_info', 'ant_ea01', 'ant_ea02', 'ant_ea04', 'ant_ea05', 'ant_ea06', 'ant_ea07', 'ant_ea08', 'ant_ea09', 'ant_ea10', 'ant_ea11', 'ant_ea12', 'ant_ea13', 'ant_ea15', 'ant_ea16', 'ant_ea17', 'ant_ea18', 'ant_ea19', 'ant_ea20', 'ant_ea21', 'ant_ea22', 'ant_ea23', 'ant_ea24', 'ant_ea25', 'ant_ea26', 'ant_ea27', 'ant_ea28']
        
        for key in locit_mds.keys():
            assert key in expected_keys
        
    def test_extract_locit_antenna_select(self):
    
        locit_mds = extract_locit(self.cal_table, locit_name=self.locit_name, ant_id='ea17')
        
        assert len(locit_mds.keys()) == 3
        assert 'ant_ea17' in locit_mds.keys()
    
    def test_extract_locit_ddi(self):
    
        locit_mds = extract_locit(self.cal_table, locit_name=self.locit_name, ddi=0)
        
        assert len(locit_mds['ant_ea01'].keys()) == 1
        assert 'ddi_0' in locit_mds['ant_ea01'].keys()
        
    def test_extract_locit_overwrite(self):
    
        locit_mds = extract_locit(self.cal_table, locit_name=self.locit_name)
        
        try:
            locit_mds = extract_locit(self.cal_table, locit_name=self.locit_name, overwrite=True)
        except:
            assert False, "Failed to overwrite"
            
    def test_extract_locit_no_overwrite(self):
    
        locit_mds = extract_locit(self.cal_table, locit_name=self.locit_name, overwrite=False)
        
        try:
            locit_mds = extract_locit(self.cal_table, locit_name=self.locit_name, overwrite=False)
        except:
            pass
