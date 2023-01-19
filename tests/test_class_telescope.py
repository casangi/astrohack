import pytest
from astrohack._classes.telescope import Telescope, _find_cfg_file, tel_data_path
import os
import filecmp
import shutil


class TestClassTelescope:
    def test_init(self):
        tel = Telescope('vla')
        assert tel.name == 'VLA'
        assert tel.diam == 25.0

        with pytest.raises(Exception):
            tel = Telescope('xxx')

    def test_read(self):
        tel = Telescope('vla')
        tel.read(tel_data_path+'/vlba.zarr')
        assert tel.name == 'VLBA'
        assert tel.focus == 8.75

        with pytest.raises(FileNotFoundError):
            tel.read('xxx')

    def test_write(self):
        testfile = 'test-tel.zarr'
        tel = Telescope('vla')
        tel.write(testfile)
        assert os.path.exists(testfile)
        assert filecmp.cmp(tel_data_path+'/vlba.zarr/.zattrs', testfile+'/.zattrs') == 0
        shutil.rmtree(testfile)

    def test_ringed_consistency(self):
        tel = Telescope('vla')
        tel.onaxisoptics = False
        with pytest.raises(Exception):
            tel._ringed_consistency()
        tel.nrings = 1000
        with pytest.raises(Exception):
            tel._ringed_consistency()

    def test_general_consistency(self):
        tel = Telescope('vla')
        with pytest.raises(Exception):
            tel._general_consistency()

    def test_find_cfg_file(self):
        filefullpath = _find_cfg_file('vla.zarr', tel_data_path)
        assert filefullpath == tel_data_path+'/vla.zarr'
        with pytest.raises(FileNotFoundError):
            dummy = _find_cfg_file('xxx', './xxx')
