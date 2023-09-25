import pytest

import os
import json
import shutil
import astrohack

import numpy as np

from astrohack.locit import locit
from astrohack.dio import open_locit
from astrohack.dio import open_position
from astrohack.extract_locit import extract_locit

def relative_difference(result, expected):
        return 2*np.abs(result - expected)/(abs(result) + abs(expected))

class TestLocit():
    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given test class
        such as fetching test data """
        astrohack.data.datasets.download('locit-input-pha.cal', folder='data')

        locit_mds = extract_locit(
            cal_table="data/locit-input-pha.cal",
            locit_name="data/locit-input-pha.locit.zarr",
            overwrite=True
        )

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            elevation_limit=10.0,
            polarization='both',
            fit_engine='scipy',
            parallel=False,
            overwrite=True
        )

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
    
    def test_locit_name(self):

        assert os.path.exists("data/locit-input-pha.locit.zarr")

    def test_locit_ant_id(self):

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            ant_id=["ea25"],
            parallel=False,
            overwrite=True
        )

        assert list(position_mds.keys()) == ['ant_ea25']

    def test_locit_ddi(self):

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            ddi=[0],
            combine_ddis=False,
            parallel=False,
            overwrite=True
        )

        for ant in position_mds.keys():
            for ddi in position_mds[ant].keys():
                assert ddi == "ddi_0"

    def test_locit_fit_kterm(self):

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            fit_kterm = True,
            combine_ddis=False,
            parallel=False,
            overwrite=True
        )

        position_mds["ant_ea25"]["ddi_0"].koff_fit

    def test_locit_fit_slope(self):


        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            fit_slope = True,
            combine_ddis=False,
            parallel=False,
            overwrite=True
        )

        position_mds["ant_ea25"]["ddi_0"].slope_fit

    def test_locit_elevation_limit(self):
        
        failed = False

        try:
            position_mds = locit(
                locit_name="data/locit-input-pha.locit.zarr",
                position_name="data/locit-input-pha.position.zarr",
                elevation_limit=90.0,
                parallel=False,
                overwrite=True
            )

            failed = True

        except KeyError:
            pass

        finally:
            assert failed == False

    def test_locit_polarization(self):
        
        position_mds = locit(
                locit_name="data/locit-input-pha.locit.zarr",
                position_name="data/locit-input-pha.position.zarr",
                polarization="R",
                parallel=False,
                overwrite=True
        )

        assert position_mds["ant_ea25"].polarization == "R"

    def test_locit_overwrite(self):
        initial_time = os.path.getctime("data/locit-input-pha.position.zarr")
        
        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            parallel=False,
            overwrite=True
        )     

        modified_time = os.path.getctime("data/locit-input-pha.position.zarr")

        assert initial_time != modified_time

    def test_locit_not_overwrite(self):
        initial_time = os.path.getctime("data/locit-input-pha.position.zarr")
        
        try:
            position_mds = locit(
                locit_name="data/locit-input-pha.locit.zarr",
                position_name="data/locit-input-pha.position.zarr",
                parallel=False,
                overwrite=False
            )

        except FileExistsError:
            pass

        finally:
            modified_time = os.path.getctime("data/locit-input-pha.position.zarr")

            assert initial_time == modified_time

        


    