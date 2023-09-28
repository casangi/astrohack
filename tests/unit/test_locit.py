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
        """
            User story: As a user, I will run locit with a specified locit_name and I will expect the file to be created on disk.
        """

        assert os.path.exists("data/locit-input-pha.locit.zarr")

    def test_locit_ant_id(self):
        """
            As a user, I will run locit with a specified antenna id. I will expect the file to be created on disk containing delays and position solutions only from that antenna id.

            The multiple data structure (mds) objects have a dictionary structure; locit is run to include only the specified antenna, thus if the lists are not equal there is an error.
        """

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            ant_id=["ea25"],
            parallel=False,
            overwrite=True
        )

        assert list(position_mds.keys()) == ['ant_ea25']

    def test_locit_ddi(self):
        """
            As a user, I will run locit with a specified DDI. I will expect the file to be created on disk containing delays and position solutions only from that DDI.

            The multiple data structure (mds) objects have a dictionary structure; locit is run to include only the specified ddi. As there are multiple antenna values
            all of which should include only one ddi, we check all antennae in the list requiring that only one ddi exist for each.
        """

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
        """
            As a user, I will run locit specifying fit_kterm=True and I will expect the file to be created on disk to contain a solution for the kterm.

            In the case that the kterm is not included, it will not be included when the xarray data structure is written. Therefore. calling it will
            throw and error.
        """

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            fit_kterm = True,
            combine_ddis=False,
            parallel=False,
            overwrite=True
        )

        for ant in position_mds.keys():
            for ddi in position_mds[ant].keys():
                position_mds[ant][ddi].koff_fit

    def test_locit_fit_slope(self):
        """
            As a user, I will run locit specifying fit_slope=False and I will expect the file to be created on disk to contain no solution for the delay slope.

            In the case that the slope is not included, it will not be included when the xarray data structure is written. Therefore. calling it will
            throw and error.
        """


        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            fit_slope = True,
            combine_ddis=False,
            parallel=False,
            overwrite=True
        )

        for ant in position_mds.keys():
            for ddi in position_mds[ant].keys():
                position_mds[ant]["ddi_0"].slope_fit

    def test_locit_elevation_limit(self):
        """
            As a user, I will run locit specifying elevation_limit=90 and I will expect locit to fail because there is no available data.

            And exception is thrown when elevation_limit is set to 90 degrees. Catch the exception as it is what is expected and check the 
            failed variable.
        """
        
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
        """
            As a user, I will run locit specifying polarization='R' and I will expect the file to be created on disk to contain only delays for the R polarization and position solutions derived only with the R polarization.

            As the polarization states for each antenna are written to the xds we check them for each key and make sure they have only the desired polarization.
        """
        
        position_mds = locit(
                locit_name="data/locit-input-pha.locit.zarr",
                position_name="data/locit-input-pha.position.zarr",
                polarization="R",
                parallel=False,
                overwrite=True
        )

        for ant in position_mds.keys():
            assert position_mds[ant].polarization == "R"

    def test_locit_combine_ddis(self):
        """
           As a user I will run locit specifying combine_ddis=False and I will expect the file to be created on disk to contain delays and position solutions for all DDIs.

           Because we have used teh combine_ddis argument here the key order geoes from antenna->ddi to only antenna and the keys list becomes the meta data variable names for the xds file.
        """

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            combine_ddis=True,
            parallel=False,
            overwrite=True
        )

        for key in position_mds.keys():
            assert list(position_mds[key].keys()) == ['DECLINATION', 'DELAYS', 'ELEVATION', 'HOUR_ANGLE', 'LST', 'MODEL']

    def test_locit_overwrite(self):
        """
             As a user I will run locit specifying combine_ddis=False and I will expect the file to be created on disk to contain delays and position solutions for all DDIs.

             Simply check that the file modification time has been changed after being overwritten. This provides a good confirmation things worked according to plan.
        """
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
        """
             As a user, I will run locit specifying overwrite=False and I will expect the file to be created on disk to be not overwritten.

             If overwtie=False and error is thrown when the same file name is used. We catch the error as it is what we expect and don't want the test to 
             exit early. We can then assert that the modification time has not changed as a redundant check on the parameter.
        """
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

        


    