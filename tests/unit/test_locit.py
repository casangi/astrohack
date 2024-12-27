import os

import shutil
import toolviper

import numpy as np

from astrohack.locit import locit
from astrohack.dio import open_locit
from astrohack.dio import open_position
from astrohack.extract_locit import extract_locit


def relative_difference(result, expected):
    return 2 * np.abs(result - expected) / (abs(result) + abs(expected))


class TestLocit():
    @classmethod
    def setup_class(cls):
        """
            Setup any state specific to the execution of the given test class
            such as fetching test data
        """
        toolviper.utils.data.download('locit-input-pha.cal', folder='data')

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
            Run locit with a specified locit_name and expect a file to be created on disk.
        """

        assert os.path.exists("data/locit-input-pha.locit.zarr")

    def test_locit_ant_id(self):
        """
            Run locit with an antenna id and create a file on disk containing delays and position solutions only \
            from that antenna id.
        """

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            ant=["ea25"],
            parallel=False,
            overwrite=True
        )

        assert list(position_mds.keys()) == ['ant_ea25']

    def test_locit_ddi(self):
        """
            Run locit with a specified DDI and create a file on disk containing delays and position solutions \
            only from that DDI.
        """

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            ddi=[0],
            combine_ddis="no",
            parallel=False,
            overwrite=True
        )

        for ant in position_mds.keys():
            for ddi in position_mds[ant].keys():
                assert ddi == "ddi_0"

    def test_locit_fit_kterm(self):
        """
            Run locit with fit_kterm=True and expect a file to be created on disk containing a solution for the kterm.
        """

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            fit_kterm=True,
            combine_ddis="no",
            parallel=False,
            overwrite=True
        )

        for ant in position_mds.keys():
            for ddi in position_mds[ant].keys():
                # This is a bit redundant since calling koff_fit when it doesn't exist throws and exception, but it
                # makes what is happening more obvious to others.
                try:
                    position_mds[ant][ddi].koff_fit
                except Exception:
                    raise Exception

    def test_locit_fit_rate(self):
        """
            Run locit with fit_rate=False and check that the file created on disk contains no solution for the \
            delay rate.
        """

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            fit_delay_rate=True,
            combine_ddis='no',
            parallel=False,
            overwrite=True
        )

        for ant in position_mds.keys():
            for ddi in position_mds[ant].keys():
                # This is a bit redundant since calling rate_fit when it doesn't exist throws and exception, but it
                # makes what is happening more obvious to others.
                try:
                    position_mds[ant]["ddi_0"].rate_fit

                except Exception:
                    raise Exception

    def test_locit_elevation_limit(self):
        """
            Run locit with elevation_limit=90 and expect locit to fail because there is no available data.
        """

        assert locit(
                locit_name="data/locit-input-pha.locit.zarr",
                position_name="data/locit-input-pha.position.zarr",
                elevation_limit=90.0,
                parallel=False,
                overwrite=True
            ) is None

    def test_locit_polarization(self):
        """
            Run locit with polarization='R' and check that the file created on disk contains only delays for R.
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
          Run locit with combine_ddis=False and check that the file created on disk contains delays and position \
          solutions for all DDIs.
        """

        position_mds = locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            combine_ddis='simple',
            parallel=False,
            overwrite=True
        )

        for key in position_mds.keys():
            assert list(position_mds[key].keys()) == [
                'DECLINATION',
                'DELAYS',
                'ELEVATION',
                'HOUR_ANGLE',
                'LST',
                'MODEL'
            ]

    def test_locit_overwrite(self):
        """
            Simply check that the file modification time has been changed after being overwritten.
        """
        initial_time = os.path.getctime("data/locit-input-pha.position.zarr")

        locit(
            locit_name="data/locit-input-pha.locit.zarr",
            position_name="data/locit-input-pha.position.zarr",
            parallel=False,
            overwrite=True
        )

        modified_time = os.path.getctime("data/locit-input-pha.position.zarr")

        assert initial_time != modified_time

    def test_locit_not_overwrite(self):
        """
            Run locit with overwrite=False and expect the file created on disk to not be overwritten.
        """
        initial_time = os.path.getctime("data/locit-input-pha.position.zarr")

        try:
            locit(
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
