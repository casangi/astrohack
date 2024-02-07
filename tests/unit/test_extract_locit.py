import os
import shutil

import graphviper
from astrohack.extract_locit import extract_locit


class TestExtractLocit():
    cal_table = './data/locit-input-pha.cal'
    locit_name = './data/locit-input-pha.locit.zarr'

    @classmethod
    def setup_class(cls):
        """
            Setup any state specific to the execution of the given test class
            such as fetching test data
        """

        graphviper.utils.data.download(file="locit-input-pha.cal", folder="data")

    @classmethod
    def teardown_class(cls):
        """ teardown any state that was previously setup with a call to setup_class
        such as deleting test data """
        shutil.rmtree("data")

    def teardown_method(self):
        shutil.rmtree(self.locit_name)

    def test_extract_locit_creation(self):
        """
           Create locit file with a cal-table and check it is created correctly.
        """

        # Create locit_mds and check the dictionary structure
        locit_mds = extract_locit(cal_table=self.cal_table, locit_name=self.locit_name)

        expected_keys = ['obs_info', 'ant_info', 'ant_ea01', 'ant_ea02', 'ant_ea04', 'ant_ea05', 'ant_ea06', 'ant_ea07',
                         'ant_ea08', 'ant_ea09', 'ant_ea10', 'ant_ea11', 'ant_ea12', 'ant_ea13', 'ant_ea15', 'ant_ea16',
                         'ant_ea17', 'ant_ea18', 'ant_ea19', 'ant_ea20', 'ant_ea21', 'ant_ea22', 'ant_ea23', 'ant_ea24',
                         'ant_ea25', 'ant_ea26', 'ant_ea27', 'ant_ea28']

        for key in locit_mds.keys():
            assert key in expected_keys

        assert os.path.exists(self.locit_name)

    def test_extract_locit_antenna_select(self):
        """
           Check that only specified antenna is processed.
        """

        locit_mds = extract_locit(
            cal_table=self.cal_table,
            locit_name=self.locit_name,
            ant='ea17'
        )

        # There should only be 1 antenna in the dict named ea17
        assert len(locit_mds.keys()) == 3

        # Check that only the specific antenna is in the keys.
        assert list(locit_mds.keys()) == ['obs_info', 'ant_info', 'ant_ea17']

    def test_extract_locit_ddi(self):
        """
           Check that only specified ddi is processed.
        """

        locit_mds = extract_locit(cal_table=self.cal_table, locit_name=self.locit_name, ddi=0)

        # Check that only the specific ddi is in the keys.
        assert len(locit_mds['ant_ea01'].keys()) == 1
        assert list(locit_mds['ant_ea01'].keys()) == ["ddi_0"]

    def test_extract_locit_overwrite(self):
        """
           Specify the output file should be overwritten; check that it WAS.
        """

        extract_locit(
            cal_table=self.cal_table,
            locit_name=self.locit_name
        )

        # To check this properly we need to not only know an exception was not thrown but that the file is ACTUALLY
        # overwritten. We do this by checking the modification time.
        initial_time = os.path.getctime(self.locit_name)

        extract_locit(
            cal_table=self.cal_table,
            locit_name=self.locit_name,
            overwrite=True
        )

        modified_time = os.path.getctime(self.locit_name)

        assert initial_time != modified_time

    def test_extract_locit_no_overwrite(self):
        """
            Specify the output file should be NOT be overwritten; check that it WAS NOT.
        """
        extract_locit(
            cal_table=self.cal_table,
            locit_name=self.locit_name
        )

        initial_time = os.path.getctime(self.locit_name)

        try:
            extract_locit(
                cal_table=self.cal_table,
                locit_name=self.locit_name,
                overwrite=False
            )

        except FileExistsError:
            pass

        finally:
            modified_time = os.path.getctime(self.locit_name)

            assert initial_time == modified_time
