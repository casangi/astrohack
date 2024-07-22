import os
import shutil
import graphviper

from astrohack.extract_pointing import extract_pointing


class TestExtractPointing:
    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class"""
        cls.datafolder = "point_data"
        cls.ms = "ea25_cal_small_before_fixed.split.ms"
        
        graphviper.utils.data.download(file=cls.ms, folder=cls.datafolder)
        
        cls.ms_name = os.path.join(cls.datafolder, cls.ms)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class"""
        shutil.rmtree(cls.datafolder)

    def setup_method(self):
        """setup any state specific to a method of the given class"""
        pass

    def teardown_method(self):
        """teardown any state that was previously setup for all methods of the given class"""
        pass

    def test_extract_pointing_default(self):
        """Test extract_pointing with default parameters"""
        point_obj = extract_pointing(ms_name=self.ms_name)

        # Check the keys of the returned dictionary
        expected_keys = ["point_meta_ds", "ant_ea04", "ant_ea06", "ant_ea25"]

        for key in point_obj.keys():
            assert key in expected_keys

    def test_extract_pointing_point_name(self):
        """Test extract_pointing and save to given point name"""
        point_name = os.path.join(self.datafolder, "test_user_point_name.zarr")
        point_obj = extract_pointing(ms_name=self.ms_name, point_name=point_name)

        assert os.path.exists(point_name)

        # Check that the returned dictionary contains the given point_name
        assert point_obj.file == point_name

    def test_extract_pointing_overwrite_true(self):
        """Test extract_pointing and overwrite existing pointing file"""
        point_name = os.path.join(self.datafolder, "test_user_overwrite.zarr")
        extract_pointing(ms_name=self.ms_name, point_name=point_name)
        initial_time = os.path.getctime(point_name)

        extract_pointing(ms_name=self.ms_name, point_name=point_name, overwrite=True)
        modified_time = os.path.getctime(point_name)

        assert initial_time != modified_time

    def test_extract_pointing_invalid_ms_name(self):
        """Test extract_pointing and check that invalid_ms does not create point file"""
        # Exceptions are not raised by the code, therefore doing a silly check here
        try:
            # Changed this because it does fail if you give a non-existent ms. So we will check that.
            extract_pointing(ms_name="invalid_name.ms")

        except Exception:
            return

        assert False
