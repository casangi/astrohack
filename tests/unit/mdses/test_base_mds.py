import copy
import pathlib
import shutil
from collections.abc import ItemsView, KeysView, ValuesView

import pytest
import xarray
from toolviper.utils import data

from astrohack.io.base_mds import AstrohackBaseFile
from astrohack.utils.verification_tools import (
    add_data_folder_to_names_in_class,
    analyse_summary,
    are_dicts_close,
    are_lists_equal,
    create_origin_dict,
)


class TestBaseMds:
    """Here we use a beamcut file as an example of a base file"""

    data_dir = "base_mds_data"

    silly_name = "Anything"
    pos_mds_name = "locit-reference.position.zarr"
    output_file_name = "base_mds_test.base.zarr"

    exp_n_ant_in_mds = 26
    get_ant_key = "ant_ea15"
    set_ant_key = "ant_na29"

    ref_input_pars = {
        "adenine": "a",
        "guanine": "g",
        "thymine": "t",
        "cytosine": "c",
    }

    @classmethod
    def setup_class(cls):
        """setup any state specific to the execution of the given test class
        such as fetching test data"""
        data.download(file=cls.pos_mds_name, folder=cls.data_dir)

        add_data_folder_to_names_in_class(cls)

    @classmethod
    def teardown_class(cls):
        """teardown any state that was previously setup with a call to setup_class
        such as deleting test data"""
        shutil.rmtree(cls.data_dir, ignore_errors=True)
        return

    def test_init_and_open_base_mds(self):
        base_mds = AstrohackBaseFile(self.silly_name)

        assert base_mds.filename == self.silly_name, (
            "base mds file name should be the same as the one given as argument to __init__"
        )

        assert not base_mds.is_open, "base mds file should not be opened yet"

        with pytest.raises(FileNotFoundError):
            successful_open = base_mds.open()

        successful_open = base_mds.open(self.pos_mds_name)
        assert successful_open, (
            "Opening base_mds should succeed now that the correct file name is given"
        )

        assert base_mds.is_open, (
            "is_open property needs to return True now that the file has been opened"
        )

        assert base_mds.filename == self.pos_mds_name, (
            ".file attribute should now be set to the name of the given file."
        )

        return

    def test_base_mds_keys_getitem_and_setitem(self):
        base_mds = AstrohackBaseFile(self.pos_mds_name)
        base_mds.open()

        old_xdt_keys = base_mds.keys()
        assert isinstance(old_xdt_keys, KeysView), (
            "Keys method should return a dict_keys object"
        )

        n_old_keys = len(old_xdt_keys)
        assert n_old_keys == self.exp_n_ant_in_mds, (
            f"File should have {self.exp_n_ant_in_mds}, but has {n_old_keys} antenna subtrees"
        )

        get_ant_subtree = base_mds[self.get_ant_key]
        assert isinstance(get_ant_subtree, xarray.DataTree)

        base_mds[self.set_ant_key] = get_ant_subtree
        new_xdt_keys = base_mds.keys()
        n_new_keys = len(new_xdt_keys)
        assert n_new_keys == n_old_keys + 1, (
            f"File should now contain {n_old_keys + 1} antenna subtrees"
        )

        assert self.set_ant_key in new_xdt_keys, (
            "New antenna subtree should appear amongst keys"
        )

        return

    def test_base_mds_write_items_and_values_and_create_from_input_parameters(self):
        base_mds = AstrohackBaseFile(self.pos_mds_name)
        base_mds.open()

        old_xdt_values = base_mds.values()
        assert isinstance(old_xdt_values, ValuesView), (
            "Values method should return a ValuesView object"
        )
        assert len(old_xdt_values) == self.exp_n_ant_in_mds, (
            f"File should contain {self.exp_n_ant_in_mds} antenna subtrees"
        )

        any_ant_subtree = list(old_xdt_values)[0]
        assert isinstance(any_ant_subtree, xarray.DataTree)

        old_xdt_items = base_mds.items()
        assert isinstance(old_xdt_items, ItemsView), (
            "Items method should return a ItemsView object"
        )

        new_base_mds = AstrohackBaseFile.create_from_input_parameters(
            self.output_file_name, self.ref_input_pars
        )

        assert isinstance(new_base_mds.root, xarray.DataTree), (
            "Root attribute must be a DataTree"
        )

        assert len(new_base_mds.root.children) == 0, (
            "Root data tree must not contain any children"
        )

        assert new_base_mds.root.attrs["input_parameters"] == self.ref_input_pars, (
            "input_parameters should be equal to the given dictionary"
        )

        assert are_dicts_close(
            new_base_mds.root.attrs["origin_info"],
            create_origin_dict(
                "test_base_mds_write_items_and_values_and_create_from_input_parameters"
            ),
            ignored_keys=["creation_time"],
        ), "Origin information should be equal to reference"

        # Add a child
        new_base_mds[self.set_ant_key] = any_ant_subtree
        new_base_mds.write()

        # Delete old objs for better testing
        del base_mds, new_base_mds

        assert pathlib.Path(self.output_file_name).is_dir(), (
            f"Write method should create a directory named {self.output_file_name} containing the DataTree"
        )

        new_base_mds = AstrohackBaseFile(self.output_file_name)
        new_base_mds.open()
        assert new_base_mds.root.attrs["input_parameters"] == self.ref_input_pars, (
            "input_parameters from file on disk should be equal to the given dictionary"
        )

        assert len(new_base_mds.keys()) == 1, (
            "Disk read from file should contain one single key"
        )

        assert new_base_mds[self.set_ant_key] == any_ant_subtree, (
            "Antenna subtree in writen file should be equal to the one given to it"
        )

    def test_incremental_write_and_closeness(self):
        shutil.rmtree(self.output_file_name, ignore_errors=True)
        base_mds = AstrohackBaseFile(self.pos_mds_name)
        base_mds.open()

        new_mds = AstrohackBaseFile.create_from_input_parameters(
            self.output_file_name, self.ref_input_pars
        )
        exp_n_keys = 0
        assert len(new_mds.keys()) == exp_n_keys, (
            f"new_mds should have {exp_n_keys} keys at this moment."
        )

        assert not pathlib.Path(self.output_file_name).is_dir(), (
            f"{self.output_file_name} should not exist on disk at this point"
        )

        new_mds.write(mode="w")
        assert pathlib.Path(self.output_file_name).is_dir(), (
            f"{self.output_file_name} should exist on disk at this point"
        )

        ant_key_list = []
        # Test adding datasets
        for ant_key, ant_xdt in base_mds.items():
            xds = ant_xdt.ds
            new_mds.add_node(xds, [ant_key])
            ant_key_list.append(ant_key)
            data_path = pathlib.Path(f"{self.output_file_name}/{ant_key}")
            assert data_path.is_dir(), f"{data_path} should exist on disk at this point"

        assert len(new_mds.keys()) == 0, "Key list should not have grown at this point"
        new_mds.consolidate(["ant"])

        this_ant_key_list = list(new_mds.keys())
        assert are_lists_equal(ant_key_list, this_ant_key_list)

        alternative_name = f"{self.data_dir}/by_xdt.base.zarr"
        by_xdt_mds = AstrohackBaseFile.create_from_input_parameters(
            alternative_name, self.ref_input_pars
        )
        by_xdt_mds.write(mode="w")
        for ant_key, ant_xdt in base_mds.items():
            by_xdt_mds.add_node(ant_xdt.copy(), [ant_key])
            data_path = pathlib.Path(f"{self.output_file_name}/{ant_key}")
            assert data_path.is_dir(), f"{data_path} should exist on disk at this point"
        assert len(by_xdt_mds.keys()) == 0, (
            "Key list should not have grown at this point"
        )
        by_xdt_mds.consolidate(["ant"])

        this_ant_key_list = list(by_xdt_mds.keys())
        assert are_lists_equal(ant_key_list, this_ant_key_list)

        assert by_xdt_mds.is_close_to(new_mds), (
            "MDSes created using datatrees and datasets should be equal"
        )

        assert new_mds.is_close_to(by_xdt_mds), (
            "MDSes created using datasets and datatrees should be equal"
        )

        assert new_mds.is_close_to(base_mds), "New mds should be close to base mds"

        new_mds.root.attrs["random"] = 42
        assert not new_mds.is_close_to(base_mds), (
            "New mds should not be close to base mds"
        )

    def test_summary(self):
        base_mds = AstrohackBaseFile(self.pos_mds_name)
        base_mds.open()
        any_ant_tree = base_mds[self.get_ant_key]

        new_mds = AstrohackBaseFile.create_from_input_parameters(
            self.output_file_name, self.ref_input_pars
        )
        new_mds[self.get_ant_key] = any_ant_tree
        new_ant_tree = copy.deepcopy(any_ant_tree)
        new_ant_tree.name = self.set_ant_key
        new_mds[self.set_ant_key] = new_ant_tree

        analyse_summary(
            new_mds,
            self.output_file_name,
            self.ref_input_pars,
            [self.set_ant_key, self.get_ant_key],
        )
        return
