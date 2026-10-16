import glob
import pathlib

import toolviper.utils.logger as logger
import xarray as xr
import zarr

from astrohack.utils.file import add_caller_and_version_to_dict
from astrohack.utils.text import (
    get_data_content_string,
    get_method_list_string,
    get_property_string,
    get_summary_header,
    lnbr,
)
from astrohack.utils.verification_tools import are_data_trees_close, are_dicts_close


class AstrohackBaseFile:
    """Base Data class for astrohack.

    Data within an object of this class can be selected for further inspection, plotted or produce a report
    """

    def __init__(self, file: str):
        """Initialize an AstrohackBaseFile object.

        :param file: File to be linked to this object
        :type file: str

        :return: AstrohackBaseFile object
        :rtype: AstrohackBaseFile
        """
        self.filename = file
        self._file_is_open = False
        self.root = None

    def __getitem__(self, key: str) -> xr.DataTree:
        """
        get item implementation that gets the xdtree at key.

        :param key: Key for which to fetch a subtree
        :type key: str

        :return: corresponding subtree
        :rtype: xr.DataTree
        """
        return self.root[key]

    def __setitem__(self, key: str, subtree: xr.DataTree) -> None:
        """
        Set item implementation that sets the xdtree at key.

        :param key: Key for which to set a subtree
        :type key: str

        :param subtree: Subtree to attach at key
        :type subtree: xr.DataTree

        :return: None
        :rtype: NoneType
        """
        self.root[key] = subtree
        return

    def __eq__(self, other: object) -> bool:
        """
        Compare two AstrohackBaseFile objects, ignoring input_parameters and origin_info to check if their data and \
        attributes are equal

        :param other: Second AstrohackBaseFile object
        :type other: AstrohackBaseFile

        :return: equality result
        :rtype: bool
        """
        if not isinstance(other, AstrohackBaseFile):
            return NotImplemented

        excluded_keys = ["input_parameters", "origin_info"]
        equality = True
        for key, item in self.root.attrs.items():
            if key in excluded_keys:
                continue
            equality = equality and (item == other.root.attrs[key])

        if not equality:
            return False

        for key, sub_tree in self.root.items():
            equality = equality and sub_tree.identical(other[key])
        return equality

    @property
    def is_open(self) -> bool:
        """
        Check whether the object has opened the corresponding hack file.

        :return: True if open, else False.
        :rtype: bool
        """
        return self._file_is_open

    def keys(self, *args, **kwargs):
        """
        Get children keys

        :param args: args to deliver to dict.keys() method
        :type args: list

        :param kwargs: Dict of keyword args to deliver to dict.keys() method
        :type kwargs: dict

        :return: dict keys iterable
        :rtype: dict_keys
        """
        return self.root.children.keys(*args, **kwargs)

    def items(self, *args, **kwargs):
        """
        Get children items

        :param args: args to deliver to dict.items() method
        :type args: list

        :param kwargs: Dict of keyword args to deliver to dict.items() method
        :type kwargs: dict

        :return: dict items iterable
        :rtype: dict_items
        """
        return self.root.children.items(*args, **kwargs)

    def values(self, *args, **kwargs):
        """
        Get children values

        :param args: args to deliver to dict.values() method
        :type args: list

        :param kwargs: Dict of keyword args to deliver to dict.values() method
        :type kwargs: dict

        :return: dict values iterable
        :rtype: dict_values
        """
        return self.root.children.values(*args, **kwargs)

    def open(self, file: str | None = None) -> bool:
        """
        Open Base file.

        :param file: File to be opened, if None defaults to the previously defined file
        :type file: str, optional

        :return: True if file is properly opened, else returns False
        :rtype: bool
        """

        if file is None:
            file = self.filename

        try:
            # Chunks='auto' means lazy dask loading with automatic choice of chunk size
            # chunks=None is direct opening.
            self.root = xr.open_datatree(file, engine="zarr", chunks="auto")

            self._file_is_open = True
            self.filename = file

        except FileNotFoundError:
            self._file_is_open = False
            msg = f"File not found at {self.filename}"
            raise FileNotFoundError(msg)

        except Exception as error:
            self._file_is_open = False
            msg = f"There was an exception opening the file: {error}"
            logger.error(msg)
            raise RuntimeError(msg)

        return self._file_is_open

    def write(self, mode="w"):
        """
        Write mds to disk by saving the data tree to a file

        :param mode: File mode
        :type mode: str
        """
        self.root.to_zarr(self.filename, mode=mode, consolidated=True)

    def summary(self) -> None:
        """
        Prints summary of this Astrohack File object, with available data, attributes and methods

        :return: None
        :rtype: NoneType
        """
        outstr = get_summary_header(self.filename)
        outstr += get_property_string(self.root.attrs)
        outstr += get_method_list_string(self)
        outstr += get_data_content_string(self.root)
        print(outstr)

    @classmethod
    def create_from_input_parameters(cls, file_name: str, input_parameters: dict):
        """
        Create an AstrohackBaseFile object from a filename and initializes xdtree root attributes.

        :param file_name: Name of the file in disk to be created
        :type file_name: str

        :param input_parameters: Input parameters for the calling function to be stored in root attributes.
        :type input_parameters: dict

        :return: Initialized AstrohackBaseFile object
        :rtype: AstrohackBaseFile
        """
        data_obj = cls(file_name)
        data_obj.root = xr.DataTree(name="root")
        add_caller_and_version_to_dict(data_obj.root.attrs, direct_call=False)
        data_obj.root.attrs["input_parameters"] = input_parameters
        return data_obj

    def add_node(
        self,
        xarray_data: xr.Dataset | xr.DataTree,
        key_list: list[str] | tuple[str],
    ):
        """
        Add a node to the data tree file structure, however this node is not yet consolidated into the data tree \
        structure, consolidate must be called to integrate all nodes writen by add_node onto the tree structure.

        :param xarray_data: XDS or XDT to be included into the data tree structure.
        :type xarray_data: xr.DataSet, xr.DataTree

        :param key_list: list of data identifying keys to determine where to add node
        :type key_list: list, tuple

        :return: None
        :rtype: NoneType
        """
        assert isinstance(key_list, (list, tuple))
        final_key = key_list[-1]
        new_node_path = "/".join([self.filename, *key_list])

        if isinstance(xarray_data, xr.Dataset):
            xr.DataTree(dataset=xarray_data, name=final_key).to_zarr(
                new_node_path, mode="w"
            )
        elif isinstance(xarray_data, xr.DataTree):
            xarray_data.name = final_key
            xarray_data.to_zarr(new_node_path, mode="w")
        else:
            raise NotImplementedError(
                f"Don't know how to handle nodes of type {type(xarray_data)}"
            )

    def __repr__(self):
        """
        Simple printing function to glance at the datatree inside
        :return: Print contents
        """
        outstr = f"<{type(self).__name__}>{lnbr}"
        outstr += f"File on disk: {self.filename}{lnbr}"
        outstr += f"Data tree: {lnbr}{self.root.__repr__()}"
        return outstr

    def is_close_to(self, other_mds, tol=1e-6):
        """
        Tests if self and other_mds are close to each other.

        :param other_mds: Another mds
        :type other_mds: AstrohackBaseFile

        :param tol: Tolerance
        :type tol: float

        :return: True if Mdses are close up to tolerance
        :rtype: bool
        """
        if not isinstance(other_mds, AstrohackBaseFile):
            return NotImplemented

        is_close = are_dicts_close(
            self.root.attrs,
            other_mds.root.attrs,
            tol,
            ignored_keys=["input_parameters", "origin_info"],
        )

        if is_close:
            for key, self_subtree in self.items():
                if key not in other_mds.keys():
                    return False
                else:
                    is_close = is_close and are_data_trees_close(
                        self_subtree, other_mds[key], tol=tol
                    )
        else:
            return False

        return is_close

    def consolidate(self, key_order: list[str]):
        """
        Traverse own file structure on disk consolidating metadata to create a unified data tree entity.

        :param key_order: Order in which keys appear in file structure, ordered by depth.
        :type key_order: list

        :return: None
        :rtype: NoneType
        """

        # This function would be more robust if it were recursive, then a key order list probably wouldn't even be
        # necessary.
        mds_path = self.filename
        logger.info(f"Consolidating {mds_path}...")

        # Hardcoded number of levels of extract_holog products as they are 3 leveled but execution is 2 leveled.
        if self.root.attrs["origin_info"]["creator_function"] == "extract_holog":
            n_lvls = 3
        elif self.root.attrs["origin_info"]["creator_function"] == "combine":
            n_lvls = 2
        else:
            n_lvls = len(key_order)

        if n_lvls == 1:
            pass
        elif n_lvls == 2 or n_lvls == 3:
            lvl_0_list = glob.glob(f"{mds_path}/*")
            for key_path_0 in lvl_0_list:
                if n_lvls == 3:
                    lvl_1_list = glob.glob(f"{key_path_0}/*")
                    for key_path_1 in lvl_1_list:
                        _consolidate_a_level(key_path_1)
                _consolidate_a_level(key_path_0)
        else:
            raise NotImplementedError(f"Unsupported number of levels: {n_lvls}")

        root_group = zarr.open(mds_path, mode="r+")  # Open in read/write mode
        zarr.convenience.consolidate_metadata(root_group.store)
        self.open()

    def open_subset(self, key_order: list[str]):
        """
        Open a subset of the file structure on disk.

        :param key_order: Order in which keys appear in file structure, ordered by depth.
        :type key_order: list

        Returns: Xarray subset object
        :rtype: xr.Dataset | xr.DataTree

        """
        subset_path = f"{self.filename}/{'/'.join(key_order)}"
        return xr.open_zarr(subset_path)


def _consolidate_a_level(key_path: str):
    """
    Consolidate a level containing data trees onto a single unified data tree entity.

    :param key_path: path at which to consolidate
    :type key_path: str

    :return: None
    :rtype: NoneType
    """
    if pathlib.Path(key_path).is_dir():
        key = key_path.split("/")[-1]
        try:
            this_lvl_xdt = xr.open_datatree(key_path, mode="r+", engine="zarr")
        except FileNotFoundError:
            this_lvl_xdt = xr.DataTree(name=key)
            this_lvl_xdt.to_zarr(key_path)

        del this_lvl_xdt
        this_zarr_group = zarr.open(key_path, mode="r+")
        zarr.convenience.consolidate_metadata(this_zarr_group.store)
    else:
        logger.warning(f"There is an unexpected entity at {key_path}")
