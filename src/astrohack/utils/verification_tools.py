import contextlib
import inspect
import io

import numpy as np
import xarray as xr
import xarray.testing
from PIL import Image, ImageChops

from astrohack.utils.fits import read_fits_no_checks
from astrohack.utils.package_info import get_astrohack_version
from astrohack.utils.text import undscr


def are_lists_equal(list_a, list_b):
    n_a = len(list_a)
    n_b = len(list_b)
    if n_a != n_b:
        return False
    else:
        equal = True
        for item in list_a:
            equal = equal and item in list_b
        return equal


def are_fits_files_close(fits_path1, fits_path2, tol=1e-5):
    head1, data1 = read_fits_no_checks(fits_path1)
    head2, data2 = read_fits_no_checks(fits_path2)

    if are_dicts_close(head1, head2, tol=tol, ignored_keys=["DATE", "ORIGIN"]):
        return np.allclose(data1, data2, equal_nan=True, atol=tol)
    else:
        return False


def are_png_files_close(img_path1, img_path2, tol=1e-5):
    try:
        # Open images (Pillow handles various modes and removes metadata concerns for pixel data)
        with Image.open(img_path1) as img1, Image.open(img_path2) as img2:
            # Ensure both images are in the same mode for a reliable comparison (e.g., 'RGBA')
            img1 = img1.convert("RGBA")
            img2 = img2.convert("RGBA")

            # Check if dimensions are the same
            if img1.size != img2.size:
                return False, "PNG sizes differ"

            # Calculate the difference between the images
            # This results in a new image where differing pixels are non-zero
            diff = ImageChops.difference(img1, img2)
            mean_diff = np.mean(diff)
            return (
                np.abs(mean_diff) < tol,
                f"Mean diff: {float(np.mean(np.absolute(diff)))}",
            )

    except OSError as e:
        print(f"Error opening images: {e}")
        return False, "Failed opening images"


def capture_prints_from_function(function, args=None):
    # Use redirect_stdout to capture the function's output
    output_capture = io.StringIO()
    with contextlib.redirect_stdout(output_capture):
        if args is None:
            function()
        else:
            function(*args)

    # Get the captured output as a string
    return output_capture.getvalue()


def are_txt_files_equal(txt_path1, txt_path2, ignored_key_words=()):
    with open(txt_path1) as txt_file1:
        txt1_content = txt_file1.read()
    with open(txt_path2) as txt_file2:
        txt2_content = txt_file2.read()

    txt1_lines = txt1_content.splitlines()
    txt2_lines = txt2_content.splitlines()
    if len(txt1_lines) != len(txt2_lines):
        return False
    else:
        for i_line, line1 in enumerate(txt1_lines):
            line2 = txt2_lines[i_line]
            if line1.strip() == line2.strip():
                continue
            else:
                found_ignored_keywords = False
                for key_word in ignored_key_words:
                    if key_word in line1 and key_word in line2:
                        found_ignored_keywords = True
                        break
                if found_ignored_keywords:
                    continue
                else:
                    return False
        return True


def is_captured_output_equal_to_txt_reference(function, txt_ref, args=None):
    captured_output = capture_prints_from_function(function, args)
    with open(txt_ref) as ref_file:
        ref_content = ref_file.read()
    return ref_content == captured_output


def _get_ds_metadata(ds):
    if hasattr(ds, "_input_pars"):
        metadata = ds._input_pars
    elif isinstance(ds, xr.Dataset) or isinstance(ds, xr.DataTree):
        metadata = ds.attrs
    else:
        metadata = ds.root.attrs
    return metadata


def are_dicts_close(dict_a, dict_b, tol=1e-8, ignored_keys=None):
    """
    Compares dictionaries and returns True if data is close up to tolerance.

    :param dict_a: First dictionary
    :type dict_a: dict

    :param dict_b: Second dictionary
    :type dict_b: dict

    :param tol: Tolerance
    :type tol: float

    :param ignored_keys: Keys to be ignored in comparison
    :type ignored_keys: list, NoneType

    :return: is_close
    :rtype: bool
    """
    if ignored_keys is None:
        ignored_keys = []
    is_close = True
    b_keys = list(dict_b.keys())
    for key, a_value in dict_a.items():
        if key in ignored_keys:
            mode = "ignored"
        else:
            if key in b_keys:
                key_in_b = True
            elif str(key) in b_keys:
                key_in_b = True
                key = str(key)
            else:
                try:
                    b_type = type(b_keys[0])
                    if b_type(key) in b_keys:
                        key_in_b = True
                        key = b_type(key)
                    else:
                        key_in_b = False
                except TypeError:
                    key_in_b = False
            if key_in_b:
                b_value = dict_b[key]
                if isinstance(a_value, dict) and isinstance(b_value, dict):
                    is_close = is_close and are_dicts_close(
                        a_value, b_value, tol=tol, ignored_keys=None
                    )
                    mode = "dict"
                elif type(a_value) is not type(b_value):
                    is_close = False
                    mode = "type diff"
                elif a_value is None:
                    is_close = is_close and (b_value is None)
                    mode = "None"
                elif isinstance(a_value, (np.ndarray, float, int)):
                    mode = "nparray"
                    is_close = is_close and np.allclose(
                        a_value, b_value, equal_nan=True, rtol=tol
                    )
                elif isinstance(a_value, str):
                    mode = "str"
                    is_close = is_close and a_value == b_value
                elif isinstance(a_value, (list, tuple)):
                    if isinstance(a_value[0], (float, int)):
                        mode = "number list"
                        is_close = is_close and np.allclose(
                            np.array(a_value),
                            np.array(b_value),
                            equal_nan=True,
                            rtol=tol,
                        )
                    else:
                        mode = "obj list"
                        is_close = is_close and a_value == b_value
                else:
                    raise TypeError(f"Unrecognized type {type(a_value)}")
            else:
                mode = "missing"
                is_close = False

        if not is_close:
            print(f"Key: {key} => {mode}")
            return False
        # print(f"Key: {key} => {mode} == {is_close}")
    return is_close


def are_data_trees_close(tree_a, tree_b, tol=1e-8):
    """
    Compares data trees and returns True if data is close up to tolerance.
    :param tree_a: First data tree
    :type tree_a: xarray.DataTree

    :param tree_b: Second data tree
    :type tree_b: xarray.DataTree

    :param tol: Tolerance
    :type tol: float

    :return: is_close
    :rtype: bool
    """
    is_close = True
    if are_dicts_close(tree_a.attrs, tree_b.attrs, tol=tol):
        try:
            xarray.testing.assert_allclose(tree_a.dataset, tree_b.dataset, rtol=tol)
        except AssertionError:
            print(f"Failed dataset comparison at {tree_a.name}")
            return False
        if tree_a.is_leaf and tree_b.is_leaf:
            pass
        else:
            a_key_list = list(tree_a.keys())
            b_key_list = list(tree_b.keys())
            if not are_lists_equal(a_key_list, b_key_list):
                print(f"Differing key lists at {tree_a.name}")
                return False
            for key, subtree_a in tree_a.items():
                is_close = is_close and are_data_trees_close(
                    subtree_a, tree_b[key], tol
                )
                if not is_close:
                    print(f"Failed key = {key}")
                    return False
    else:
        return False
    return is_close


def add_data_folder_to_names_in_class(class_ref):
    # Add datafolder to names for execution
    for varname, varvalue in class_ref.__dict__.items():
        if isinstance(varvalue, str):
            if varname.split(undscr)[-1] == "name":
                setattr(class_ref, varname, f"{class_ref.data_dir}/{varvalue}")


def analyse_summary(mds_obj, exp_file_name, exp_input_pars, exp_ant_keys_list):
    """analyse summary file"""
    summ_str = capture_prints_from_function(mds_obj.summary)
    n_input_pars = len(exp_input_pars.keys())

    i_start_input = 15
    i_end_input = i_start_input + n_input_pars
    i_start_orig = 6
    i_end_orig = 9
    inside_method_table = False
    inside_antenna_table = False
    exp_orig_info = create_origin_dict("test_summary")

    method_list = inspect.getmembers(mds_obj, predicate=inspect.ismethod)
    exp_method_list = []
    for name, method in method_list:
        if name[0] == undscr:
            continue
        else:
            exp_method_list.append(name)

    this_method_list = []
    this_ant_list = []
    this_input_pars = {}
    this_orig_info = {}
    this_filename = None
    for i_line, line in enumerate(summ_str.splitlines()):
        if i_line == 2:
            this_filename = line.split()[1]
        if i_start_orig <= i_line < i_end_orig:
            wrds = line.split(":")
            key = wrds[0].strip()
            value = wrds[1].strip()
            this_orig_info[key] = value
        elif i_start_input <= i_line < i_end_input:
            wrds = line.split("|")
            key = wrds[1].strip()
            value = wrds[2].strip()
            this_input_pars[key] = value
        elif line.strip() == "Available methods:":
            inside_method_table = True
        elif inside_method_table:
            if line.strip() == "":
                inside_method_table = False
            elif line[0] == "+":
                pass
            else:
                method_wrd = line.split("|")[1].strip()
                if method_wrd == "" or method_wrd == "Methods":
                    pass
                else:
                    this_method_list.append(method_wrd)
        elif line.strip() == "Data Contents:":
            inside_antenna_table = True
        elif inside_antenna_table:
            if line.strip() == "":
                inside_antenna_table = False
            elif line[0] == "+":
                pass
            else:
                ant_wrd = line.split("|")[1].strip()
                if ant_wrd == "" or ant_wrd == "Antenna":
                    pass
                else:
                    this_ant_list.append(ant_wrd)
        else:
            pass

    assert this_filename == exp_file_name, (
        "File name in Summary should be equal to the expected one"
    )

    assert are_dicts_close(this_input_pars, exp_input_pars), (
        "Input parameter dictionaries should identical"
    )

    assert are_dicts_close(this_orig_info, exp_orig_info), (
        "Origin info dictionaries should be identical"
    )

    assert are_lists_equal(this_ant_list, exp_ant_keys_list), (
        "Antenna list should be equal to the expected one"
    )

    assert are_lists_equal(this_method_list, exp_method_list), (
        "Method list should be equal to the expected one"
    )


def create_origin_dict(caller):
    orig_dict = {
        "origin": "astrohack",
        "version": get_astrohack_version(),
        "creator_function": caller,
    }
    return orig_dict


def execute_cleanup():
    import os

    skip_cleanup = os.getenv("SKIP_PYTEST_CLEANUP")
    if skip_cleanup == "True":
        return False
    else:
        return True


def produce_reference_data():
    import os

    reference_products = os.getenv("PRODUCE_REFERENCE_PRODUCTS")
    if reference_products == "True":
        return True
    else:
        return False
