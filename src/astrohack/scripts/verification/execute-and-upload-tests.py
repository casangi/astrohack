from astrohack.utils.package_info import get_astrohack_path
import os
import shutil
import pytest
import subprocess

from astrohack.utils.text import format_duration


def set_tests_to_update():
    test_dict = {
        "unit/mdses/test_beamcut_mds.py": {
            "n_items": 1,
            "item_0": {
                "creation": "beamcut_exports",
                "destiny": "ref_beamcut_products",
                "description": "beamcut reference export products",
                "type": "Beam cut",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["beamcut_data"],
        },
        "unit/mdses/test_holog_mds.py": {
            "n_items": 1,
            "item_0": {
                "creation": "holog_exports",
                "destiny": "ref_holog_products",
                "description": "holog export products",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["holog_data"],
        },
        "unit/mdses/test_image_mds.py": {
            "n_items": 1,
            "item_0": {
                "creation": "image_exports",
                "destiny": "ref_image_products",
                "description": "image export products",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["image_data"],
        },
        "unit/mdses/test_locit_mds.py": {
            "n_items": 1,
            "item_0": {
                "creation": "locit_exports",
                "destiny": "ref_locit_products",
                "description": "locit reference export products",
                "type": "Antenna position corrections",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["locit_data"],
        },
        "unit/mdses/test_panel_mds.py": {
            "n_items": 1,
            "item_0": {
                "creation": "panel_exports",
                "destiny": "ref_panel_products",
                "description": "panel export products",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["panel_data"],
        },
        "unit/mdses/test_position_mds.py": {
            "n_items": 1,
            "item_0": {
                "creation": "position_exports",
                "destiny": "ref_position_products",
                "description": "position export products",
                "type": "Antenna position corrections",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["position_data"],
        },
        "unit/user_facing_functions/test_beamcut.py": {
            "n_items": 1,
            "item_0": {
                "creation": "beamcut_data/kband_beamcut_small_local.beamcut.zarr",
                "destiny": "kband_beamcut_small.beamcut.zarr",
                "description": "beamcut reference Astrohack beamcut file",
                "type": "Beam cut",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["beamcut_data"],
        },
        "unit/user_facing_functions/test_extract_holog.py": {
            "n_items": 1,
            "item_0": {
                "creation": "ext_holog_data/ea25_cal_small_before_fixed.split.holog.zarr",
                "destiny": "ea25_cal_small_before_reference.holog.zarr",
                "description": "Reference Astrohack holog file",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["ext_holog_data"],
        },
        "unit/user_facing_functions/test_holog.py": {
            "n_items": 2,
            "item_0": {
                "creation": "holog_data/ea25_cal_small_before_reference.image.zarr",
                "destiny": "ea25_cal_before_reference.image.zarr",
                "description": "Reference Astrohack image file",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "item_1": {
                "creation": "holog_data/holog-ref-values.json",
                "destiny": "holog-ref-values.json",
                "description": "json with reference values for holog execution",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["holog_data"],
        },
        "unit/user_facing_functions/test_panel.py": {
            "n_items": 1,
            "item_0": {
                "creation": "panel_data/ea25_cal_before_reference.panel.zarr",
                "destiny": "ea25_before_reference.panel.zarr",
                "description": "Reference Astrohack panel file",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["panel_data"],
        },
        "unit/user_facing_functions/test_combine.py": {
            "n_items": 1,
            "item_0": {
                "creation": "combine_data/ea25_cal_before_reference.combine.zarr",
                "destiny": "ea25_before_reference.combine.zarr",
                "description": "Reference Astrohack combine file",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["combine_data"],
        },
        "unit/antenna_classes/test_class_antenna_surface.py": {
            "n_items": 1,
            "item_0": {
                "creation": "ant_class_data/ant_class_ref.json",
                "destiny": "ant_class_ref.json",
                "description": "Antenna class reference value json file",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": True,
            },
            "cleanup_names": ["ant_class_data"],
        },
        "stakeholder/test_stakeholder_vla.py": {
            "n_items": 1,
            "item_0": {
                "creation": "stakeholder_test_data/vla_stakeholder_ref.json",
                "destiny": "vla_stakeholder_ref.json",
                "description": "Reference VLA Astrohack stakeholder json file",
                "type": "Holography",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["stakeholder_test_data"],
        },
        "unit/user_facing_functions/test_locit.py": {
            "n_items": 1,
            "item_0": {
                "creation": "locit_data/locit-input-pha-reference.position.zarr",
                "destiny": "locit-reference.position.zarr",
                "description": "Reference Astrohack position file",
                "type": "Antenna position corrections",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["locit_data"],
        },
        "unit/user_facing_functions/test_extract_locit.py": {
            "n_items": 1,
            "item_0": {
                "creation": "extract_locit_data/locit-input-pha.locit.zarr",
                "destiny": "locit-input-pha-reference.locit.zarr",
                "description": "Reference Astrohack locit file",
                "type": "Antenna position corrections",
                "telescope": "VLA",
                "update_manifest": False,
            },
            "cleanup_names": ["extract_locit_data"],
        },
        "unit/user_facing_functions/test_fringefit_locit.py": {
            "n_items": 1,
            "item_0": {
                "creation": "fringefit_locit_data/fft_locit_input.position.zarr",
                "destiny": "fft_locit_reference.position.zarr",
                "description": "Reference Astrohack fringefit position file",
                "type": "Antenna position corrections",
                "telescope": "ngVLA",
                "update_manifest": False,
            },
            "cleanup_names": ["fringefit_locit_data"],
        },
    }

    return test_dict


def execute_and_upload(test_file, test_params):

    pytest.main([test_file])
    n_items = test_params["n_items"]
    for i_item in range(n_items):
        item_pars = test_params[f"item_{i_item}"]
        subproc_exec_list = [
            "python",
            str(get_astrohack_path() / "scripts" / "verification" / "file_uploader.py"),
            item_pars["destiny"],
            "-t",
            item_pars["telescope"],
            "-m",
            item_pars["type"],
            "-d",
            item_pars["description"],
        ]

        shutil.rmtree(item_pars["destiny"], ignore_errors=True)
        shutil.move(item_pars["creation"], item_pars["destiny"])
        if item_pars["update_manifest"]:
            subproc_exec_list.append("-u")
        print(" ".join(subproc_exec_list))
        subprocess.call(subproc_exec_list)
        shutil.rmtree(item_pars["destiny"], ignore_errors=True)
        if os.path.exists(item_pars["destiny"] + ".zip"):
            os.remove(item_pars["destiny"] + ".zip")

    for name in test_params["cleanup_names"]:
        shutil.rmtree(name, ignore_errors=True)


def main():
    import time

    wait_time = 60
    distro_path = get_astrohack_path()
    os.chdir(distro_path / "../../tests")

    tests_to_execute = [
        # "unit/user_facing_functions/test_extract_holog.py",
        # "unit/user_facing_functions/test_holog.py",
        # "unit/user_facing_functions/test_panel.py",
        # "unit/user_facing_functions/test_combine.py",
        # "unit/user_facing_functions/test_beamcut.py",
        # "unit/antenna_classes/test_class_antenna_surface.py"
        # "unit/mdses/test_holog_mds.py",
        # "unit/mdses/test_image_mds.py",
        # "unit/mdses/test_locit_mds.py",
        # "unit/mdses/test_beamcut_mds.py",
        # "unit/mdses/test_panel_mds.py",
        # "unit/mdses/test_position_mds.py"
        # "stakeholder/test_stakeholder_vla.py",
        # "unit/user_facing_functions/test_locit.py",
        # "unit/user_facing_functions/test_extract_locit.py",
        # "unit/user_facing_functions/test_fringefit_locit.py",
    ]

    os.environ["SKIP_PYTEST_CLEANUP"] = "True"
    os.environ["PRODUCE_REFERENCE_PRODUCTS"] = "True"
    test_dict = set_tests_to_update()
    for test_file in tests_to_execute:
        test_params = test_dict[test_file]
        execute_and_upload(test_file, test_params)
        if len(tests_to_execute) > 1:
            print(
                f"{test_file} done, waiting {format_duration(wait_time)} for cloudflare sync before executing next test."
            )
            time.sleep(wait_time)
    os.environ["SKIP_PYTEST_CLEANUP"] = "False"
    os.environ["PRODUCE_REFERENCE_PRODUCTS"] = "False"


main()
