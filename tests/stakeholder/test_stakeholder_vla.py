import json
import pytest
import toolviper

import numpy as np

from astrohack.dio import open_panel
from astrohack.extract_holog import extract_holog
from astrohack.extract_pointing import extract_pointing
from astrohack.holog import holog
from astrohack.panel import panel

base_name = "ea25_cal_small_"


@pytest.fixture(scope="session")
def set_data(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")

    # Data files
    toolviper.utils.data.download(
        file="ea25_cal_small_before_fixed.split.ms", folder=str(data_dir)
    )
    toolviper.utils.data.download(
        file="ea25_cal_small_after_fixed.split.ms", folder=str(data_dir)
    )

    # Verification json information
    toolviper.utils.data.download(
        file="extract_holog_verification.json", folder=str(data_dir)
    )
    toolviper.utils.data.download(
        file="holog_numerical_verification.json", folder=str(data_dir)
    )

    return data_dir


def relative_difference(result, expected):
    return 2 * np.abs(result - expected) / (abs(result) + abs(expected))


def verify_panel_shifts(
    data_dir="",
    panel_list=None,
    expected_shift=np.array([-100, 75, 0, 150]),
    ref_mean_shift=np.array([-91.47636864, 60.34743659, 4.16119043, 122.40537789]),
    antenna="ant_ea25",
    ddi="ddi_0",
):
    if panel_list is None:
        panel_list = ["3-4", "5-27", "5-37", "5-38"]

    M_TO_MILS = 39370.1

    before_mds = open_panel("{data}/vla.before.split.panel.zarr".format(data=data_dir))
    after_mds = open_panel("{data}/vla.after.split.panel.zarr".format(data=data_dir))

    before_shift = (
        before_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values * M_TO_MILS
    )
    after_shift = (
        after_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values * M_TO_MILS
    )

    difference = after_shift - before_shift

    mean_shift = np.mean(difference, axis=1)
    print(mean_shift)

    delta_mean_shift = np.abs(mean_shift - expected_shift)
    delta_ref_shift = np.abs(ref_mean_shift - expected_shift)

    # New corrections - old corrections --> delta if delta < 0 ==> we improved.
    delta_shift = delta_mean_shift - delta_ref_shift
    relative_shift = relative_difference(delta_mean_shift, delta_ref_shift)
    print(relative_shift)

    return np.all(relative_shift < 6e-2)


def verify_center_pixels(file, antenna, ddi, reference_center_pixels, tolerance=1e-6):
    from astrohack.dio import open_image

    mds = open_image(file)[antenna][ddi]

    aperture_shape = mds.APERTURE.values.shape[-2], mds.APERTURE.values.shape[-1]
    beam_shape = mds.BEAM.values.shape[-2], mds.BEAM.values.shape[-1]

    aperture_center_pixels = np.squeeze(
        mds.APERTURE.values[..., aperture_shape[0] // 2, aperture_shape[1] // 2]
    )
    beam_center_pixels = np.squeeze(
        mds.BEAM.values[..., beam_shape[0] // 2, beam_shape[1] // 2]
    )

    aperture_ref = list(map(complex, reference_center_pixels["aperture"]))
    beam_ref = list(map(complex, reference_center_pixels["beam"]))

    for i in range(len(aperture_ref)):
        aperture_check = (
            relative_difference(aperture_ref[i].real, aperture_center_pixels[i].real)
            < tolerance
        )

        beam_check = (
            relative_difference(beam_ref[i].real, beam_center_pixels[i].real)
            < tolerance
        )

        real_check = aperture_check and beam_check

        aperture_check = (
            relative_difference(aperture_ref[i].imag, aperture_center_pixels[i].imag)
            < tolerance
        )

        beam_check = (
            relative_difference(beam_ref[i].imag, beam_center_pixels[i].imag)
            < tolerance
        )

        imag_check = aperture_check and beam_check

    return real_check and imag_check


def verify_holog_obs_dictionary(after_file, holog_obs_dict):
    with open(f"{after_file}/holog_obs_dict.json", "r") as json_file:
        holog_obj = json.load(json_file)

    return holog_obj == holog_obs_dict


def test_holography_pipeline(set_data):
    before_ms = str(set_data / "".join((base_name, "before_fixed.split.ms")))
    before_point = str(set_data / "vla.before.split.point.zarr")
    before_holog = str(set_data / "vla.before.split.holog.zarr")

    after_ms = str(set_data / "".join((base_name, "after_fixed.split.ms")))
    after_point = str(set_data / "vla.after.split.point.zarr")
    after_holog = str(set_data / "vla.after.split.holog.zarr")

    with open(str(set_data / "extract_holog_verification.json")) as file:
        holog_obs_dict = json_dict = json.load(file)

    extract_pointing(
        ms_name=before_ms, point_name=before_point, parallel=False, overwrite=True
    )

    extract_pointing(
        ms_name=after_ms, point_name=after_point, parallel=False, overwrite=True
    )

    extract_holog(
        ms_name=before_ms,
        point_name=before_point,
        holog_name=before_holog,
        data_column="CORRECTED_DATA",
        parallel=False,
        overwrite=True,
    )

    extract_holog(
        ms_name=after_ms,
        point_name=after_point,
        holog_name=after_holog,
        data_column="CORRECTED_DATA",
        parallel=False,
        overwrite=True,
    )

    assert verify_holog_obs_dictionary(
        after_holog, holog_obs_dict["vla"]["after"]
    ), "Verifiy holog obs dictionary"

    with open(str(set_data / "holog_numerical_verification.json")) as file:
        reference_dict = json.load(file)

    before_holog = str(set_data / "vla.before.split.holog.zarr")
    after_holog = str(set_data / "vla.after.split.holog.zarr")

    before_image = str(set_data / "vla.before.split.image.zarr")
    after_image = str(set_data / "vla.after.split.image.zarr")

    holog(holog_name=before_holog, overwrite=True, parallel=False, grid_size=[31, 31],
          cell_size=[-0.0006386556122807017, 0.0006386556122807017])

    assert verify_center_pixels(
        file=before_image,
        antenna="ant_ea25",
        ddi="ddi_0",
        reference_center_pixels=reference_dict["vla"]["pixels"]["before"],
        tolerance=1.5e-6,
    ), "Verifiy center pixels-before"

    holog(holog_name=after_holog, overwrite=True, parallel=False, grid_size=[31, 31],
          cell_size=[-0.0006386556122807017, 0.0006386556122807017])

    assert verify_center_pixels(
        file=after_image,
        antenna="ant_ea25",
        ddi="ddi_0",
        reference_center_pixels=reference_dict["vla"]["pixels"]["after"],
        tolerance=1.5e-6,
    ), "Verifiy center pixels-after"

    before_panel = panel(
        image_name=before_image,
        panel_model="rigid",
        parallel=False,
        overwrite=True,
        exclude_shadows=False,
    )

    after_panel = panel(
        image_name=after_image,
        panel_model="rigid",
        parallel=False,
        overwrite=True,
        exclude_shadows=False,
    )

    reference_shifts = np.array([-91.6455227, 61.69666059, 4.39843319, 122.26547831])
    assert verify_panel_shifts(
        data_dir=str(set_data), ref_mean_shift=reference_shifts
    ), "Verify panel shifts"
    # This test using reference values is very hard to be updated, using this hardcoded reference_shifts is a
    # temporary work around
    # assert verify_panel_shifts(data_dir=str(set_data), ref_mean_shift=reference_dict["vla"]["offsets"]), \
    #     "Verify panel shifts"
