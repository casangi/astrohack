import json
import numpy as np

from astrohack.dio import open_image


def get_center_pixel(file, antenna, ddi):

    mds = open_image(file)[antenna][ddi]

    aperture_shape = mds.APERTURE.values.shape[-2], mds.APERTURE.values.shape[-1]
    beam_shape = mds.BEAM.values.shape[-2], mds.BEAM.values.shape[-1]

    aperture_center_pixels = mds.APERTURE.values[..., aperture_shape[0] // 2, aperture_shape[1] // 2]
    beam_center_pixels = mds.BEAM.values[..., beam_shape[0] // 2, beam_shape[1] // 2]

    return {"aperture": np.squeeze(aperture_center_pixels), "beam": np.squeeze(beam_center_pixels)}


def get_grid_parameters(file):
    with open(f"{file}/.holog_attr") as json_file:
        json_object = json.load(json_file)

    cell_size = json_object["cell_size"]
    grid_size = int(np.sqrt(json_object["n_pix"]))

    return cell_size, grid_size


def generate_verification_json(path, antenna, ddi, write=False):
    numerical_dict = {
        "vla": {
            "pixels": {
                "before": {
                    "aperture": [],
                    "beam": []
                },
                "after": {
                    "aperture": [],
                    "beam": []
                }
            },
            "cell_size": [],
            "grid_size": []
        }
    }

    for tag in ["before", "after"]:
        pixels = get_center_pixel(
            file=f"{path}/{tag}.split.image.zarr",
            antenna=antenna,
            ddi=ddi
        )

        numerical_dict["vla"]["pixels"][tag]["aperture"] = list(map(str, pixels["aperture"]))
        numerical_dict["vla"]["pixels"][tag]["beam"] = list(map(str, pixels["beam"]))

        cell_size, grid_size = get_grid_parameters(file=f"{path}/{tag}.split.holog.zarr")

        numerical_dict["vla"]["cell_size"] = [-cell_size, cell_size]
        numerical_dict["vla"]["grid_size"] = [grid_size, grid_size]

        if write:
            with open("holog_numerical_verification.json", "w") as json_file:
                json.dump(numerical_dict, json_file)

    return numerical_dict
