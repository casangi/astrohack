import json
import numpy as np
import pathlib
import pandas as pd
import dropbox

from dropbox.exceptions import AuthError
from astrohack.dio import open_image

ACCESS_TOKEN = ""

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


def dropbox_connect():
    """Create a connection to Dropbox."""

    try:
        dbx = dropbox.Dropbox(ACCESS_TOKEN)
    except AuthError as e:
        print('Error connecting to Dropbox with access token: ' + str(e))
    return dbx


def dropbox_list_files(path):
    """Return a Pandas dataframe of files in a given Dropbox folder path in the Apps directory.
    """

    dbx = dropbox_connect()

    try:
        files = dbx.files_list_folder(path).entries
        files_list = []
        for file in files:
            if isinstance(file, dropbox.files.FileMetadata):
                metadata = {
                    'name': file.name,
                    'path_display': file.path_display,
                    'client_modified': file.client_modified,
                    'server_modified': file.server_modified
                }
                files_list.append(metadata)

        df = pd.DataFrame.from_records(files_list)
        return df.sort_values(by='server_modified', ascending=False)

    except Exception as e:
        print('Error getting list of files from Dropbox: ' + str(e))


def dropbox_download_file(dropbox_file_path, local_file_path):
    """Download a file from Dropbox to the local machine."""

    try:
        dbx = dropbox_connect()

        with open(local_file_path, 'wb') as f:
            metadata, result = dbx.files_download(path=dropbox_file_path)
            f.write(result.content)
    except Exception as e:
        print('Error downloading file from Dropbox: ' + str(e))


def dropbox_upload_file(local_path, local_file, dropbox_file_path):
    """Upload a file from the local machine to a path in the Dropbox app directory.

    Args:
        local_path (str): The path to the local file.
        local_file (str): The name of the local file.
        dropbox_file_path (str): The path to the file in the Dropbox app directory.

    Example:
        dropbox_upload_file('.', 'test.csv', '/stuff/test.csv')

    Returns:
        meta: The Dropbox file metadata.
    """

    try:
        dbx = dropbox_connect()

        local_file_path = pathlib.Path(local_path) / local_file

        with local_file_path.open("rb") as f:
            meta = dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode("overwrite"))

            return meta
    except Exception as e:
        print('Error uploading file to Dropbox: ' + str(e))
