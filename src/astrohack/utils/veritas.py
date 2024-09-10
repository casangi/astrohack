import json
import dropbox
import pathlib
import toolviper

import pandas as pd
import numpy as np

from astrohack.extract_pointing import extract_pointing
from astrohack.extract_holog import extract_holog
from astrohack.holog import holog
from astrohack.panel import panel
from astrohack.dio import open_panel

from dropbox.exceptions import AuthError
from astrohack.dio import open_image
from toolviper.utils import logger


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


def generate_verification_json(path, antenna, ddi, write=False, generate_files=True):
    from astrohack.dio import open_panel

    if generate_files:
        generate_verification_files()

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
            "grid_size": [],
            "offsets": []
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

    cell_size, grid_size = get_grid_parameters(file=f"{path}/before.split.holog.zarr")

    numerical_dict["vla"]["cell_size"] = [-cell_size, cell_size]
    numerical_dict["vla"]["grid_size"] = [grid_size, grid_size]

    # Fill panel offsets
    panel_list = ['3-4', '5-27', '5-37', '5-38']

    M_TO_MILS = 39370.1

    before_mds = open_panel(f"{path}/before.split.panel.zarr")
    after_mds = open_panel(f"{path}/after.split.panel.zarr")

    before_shift = before_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values * M_TO_MILS
    after_shift = after_mds[antenna][ddi].sel(labels=panel_list).PANEL_SCREWS.values * M_TO_MILS

    numerical_dict["vla"]["offsets"] = np.mean(after_shift - before_shift, axis=1).tolist()
    print(np.mean(after_shift - before_shift, axis=1).tolist())

    if write:
        with open("data/holog_numerical_verification.json", "w") as json_file:
            json.dump(numerical_dict, json_file)

    return numerical_dict


def generate_verification_files():
    for stub in ["before", "after"]:
        toolviper.utils.data.download(file=f"ea25_cal_small_{stub}_fixed.split.ms", folder="data/")

        extract_pointing(
            ms_name=f"data/ea25_cal_small_{stub}_fixed.split.ms",
            point_name=f"data/{stub}.split.point.zarr",
            overwrite=True,
            parallel=False
        )

        # Extract holography data using holog_obd_dict
        holog_mds = extract_holog(
            ms_name=f"data/ea25_cal_small_{stub}_fixed.split.ms",
            point_name=f"data/{stub}.split.point.zarr",
            holog_name=f"data/{stub}.split.holog.zarr",
            data_column="CORRECTED_DATA",
            parallel=False,
            overwrite=True
        )

        image_mds = holog(
            holog_name=f"data/{stub}.split.holog.zarr",
            image_name=f"data/{stub}.split.image.zarr",
            overwrite=True,
            parallel=False
        )

        before_mds = panel(
            image_name=f"data/{stub}.split.image.zarr",
            panel_model='rigid',
            parallel=False,
            overwrite=True
        )


def generate_panel_mask_array(generate_files=True):
    if generate_files:
        generate_verification_files()

    panel_mds = panel(
            image_name='data/before.split.image.zarr',
            clip_type='absolute',
            clip_level=0.0,
            parallel=False,
            overwrite=True
        )

    before_mds = open_panel("data/before.split.panel.zarr")

    with open("data/panel_cutoff_mask.npy", "wb") as outfile:
        np.save(outfile, before_mds["ant_ea25"]["ddi_0"].MASK.values)


class Veritas:
    __slots__ = ["dbx", "key", "secret", "certificate_path"]

    def __init__(self):
        self.dbx = None
        self.key = None
        self.secret = None
        self.certificate_path = None

    def __del__(self):
        self.dbx.close()

    def connect(self, certificate_path=None):
        """Create a connection to Dropbox."""

        try:
            self.decrypt_data(certificate_path)
            self.connect_o2_auth()
            logger.info("Connected to Dropbox")

        except AuthError as e:
            logger.error('Error connecting to Dropbox with access token: ' + str(e))

    def list_files(self, path):
        """Return a Pandas dataframe of files in a given Dropbox folder path in the Apps directory.
        """

        if self.dbx is None:
            logger.info("No connection to dropbox. Run connect() command.")
            return None

        try:
            files = self.dbx.files_list_folder(path).entries
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
            logger.error('Error getting list of files from Dropbox: ' + str(e))

    def download_file(self, dropbox_file_path, local_file_path):
        """Download a file from Dropbox to the local machine."""

        if self.dbx is None:
            logger.info("No connection to dropbox. Run connect() command.")
            return None

        try:
            with open(local_file_path, 'wb') as f:
                metadata, result = self.dbx.files_download(path=dropbox_file_path)
                f.write(result.content)

        except Exception as e:
            logger.error('Error downloading file from Dropbox: ' + str(e))

    def upload_file(self, local_path, local_file, remote_file_path):
        """Upload a file from the local machine to a path in the Dropbox app directory.

        Args:
            local_path (str): The path to the local file.
            local_file (str): The name of the local file.
            remote_file_path (str): The path to the file in the Dropbox app directory.

        Example:
            dropbox_upload_file('.', 'test.csv', '/stuff/test.csv')

        Returns:
            meta: The Dropbox file metadata.
        """

        if self.dbx is None:
            logger.info("No connection to dropbox. Run connect() command.")
            return None

        try:

            local_file_path = pathlib.Path(local_path) / local_file

            with local_file_path.open("rb") as f:
                meta = self.dbx.files_upload(f.read(), remote_file_path, mode=dropbox.files.WriteMode("overwrite"))

                return meta
        except Exception as e:
            logger.error('Error uploading file to Dropbox: ' + str(e))

    def connect_o2_auth(self):
        from dropbox import DropboxOAuth2FlowNoRedirect

        '''
        This example walks through a basic oauth flow using the existing long-lived token type
        Populate your app key and app secret in order to run this locally
        '''

        auth_flow = DropboxOAuth2FlowNoRedirect(self.key, self.secret)

        authorize_url = auth_flow.start()
        print("1. Go to: " + authorize_url)
        print("2. Click \"Allow\" (you might have to log in first).")
        print("3. Copy the authorization code.")
        auth_code = input("Enter the authorization code here: ").strip()

        try:
            oauth_result = auth_flow.finish(auth_code)
            self.dbx = dropbox.Dropbox(oauth2_access_token=oauth_result.access_token)

        except Exception as error:
            logger.exception(f"Error: {error}")
            raise AuthError

    def decrypt_data(self, certificate_path=None):
        from Crypto.PublicKey import RSA
        from Crypto.Cipher import AES, PKCS1_OAEP

        if certificate_path is not None:
            self.certificate_path = certificate_path

        try:
            private_key = RSA.import_key(open(f"{self.certificate_path}/private.pem").read())

            with open(f"{self.certificate_path}/encrypted_data.bin", "rb") as f:
                enc_session_key = f.read(private_key.size_in_bytes())
                nonce = f.read(16)
                tag = f.read(16)
                ciphertext = f.read()

            # Decrypt the session key with the private RSA key
            cipher_rsa = PKCS1_OAEP.new(private_key)
            session_key = cipher_rsa.decrypt(enc_session_key)

            # Decrypt the data with the AES session key
            cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
            data = cipher_aes.decrypt_and_verify(ciphertext, tag)

            token = data.decode("utf-8")
            self.key, self.secret = token.split(":")

        except Exception as error:
            logger.exception(f"Error: {error}")
            raise AuthError


def generate_rsa_key():
    from Crypto.PublicKey import RSA

    if not pathlib.Path(".keys").exists():
        pathlib.Path(".keys").mkdir(parents=True, exist_ok=True)

    key = RSA.generate(2048)
    private_key = key.export_key()
    with open(".keys/private.pem", "wb") as f:
        f.write(private_key)

    public_key = key.publickey().export_key()
    with open(".keys/receiver.pem", "wb") as f:
        f.write(public_key)


def encrypt_data(key: str, secret: str):
    from Crypto.PublicKey import RSA
    from Crypto.Random import get_random_bytes
    from Crypto.Cipher import AES, PKCS1_OAEP

    generate_rsa_key()

    data = ":".join((key, secret)).encode("utf-8")

    recipient_key = RSA.import_key(open(".keys/receiver.pem").read())
    session_key = get_random_bytes(16)

    # Encrypt the session key with the public RSA key

    cipher_rsa = PKCS1_OAEP.new(recipient_key)
    enc_session_key = cipher_rsa.encrypt(session_key)

    # Encrypt the data with the AES session key

    cipher_aes = AES.new(session_key, AES.MODE_EAX)
    ciphertext, tag = cipher_aes.encrypt_and_digest(data)

    with open(".keys/encrypted_data.bin", "wb") as f:
        f.write(enc_session_key)
        f.write(cipher_aes.nonce)
        f.write(tag)
        f.write(ciphertext)
