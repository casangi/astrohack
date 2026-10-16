#!/usr/bin/env python3

import argparse
import json
import pathlib
import shutil
import subprocess

import toolviper

parser = argparse.ArgumentParser(
    description="Upload test files to R2 cloudfare (Tokens are fetched from user Env)",
    formatter_class=argparse.RawTextHelpFormatter,
)

parser.add_argument("file", type=str, help="File to be uploaded")

parser.add_argument(
    "-p",
    "--project",
    type=str,
    default="astrohack",
    help="To which project folder to upload in cloudfare",
)

parser.add_argument(
    "-t",
    "--telescope",
    type=str,
    default="VLA",
    help="Telescope with which data was taken",
)

parser.add_argument(
    "-m",
    "--observation-mode",
    type=str,
    default="Holography",
    help="Observation mode",
)

parser.add_argument(
    "-d",
    "--data-type",
    type=str,
    default="CASA MS V2",
    help="Data type",
)

parser.add_argument(
    "-u",
    "--update-version",
    action="store_true",
    default=False,
    help="Update manifest version",
)

parser.add_argument(
    "-a",
    "--manifest-path",
    type=str,
    default=None,
    help="Path to manifest file, if None fetches it from toolviper",
)


remote_download_json_name = "file.download.json"


def execute_shell_command(command):
    results = subprocess.run(command, capture_output=True, text=True)
    return results.stdout


def download_manifest(args, manifest_filename=".manifest.json"):
    if args.manifest_path is None:
        downloaded_manifest = (
            toolviper.utils.data.__file__.split("__")[0]
            + f".cloudflare/{remote_download_json_name}"
        )
        toolviper.utils.data.update()
        shutil.copyfile(downloaded_manifest, manifest_filename)
    else:
        manifest_filename = args.manifest_path

    with open(manifest_filename) as manifest:
        manifest = json.load(manifest)
    return manifest_filename, manifest


def prepare_data_for_upload(args):
    is_dir = pathlib.Path(args.file).is_dir()
    if is_dir:
        meta_name = args.file + ".zip"
        print(f"Creating Ziped version of {args.file}...")
        execute_shell_command(["zip", "-r", meta_name, args.file])
        print("Done!")
    else:
        meta_name = args.file

    sha256sum = execute_shell_command(["sha256sum", meta_name]).split()[0]
    file_path = pathlib.Path(meta_name)
    file_size_bytes = file_path.stat().st_size

    file_properties = {
        "file": meta_name,
        "path": args.project,
        "dtype": args.data_type,
        "telescope": args.telescope,
        "size": f"{file_size_bytes:d}",
        "mode": args.observation_mode,
        "hash": sha256sum,
    }
    return file_properties


def update_manifest_version(manifest):
    current = manifest["version"]
    rev, major, minor = current.split(".")
    minor = f"{int(minor) + 1}"
    manifest["version"] = f"{rev}.{major}.{minor}"
    print(f"Updating manifest version from {current} to {manifest['version']}")


def add_data_to_manifest(args, manifest, file_properties, manifest_filename):
    if args.update_version:
        update_manifest_version(manifest)

    manifest["metadata"][args.file] = file_properties

    with open(manifest_filename, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=4)


def upload_a_file_to_cloudflare(
    s3_client, bucket_name, local_file_path, remote_file_path
):
    try:
        s3_client.upload_file(
            Filename=local_file_path, Bucket=bucket_name, Key=remote_file_path
        )
        print(
            f"File '{local_file_path}' uploaded successfully to '{bucket_name}/{remote_file_path}'"
        )
    except Exception as e:
        print(f"An error occurred during upload: {e}")


def upload_data_to_cloudflare(manifest_filename, file_properties):
    import os

    import boto3

    ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    BUCKET_NAME = "public-data"
    ACCESS_KEY_ID = os.environ.get("CLOUDFLARE_ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.environ.get("CLOUDFLARE_SECRET_ACCESS_KEY")

    s3_client = boto3.client(
        service_name="s3",
        # Provide your Cloudflare account ID
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        # Retrieve your S3 API credentials for your R2 bucket via API tokens (see: https://developers.cloudflare.com/r2/api/tokens)
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        region_name="auto",  # Required by SDK but not used by R2
    )

    upload_a_file_to_cloudflare(
        s3_client, BUCKET_NAME, manifest_filename, remote_download_json_name
    )
    local_file_name = file_properties["file"]
    remote_file_name = f"{file_properties['path']}/{local_file_name}"
    upload_a_file_to_cloudflare(
        s3_client, BUCKET_NAME, local_file_name, remote_file_name
    )

    # Update local version of toolviper database
    toolviper.utils.data.update()


def main():
    args = parser.parse_args()

    manifest_filename, manifest = download_manifest(args)

    file_properties = prepare_data_for_upload(args)

    add_data_to_manifest(
        args,
        manifest,
        file_properties,
        manifest_filename,
    )

    upload_data_to_cloudflare(manifest_filename, file_properties)

    return


if __name__ == "__main__":
    main()
