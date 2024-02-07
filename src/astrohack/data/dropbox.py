import os
import pathlib
import shutil
import json
import requests
import zipfile

import graphviper.utils.logger as logger


def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        else:
            raise ImportError

    except ImportError:
        return False


def download(file: str, folder: str = '.') -> None:
    # Load the file dropbox file meta data.
    meta_data_path = pathlib.Path(__file__).parent.joinpath(".dropbox/file.download.json")

    with open(meta_data_path) as json_file:
        file_meta_data = json.load(json_file)

    full_file_path = pathlib.Path(folder).joinpath(file)

    if full_file_path.exists():
        logger.info("File exists: {file}".format(file=str(full_file_path)))
        return

    if file not in file_meta_data.keys():
        logger.info("Requested file not found")
        logger.info(file_meta_data.keys())

        return

    fullname = file_meta_data[file]['file']
    id = file_meta_data[file]['id']
    rlkey = file_meta_data[file]['rlkey']

    url = 'https://www.dropbox.com/scl/fi/{id}/{file}?rlkey={rlkey}'.format(id=id, file=fullname, rlkey=rlkey)

    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}

    r = requests.get(url, stream=True, headers=headers)
    total = int(r.headers.get('content-length', 0))

    fullname = str(pathlib.Path(folder).joinpath(fullname))

    if is_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    with open(fullname, 'wb') as fd, tqdm(
            desc=fullname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024) as bar:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                size = fd.write(chunk)
                bar.update(size)

    if zipfile.is_zipfile(fullname):
        shutil.unpack_archive(filename=fullname, extract_dir=folder)

        # Let's clean up after ourselves
        os.remove(fullname)
