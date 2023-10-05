import os
import shutil
import requests
import zipfile

from tqdm import tqdm
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

FILE_ID = {
    'ea25_cal_small_before_fixed.split.ms':
        {
            'file': 'ea25_cal_small_before_fixed.split.ms.zip',
            'id': 'm2qnd2w6g9fhdxyzi6h7f',
            'rlkey': 'd4dgztykxpnnqrei7jhb1cu7m'
        },
    'ea25_cal_small_after_fixed.split.ms':
        {
            'file': 'ea25_cal_small_after_fixed.split.ms.zip',
            'id': 'o3tl05e3qa440s4rk5owf',
            'rlkey': 'hoxte3zzeqgkju2ywnif2t7ko'
        },
    'J1924-2914.ms.calibrated.split.SPW3':
        {
            'file': 'J1924-2914.ms.calibrated.split.SPW3.zip',
            'id': 'kyrwc5y6u7lxbmqw7fveh',
            'rlkey': 'r23qakcm24bid2x2cojsd96gs'
        },
    'extract_holog_verification.json':
        {
            'file': 'extract_holog_verification.json',
            'id': '6pzucjd48a4n0eb74wys9',
            'rlkey': 'azuynw358zxvse9i225sbl59s'
        },
    'holog_numerical_verification.json':
        {
            'file': 'holog_numerical_verification.json',
            'id': 'x69700pznt7uktwprdqpk',
            'rlkey': 'bxn9me7dgnxrtzvvay7xgicmi'
        },
    'locit-input-pha.cal':
        {
            'file': 'locit-input-pha.cal.zip',
            'id': '8fftz5my9h8ca2xdlupym',
            'rlkey': 'fxfid92953ycorh5wrhfgh78b'
        },
    'panel_cutoff_mask':
        {
            'file': 'panel_cutoff_mask.npy',
            'id': '8ta02t72vwcv4ketv8rfw',
            'rlkey': 'qsmos4hx2duz8upb83hghi6q8'
        }

}


def download(file, folder='.'):
    logger = _get_astrohack_logger()

    if os.path.exists('/'.join((folder, file))):
        logger.info("File exists.")
        return

    if file not in FILE_ID.keys():
        logger.info("Requested file not found")
        logger.info(FILE_ID.keys())

        return

    fullname = FILE_ID[file]['file']
    id = FILE_ID[file]['id']
    rlkey = FILE_ID[file]['rlkey']

    url = 'https://www.dropbox.com/scl/fi/{id}/{file}?rlkey={rlkey}'.format(id=id, file=fullname, rlkey=rlkey)

    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}

    r = requests.get(url, stream=True, headers=headers)
    total = int(r.headers.get('content-length', 0))

    fullname = '/'.join((folder, fullname))

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
