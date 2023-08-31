import os
import shutil
import requests
import zipfile

from tqdm import tqdm
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

FILE_ID = {
        'ea25_cal_small_before_fixed.split.ms':
        {
            'file':'ea25_cal_small_before_fixed.split.ms.zip',
            'id':'AAAK_b89NVwTLYGLRmDs8_fBa'
        },
    }

def download(file, folder='.'):
  logger = _get_astrohack_logger()
    
  if os.path.exists('/'.join((folder, file))):
    logger.info("File exists.")
    return
    
  if file not in FILE_ID.keys():
    logger.info("Requested file not found")
    
    return 

  fullname=FILE_ID[file]['file']
  id=FILE_ID[file]['id']
    
  url = 'https://www.dropbox.com/sh/mrwny1055w35ofk/'+ id + '/' + fullname
    
  headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}

  r = requests.get(url, stream=True, headers=headers)
  total = int(r.headers.get('content-length', 0))

  with open(fullname, 'wb') as fd, tqdm(
    desc=fullname,
    total=total,
    unit='iB',
    unit_scale=True,
    unit_divisor=1024) as bar:
      for chunk in r.iter_content(chunk_size=1024):
        if chunk:
          size=fd.write(chunk)
          bar.update(size)
                
  if zipfile.is_zipfile(fullname):                
    shutil.unpack_archive(filename=fullname, extract_dir=folder)
    
    # Let's clean up after ourselves
    os.remove(fullname)