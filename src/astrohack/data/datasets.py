import os
import astrohack

from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

def download(file, folder='.', unpack=False, source='dropbox'):
  logger = _get_astrohack_logger()

  if os.path.exists(folder) == False:
    os.mkdir(folder)
  
  if source == 'gdrive':
    astrohack.data._google_drive.download(file=file, folder=folder, unpack=unpack)

  elif source == 'dropbox':
    astrohack.data._dropbox.download(file=file, folder=folder)

  else:
    logger.error("unknown source or issue found")
    pass
