import os
import pathlib
import astrohack

import skriba.logger


def download(file: str, folder: str = '.', unpack: bool = False, source: str = 'dropbox') -> None:
    logger = skriba.logger.get_logger()

    if not pathlib.Path(folder).exists():
        os.mkdir(folder)

    if source == 'gdrive':
        astrohack.data.google_drive.download(file=file, folder=folder, unpack=unpack)

    elif source == 'dropbox':
        astrohack.data.dropbox.download(file=file, folder=folder)

    else:
        logger.error("unknown source or issue found")
        pass
