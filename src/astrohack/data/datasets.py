import pathlib
import astrohack


def download(file: str, folder: str = '.') -> None:

    if not pathlib.Path(folder).exists():
        pathlib.Path(folder).mkdir()

    astrohack.data.dropbox.download(file=file, folder=folder)