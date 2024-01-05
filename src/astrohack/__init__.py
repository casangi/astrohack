from importlib.metadata import version

__version__ = version('astrohack')

from .client import *
from .extract_holog import *
from .extract_holog import generate_holog_obs_dict
from .extract_pointing import *
from .holog import *
from .dio import *
from .panel import *
from .combine import *
from .mds import *
from .data import *
from .locit import *
from .extract_locit import *

# Set parameter checking system directory.
if os.path.exists(os.path.dirname(__file__) + "/config/"):
    os.environ["AUROR_CONFIG_PATH"] = os.path.dirname(__file__) + "/config/"

# This installs a slick, informational tracebacks logger
from rich.traceback import install
from skriba.logger import setup_logger

install(show_locals=False)

if not os.getenv("SKRIBA_LOGGER_NAME"):
    os.environ["SKRIBA_LOGGER_NAME"] = "astrohack"
    setup_logger(
        logger_name="astrohack",
        log_to_term=True,
        log_to_file=False,
        log_file="astrohack-logfile",
        log_level="INFO",
    )
