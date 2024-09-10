import os
from importlib.metadata import version

__version__ = version('astrohack')

from .extract_holog import HologObsDict, extract_holog, generate_holog_obs_dict, model_memory_usage

from .extract_pointing import *
from .holog import *
from .dio import *
from .panel import *
from .combine import *
from .mds import *
from .locit import *
from .extract_locit import *

# This installs a slick, informational tracebacks logger
from rich.traceback import install
from toolviper.utils.logger import setup_logger

install(show_locals=False)

if not os.getenv("LOGGER_NAME"):
    os.environ["LOGGER_NAME"] = "astrohack"
    setup_logger(
        logger_name="astrohack",
        log_to_term=True,
        log_to_file=False,
        log_file="astrohack-logfile",
        log_level="INFO",
    )
