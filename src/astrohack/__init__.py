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

# Set parameter checking system directory
if os.path.exists(os.path.dirname(__file__) + "/config/"):
    os.environ["AUROR_CONFIG_PATH"] = os.path.dirname(__file__) + "/config/"

import skriba.logger

logger = skriba.logger.get_logger(logger_name="__init__")
logger.info("PATH: {}".format(os.path.dirname(__file__)))
