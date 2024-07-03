import numpy as np

from .logging_config import setup_logger
import logging


def initialize_logger(log_file=None, level=logging.INFO):
    """
    Initialize the logger for the package.

    :param log_file: (str) Optional. Path to a file where logs should be written.
    :param level: (int) Logging level. Default is logging.INFO.
    """
    global logger
    logger = setup_logger(log_file, level)

from . import simulate_data as simulate_data
from .cnvHMM import cnvHMM
from .dynamicHMM import HiddenMarkovModel as HMM

from . import utils
from . import optimize_likelihood as opt
from . import preprocess as pp
