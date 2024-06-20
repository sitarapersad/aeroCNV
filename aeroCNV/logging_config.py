# logger_config.py
import logging


def setup_logger(log_file=None, level=logging.INFO):
    """
    Set up and return a logger instance.

    :param log_file: (str) Optional. Path to a file where logs should be written.
    :param level: (int) Logging level. Default is logging.INFO.
    :return: (logging.Logger) Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger('aeroCNV')
    logger.setLevel(level)

    # Prevent propagation to avoid duplicate logging
    logger.propagate = False

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Create formatter and set it for handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    if log_file:
        file_handler.setFormatter(formatter)

    return logger


# Initialize the logger at the module level
log = setup_logger()
