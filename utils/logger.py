import logging
import logging.config
from pathlib import Path

logging.config.fileConfig(str(Path(__file__).parents[1] / 'config' / 'log.ini'))


def get_logger(logger_name):
    return logging.getLogger(logger_name)
