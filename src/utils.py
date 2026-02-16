import logging
import sys

"""
utils.py: Logging and System Helpers
Provides a standardized logging configuration to track experimental 
progress and errors across distributed processes.
"""

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger
