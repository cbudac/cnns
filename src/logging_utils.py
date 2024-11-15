import logging

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])


def get_logger(name: str = None):
    return logging.getLogger(name)
