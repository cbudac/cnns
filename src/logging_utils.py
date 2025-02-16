import logging

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)
logging.getLogger("fsspec.local").setLevel(logging.ERROR)


logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])



def get_logger(name: str = None):
    return logging.getLogger(name)
