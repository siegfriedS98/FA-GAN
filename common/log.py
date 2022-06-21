import logging
import sys


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logging.basicConfig(
        handlers=[
            logging.FileHandler("python_debug.log"),
            logging.StreamHandler(sys.stdout)
        ],
        format='%(asctime)s [%(name)s - %(levelname)s]: %(message)s'
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger