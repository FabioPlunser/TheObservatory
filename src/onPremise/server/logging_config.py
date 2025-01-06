import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("log.log"),
        ],
    )
