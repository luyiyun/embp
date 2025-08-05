import logging


logger_embp = logging.getLogger("EMBP")
logger_embp.setLevel(logging.ERROR)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter(
    "[%(name)s][%(levelname)s][%(asctime)s]:%(message)s"
)
ch.setFormatter(formatter)
logger_embp.addHandler(ch)
