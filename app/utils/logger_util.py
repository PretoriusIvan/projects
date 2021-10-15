import logging
import datetime


def create_logger(logger_name):
    # Date settings
    today = datetime.date.today()
    today_log_date = today.strftime('%d%b%Y')

    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create console handler
    stream_handler = logging.StreamHandler()

    # create formatter
    formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(message)s')
    stream_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.handlers = []
    logger.addHandler(stream_handler)