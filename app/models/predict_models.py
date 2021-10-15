import logging
from app.config import config


class PredictModelClass:

    def __init__(self):
        self.logger = logging.getLogger(f'{config.LOGGER_NAME}')

    # TODO
    def predict_results(self):
        self.logger.info("---STARTING MODEL PREDICTION")
        return "Please build model predictions"
