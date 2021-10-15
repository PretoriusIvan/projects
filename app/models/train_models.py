import logging
from app.config import config


class ModelTrainingClass:

    def __init__(self):
        self.logger = logging.getLogger(f'{config.LOGGER_NAME}')

    # TODO
    def model_training(self):
        self.logger.info("---STARTING MODEL TRAINING")
        # df = pd.read_pickle('data/processed/df_features.pkl')
        return "Model training completed"
