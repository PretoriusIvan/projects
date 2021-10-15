import logging
import pandas as pd
from app.config import config


class FeatureEngineeringClass:

    def __init__(self):
        self.logger = logging.getLogger(f'{config.LOGGER_NAME}')

    # TODO
    def feature_engineering(self):
        self.logger.info("----- Engineering Features ------")
        # df = pd.read_pickle('data/interim/df_clean.pkl')
        return "Feature Engineering Completed"
