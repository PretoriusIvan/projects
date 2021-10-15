import logging
import pandas as pd
from app.config import config


class CleanDataClass:
    def __init__(self):
        self.logger = logging.getLogger(f'{config.LOGGER_NAME}')

    # TODO
    def data_cleaning_process(self):
        self.logger.info("--- Cleaning raw data ---")
        # df = pd.read_pickle('data/raw/df_raw.pkl')
        return "Data Cleaning completed"
