import logging
import pandas as pd
from app.config import config
from app.factory.database_models import DatabaseModelsClass


class GetDataModelClass:
    def __init__(self):
        self.logger = logging.getLogger(f'{config.LOGGER_NAME}')

    # TODO
    def import_raw_data(self):
        self.logger.info("--- Get raw data ---")
        # TODO
        sql_query_str = """ENTER QUERY HERE"""
        database_instance = DatabaseModelsClass(database_name='DATABASE NAME', server_name='SERVER NAME')
        df = database_instance.select_query_chunks(query_str=sql_query_str, chunk_size=10000)
        df.to_pickle('data/raw/df_raw_{}.pkl')

        self.logger.info("Raw data imported")

