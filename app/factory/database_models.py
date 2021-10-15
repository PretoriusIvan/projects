import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import event
import time
import logging
from app.config_secrets import secrets
from app.config import config


class DatabaseModelsClass:
    def __init__(self, database_name, server_name):
        self.logger = self.logger = logging.getLogger(f'{config.LOGGER_NAME}')

        server_name = server_name.upper()
        server = secrets.DATABASE[server_name]['SERVER_IP']
        username = secrets.DATABASE[server_name]['USERNAME']
        password = secrets.DATABASE[server_name]['PASSWORD']

        driver_str2 = 'driver=ODBC Driver 17 for SQL Server'
        self.alchemy_engine = create_engine('mssql+pyodbc://{}:{}@{}/{}?{}'.format(username,
                                                                                   password,
                                                                                   server,
                                                                                   database_name,
                                                                                   driver_str2))

    def select_query(self, query_str):
        self.logger.info("Select values from sql")
        data_set = pd.read_sql(query_str, con=self.alchemy_engine,
                               index_col=None,
                               coerce_float=True,
                               params=None,
                               parse_dates=None,
                               columns=None,
                               chunksize=None)

        self.logger.info("Done with Select")
        return data_set

    def select_query_chunks(self, query_str, chunk_size):
        data_set = pd.DataFrame({})
        valid_response = False

        while not valid_response:
            try:
                i = 1
                for chunk in pd.read_sql_query(query_str, self.alchemy_engine, chunksize=chunk_size):
                    start_time = time.time()
                    data_set = data_set.append(chunk)
                    self.logger.info("Number of rows {}".format(data_set.shape[0]))
                    self.logger.info("Loop {} took {} seconds".format(i, (round((time.time() - start_time)*100, 2))))
                    i += 1
                valid_response = True

                valid_response = True
            except:
                self.logger.exception("No response from database for given select")
        self.logger.info("Import of data completed for {} rows".format(data_set.shape[0]))
        return data_set

    def insert_table(self, data, table_name, schema, if_exists, index=False):

        self.logger.info("{} values to SQL".format(if_exists))
        conn = self.alchemy_engine
        @event.listens_for(self.alchemy_engine, "before_cursor_execute")
        def receive_before_cursor_execute(
                conn, cursor, statement, params, context, executemany
        ):
            if executemany:
                cursor.fast_executemany = True

        data.to_sql(table_name, schema=schema, con=conn,
                    if_exists=if_exists, index=index)
        self.logger.info("Done with Insert")

    def insert_table_chunks(self, data, table_name, schema, if_exists, chunk_size, index=False):
        self.logger.info("{} values to SQL".format(if_exists))
        conn = self.alchemy_engine
        data.to_sql(table_name, schema=schema, con=conn,
                    if_exists=if_exists, chunksize=chunk_size, index=index)
        self.logger.info("Done with Insert")

    def delete_table(self, query):
        self.logger.info("Deleting from")
        conn = self.alchemy_engine.connect()
        conn.execute(query)
        conn.close()
        self.logger.info("Done Deleting from")
