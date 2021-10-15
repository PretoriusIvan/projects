from app.config import config
from app.controllers.main_controller import MainControllerCass
from app.utils.logger_util import create_logger

# ----------- APP CONFIG AND LOGGING ----------------
# TODO
create_logger(config.LOGGER_NAME)

# TeamsMessenger.send_message(title='Message Title',
# text="""<h1 style='color: #44c47c;'><strong>STARTED</strong></h1>Message that indicates start of script run""",
# webhook="webhook from config file")

# Run Scripts
if __name__ == '__main__':
    main_run = MainControllerCass()
    main_run.ml_pipeline_controller()
