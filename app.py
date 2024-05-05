# import logging from logger module
from src.mlproject.logger import logging
from src.mlproject.exception import CustomExceptionHandler
from src.mlproject.components.data_ingestion import DataIngestion
import sys

# generate log
if __name__ == "__main__":
    logging.info("Successfully executed logging.")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        # Call error_details method on the class itself
        logging.info(f"Exception: {e} has occured.")
        CustomExceptionHandler.error_details(e, sys)
