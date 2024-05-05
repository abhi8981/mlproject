"""
    - Data fetch from MySQL -> local system
    - train - test split
    - Data ingestion inputs:
        1. Dataframe coming from the data source
        2. User defined os paths to store train & test data
"""

import os
import sys
from src.mlproject.exception import CustomExceptionHandler
from src.mlproject.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.mlproject.utils import read_database
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:

    data_files_path = os.path.join(os.path.curdir, "artifacts")
    train_data_path = os.path.join(data_files_path, "train.csv")
    test_data_path = os.path.join(data_files_path, "test.csv")
    raw_data_path = os.path.join(data_files_path, "raw.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # read the data from server
            logging.info("Reading data from SQL database..")
            df = read_database()
            logging.info("Data succesfully read from MySQL server.")

            os.makedirs(self.data_ingestion_config.data_files_path,
                        exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,
                      index=False, header=True)
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,
                             index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info("Data ingestion complete!")

            return (self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path)

        except Exception as e:
            CustomExceptionHandler.error_details(e, sys)
