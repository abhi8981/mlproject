# import logging from logger module
from src.mlproject.logger import logging
from src.mlproject.exception import CustomExceptionHandler
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformationConfig, DataTransformation
import sys
from src.mlproject.components.model_trainer import ModelTrainerConfig, ModelTrainer
# generate log
if __name__ == "__main__":
    logging.info("Successfully executed logging.")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_array, test_array, _, _ = data_transformation.initiate_data_transformation(
            train_path=train_data_path, test_path=test_data_path)

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(
            train_array=train_array, test_array=test_array))

    except Exception as e:
        # Call error_details method on the class itself
        logging.info(f"Exception: {e} has occured.")
        CustomExceptionHandler.error_details(e, sys)
