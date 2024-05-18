import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.mlproject.exception import CustomExceptionHandler
from src.mlproject.logger import logging
from src.mlproject.utils import read_database
from src.mlproject.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join(os.path.curdir, "artifacts")
    preprocessor_obj_fullpath = os.path.join(
        preprocessor_obj_filepath, "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Summary:
        This function is responsible for data transformation with the help of the following pipelines & columntransformers:
            - numerical_pipeline -> [SimpleImputer(strategy="median") + StandardScaler()]
            - cat_pipeline -> [SimpleImputer(strategy="mode") + OneHotEncoder() + StandardScaler()]
            - preprocessor -> ColumnTransformer([numerical_pipeline] + [cat_pipeline])
        """
        try:
            # read the data from server
            logging.info(
                "Reading data from SQL database for data transformation.")
            # df = read_database()
            # Get the directory of the current script (data_transformation.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct the path to the raw.csv file
            raw_data_path = os.path.join(
                current_dir, '..', 'notebook', 'data', 'raw.csv')
            # read the data
            df = pd.read_csv(raw_data_path)
            logging.info("Data succesfully read from MySQL server.")

            # define numerical & categorical columns
            # numeric_features = [
            #     feature for feature in df.columns if df[feature].dtype != 'O']
            # categorical_features = [
            #     feature for feature in df.columns if df[feature].dtype == 'O']
            numeric_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # define data processing pipelines for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scalar", StandardScaler())
            ]
            )
            # define data processing pipelines for categorical features
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scalar", StandardScaler(with_mean=False))
            ]
            )
            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numeric_features}")

            # combine numerical & categorical features engineering pipelines using ColumnTransformers
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numeric_features),
                    ("categorical_pipeline", cat_pipeline, categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            CustomExceptionHandler.error_details(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Summary:
        This method fetches the train & test files from the specified path & applys data transformation on them.
        Args:
            train_path (_path_): _train data file path_
            test_path (_path_): _test data file path_
        """

        try:
            # make train & test dataframes from the specified filepaths
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            logging.info("Reading traing & test files.")

            # create an instance of the method get_data_transformer_object() which returns a preprocessor
            preprocessor_obj = self.get_data_transformer_object()

            # define numeric/categorical & features
            target_feature = "math_score"
            numeric_features = ["writing_score", "reading_score"]

            # bifurcate input features & target feature in train data
            input_features_train_df = df_train.drop(
                columns=[target_feature], axis=1)
            target_feature_train_df = df_train[target_feature]

            # bifurcate input features & target feature in test data
            input_features_test_df = df_test.drop(
                columns=[target_feature], axis=1)
            target_feature_test_df = df_test[target_feature]

            logging.info("Applying preprocessing on train & test data.")

            # fit-transform preprocessor pipeline object on training data
            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_features_train_df, target_feature_train_df)

            # fit-transform preprocessor pipeline object on test data
            input_feature_test_arr = preprocessor_obj.transform(
                input_features_test_df)

            # combine input & output features to get training array
            train_arr = np.c_[
                input_feature_train_arr, np.array(
                    target_feature_train_df
                )
            ]

            # combine input & output features to get test array
            test_arr = np.c_[
                input_feature_test_arr, np.array(
                    target_feature_test_df
                )
            ]

            # save the preprocessor_obj
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_fullpath,
                obj=preprocessor_obj
            )

            logging.info("Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath,
                self.data_transformation_config.preprocessor_obj_fullpath
            )

        except Exception as e:
            CustomExceptionHandler.error_details(e, sys)
