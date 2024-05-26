import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproject.exception import CustomExceptionHandler
from src.mlproject.logger import logging

from src.mlproject.utils import save_object, evluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join(os.path.curdir, "artifacts")
    trained_model_fullpath = os.path.join(
        trained_model_filepath, "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing input data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2']
                },
                "Random Forest": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_features': ['sqrt', 'log2'],
                },
                "Gradient Boosting": {
                    'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_features': ['sqrt', 'log2'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'criterion': ['friedman_mse', 'squared_error']
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [.1, .01, .05, .001]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [.1, .01, .05, .001],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [.1, .01, .05, .001]
                }
            }
            model_report: dict = evluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # To get the best model score from the report
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"The best model is: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_fullpath, obj=best_model)

            predicted = best_model.predict(X_test)

            r2_score_pred = r2_score(y_test, predicted)
            return r2_score_pred

        except Exception as e:
            CustomExceptionHandler.error_details(e, sys)
