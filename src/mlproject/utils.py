from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import os
import sys
from src.mlproject.exception import CustomExceptionHandler
from src.mlproject.logger import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import dagshub
import warnings
warnings.filterwarnings("ignore")

# connect dagshub & MLFlow!?
dagshub.init(repo_owner='amukherjee45nalhati',
             repo_name='mlproject', mlflow=True)

# call load_dotenv() function -> fetches the contents of the .env file
load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_database():
    logging.info("Establishing connection to the database.")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection to the database is established.")
        df = pd.read_sql_query("SELECT * FROM colleges.student", mydb)
        print(df.head())

        return df
    except Exception as e:
        CustomExceptionHandler(e.sys)


def save_object(file_path, obj):
    """
    Summary:
    This function saves any user defined object(arg) to the user defined filepath(arg).
    Args:
        file_path (_path_): _train data file path_
        obj (__any_object__): _any object which is to be saved_
    """
    try:
        # define filepath where the object needs to be saved
        dir_path = os.path.dirname(file_path)

        # make the directory to save the object
        os.makedirs(dir_path, exist_ok=True)

        # write pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    except Exception as e:
        CustomExceptionHandler.error_details(e, sys)


def evluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            grid = GridSearchCV(model, param, cv=3)
            grid.fit(X_train, y_train)

            model.set_params(**grid.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        CustomExceptionHandler.error_details(e, sys)
