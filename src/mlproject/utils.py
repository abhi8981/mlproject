import os
import sys
from src.mlproject.exception import CustomExceptionHandler
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

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
