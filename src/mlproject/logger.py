import logging
import os
from datetime import datetime

# get current path
curr_dir = os.path.curdir
log_path = os.path.join(curr_dir,'logs')
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

full_filepath = os.path.join(log_path, LOG_FILE)

FORMAT = '[ %(asctime)s ] %(lineno)d %(name)s - %(filename)s - %(levelname)s - %(message)s'

# make directory using os.makedirs()
os.makedirs(log_path, exist_ok=True)

logging.basicConfig(
    filename= full_filepath,
    format= FORMAT,
    level= logging.INFO
)