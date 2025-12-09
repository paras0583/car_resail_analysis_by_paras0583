import os
import logging
from datetime import datetime

# creating the log folder if it does not exist

os.makedirs(LOG_dir,exist_ok=True)

log_file  = f"{datetime.now().strftime('%Y-%m-%H-%M-%s')}.log"
log_file_path = os.path.join(LOG_dir,log_file)

# setting up the logging configuration 

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(filename)s %(message)s]"
)

logger = logging.getLogger(__name__)