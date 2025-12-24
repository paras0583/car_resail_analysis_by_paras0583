import os
import logging
from datetime import datetime

LOG_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_dir, exist_ok=True)

log_file = datetime.now().strftime("%Y_%m_%d_%H_%M_%S.log")
log_file_path = os.path.join(LOG_dir, log_file)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(filename)s : %(message)s"
)

logger = logging.getLogger(__name__)
