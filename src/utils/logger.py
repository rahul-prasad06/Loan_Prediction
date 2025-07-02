import logging
import os
from datetime import datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = f"{log_dir}/log_{datetime.now().strftime('%Y_%m_%d')}.log"

logging.basicConfig(
    filename=log_file,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)
