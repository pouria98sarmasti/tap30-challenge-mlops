from pathlib import Path

from logger import get_logger

logger = get_logger(Path(__file__).name)

logger.info("Start the App")
logger.error("an error occured")
