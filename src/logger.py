# Copyright 2025 Pouria Sarmasti

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides a pre-configured logger for consistent logging across the application.

Import `get_logger` from this module in your own modules to obtain a
logger instance that automatically writes timestamped messages (INFO level
and above) to a daily log file located in the 'logs/' directory.
"""


import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / f"log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log"


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)


def get_logger(name):
    """
    Retrieves a logger instance with the specified name.

    This function provides a convenient way to obtain a logger object from the
    Python logging module. The logger can be used for logging messages at different
    severity levels.

    Parameters:
    ----------
    name : str
        The name of the logger. This name is used to identify the logger and can be
        any string. Loggers with the same name will share configuration and handlers.

    Returns:
    -------
    logging.Logger
        A logger object with the specified name. This logger can be used to log
        messages at various severity levels such as DEBUG, INFO, WARNING, ERROR, and CRITICAL.

    Examples:
    --------
    >>> from logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.error('This is an error message.')
    """
    return logging.getLogger(name)
