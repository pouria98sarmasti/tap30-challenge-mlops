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
Provides functionality for reading and parsing YAML configuration files.

Import `read_config` from this module to read YAML configuration files that
follow the project's configuration structure.
"""

from pathlib import Path

import yaml

from logger import get_logger

logger = get_logger(Path(__file__).name)


def read_config(config_path):
    """
    Reads configuration settings from a YAML file.

    This function reads a YAML configuration file from the specified path.
    It raises an error if the file does not exist or if there is a problem parsing
    the YAML content.

    Parameters:
    ----------
    config_path : str or Path
        The path to the YAML configuration file. This can be provided as a string or a `Path` object.
        The function will check if the file exists at this path.

    Returns:
    -------
    dict
        A dictionary containing the configuration settings extracted from the YAML file.

    Raises:
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    yaml.YAMLError
        If there is an error parsing the YAML file.

    Examples:
    --------
    >>> # Assuming 'config.yaml' is a valid YAML file in the current directory
    >>> config = read_config('config.yaml')
    >>> print(config)
    >>> # Output might be {'setting1': 'value1', 'setting2': 'value2'}

    Notes:
    -----
    - Ensure that the YAML file is properly formatted to avoid parsing errors.
    - The function uses `yaml.safe_load()` to prevent execution of arbitrary code.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at{config_path}")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise e
