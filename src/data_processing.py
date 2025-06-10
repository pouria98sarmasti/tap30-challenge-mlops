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
This module contains utilities for processing the raw data.
Processing includes: load raw data, process data, and save to csv files.
For processing use the run() method.
"""


from pathlib import Path

import pandas as pd
from pandas._libs import parsers

from logger import get_logger

logger = get_logger(Path(__file__).name)


class DataProcessing:

    def __init__(self, config):
        self.data_processing_config = config["data_processing"]
        artifact_dir = config["data_ingestion"]["artifact_dir"]
        self.raw_dir = Path(artifact_dir) / "raw"

        self.processed_dir = Path(artifact_dir) / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self):
        """
        Load raw data from CSV files.

        Returns
        -------
        train_data : pd.DataFrame
            Training dataset loaded from 'train.csv'.
        val_data : pd.DataFrame
            Validation dataset loaded from 'val.csv'.
        test_data : pd.DataFrame
            Test dataset loaded from 'test.csv'.

        Examples
        --------
        >>> processor = DataProcessing(config)
        >>> train_data, val_data, test_data = processor.load_raw_data()
        """
        train_data = pd.read_csv(self.raw_dir / "train.csv")
        val_data = pd.read_csv(self.raw_dir / "validation.csv")
        test_data = pd.read_csv(self.raw_dir / "test.csv")
        return train_data, val_data, test_data

    def process_data(self, train_data, val_data, test_data):
        """
        Process the raw data by transforming time features and sorting.

        Parameters
        ----------
        train_data : pd.DataFrame
            Raw training dataset.
        val_data : pd.DataFrame
            Raw validation dataset.
        test_data : pd.DataFrame
            Raw test dataset.

        Returns
        -------
        processed_train_data : pd.DataFrame
            Processed training dataset with transformed features.
        processed_val_data : pd.DataFrame
            Processed validation dataset with transformed features.
        processed_test_data : pd.DataFrame
            Processed test dataset with transformed features.

        Examples
        --------
        >>> processor = DataProcessing(config)
        >>> train_data, val_data, test_data = processor.load_raw_data()
        >>> processed_train, processed_val, processed_test = processor.process_data(train_data, val_data, test_data)
        """
        processed_train_data = self._process_single_data(train_data)
        processed_val_data = self._process_single_data(val_data)
        processed_test_data = self._process_single_data(test_data)
        return processed_train_data, processed_val_data, processed_test_data

    def save_to_csv_files(
        self, processed_train_data, processed_val_data, processed_test_data
    ):
        """
        Save processed data to CSV files.

        Parameters
        ----------
        processed_train_data : pd.DataFrame
            Processed training dataset to be saved.
        processed_val_data : pd.DataFrame
            Processed validation dataset to be saved.
        processed_test_data : pd.DataFrame
            Processed test dataset to be saved.

        Returns
        -------
        None

        Examples
        --------
        >>> processor = DataProcessing(config)
        >>> processed_train, processed_val, processed_test = processor.process_data(train_data, val_data, test_data)
        >>> processor.save_to_csv_files(processed_train, processed_val, processed_test)
        """
        column_headers = ["hour_of_day", "day", "row", "col", "demand"]

        processed_train_data = processed_train_data[column_headers]
        processed_val_data = processed_val_data[column_headers]
        processed_test_data = processed_test_data[column_headers]

        processed_train_data.to_csv(self.processed_dir / "train.csv", index=False)
        processed_val_data.to_csv(self.processed_dir / "val.csv", index=False)
        processed_test_data.to_csv(self.processed_dir / "test.csv", index=False)

        logger.info(f"Processed data saved to {self.processed_dir}")

    def _process_single_data(self, data):
        """
        Process a single dataset by sorting and transforming time features.

        Parameters
        ----------
        data : pd.DataFrame
            Raw dataset to be processed.

        Returns
        -------
        pd.DataFrame
            Processed dataset with transformed time features.

        Examples
        --------
        >>> processor = DataProcessing(config)
        >>> train_data = pd.read_csv("raw/train.csv")
        >>> processed_train = processor._process_single_data(train_data)
        """
        data = data.sort_values(["time", "row", "col"])
        data["time"] = data["time"] + self.data_processing_config["shift"]
        data = data.assign(
            hour_of_day=data["time"] % 24,
            day=data["time"] // 24,
        )
        data = data.drop(columns=["time"])
        return data

    def run(self):
        """
        Execute the complete data processing pipeline.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> from config_reader import read_config
        >>> config = read_config("config/config.yaml")
        >>> processor = DataProcessing(config)
        >>> processor.run()
        """
        logger.info("Data Processing strted.")
        train_data, val_data, test_data = self.load_raw_data()
        processed_train_data, processed_val_data, processed_test_data = (
            self.process_data(train_data, val_data, test_data)
        )
        self.save_to_csv_files(
            processed_train_data, processed_val_data, processed_test_data
        )
        logger.info("Data Processing completed successfully.")
