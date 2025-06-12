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
This module contains utilities for training the model.
Training includes: load data, build model, train model, and evaluate model.
For training use the run() method.
"""


from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from .logger import get_logger

logger = get_logger(Path(__file__).name)


class ModelTraining:

    def __init__(self, config):
        self.model_training_config = config["model_training"]
        artifact_dir = config["data_ingestion"]["artifact_dir"]
        self.processed_dir = Path(artifact_dir) / "processed"
        self.model_output_dir = Path(artifact_dir) / "model"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load training and validation data from processed directory.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            A tuple containing train_data and val_data pandas DataFrames.

        Examples
        --------
        >>> train_data, val_data = model_trainer.load_data()
        """
        self.train_path = self.processed_dir / "train.csv"
        self.val_path = self.processed_dir / "validation.csv"
        train_data = pd.read_csv(self.train_path)
        val_data = pd.read_csv(self.val_path)
        return train_data, val_data

    def build_model(self):
        """Build a RandomForestRegressor model with config parameters.

        Parameters
        ----------
        None

        Returns
        -------
        sklearn.ensemble.RandomForestRegressor
            A configured RandomForestRegressor model.

        Examples
        --------
        >>> model = model_trainer.build_model()
        """
        n_estimators = self.model_training_config["n_estimators"]
        max_samples = self.model_training_config["max_samples"]
        n_jobs = self.model_training_config["n_jobs"]
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_jobs=n_jobs,
            oob_score=root_mean_squared_error,
        )
        return model

    def train(self, model, train_data):
        """Train the model on the training data.

        Parameters
        ----------
        model : sklearn.ensemble.RandomForestRegressor
            The model to be trained.
        train_data : pandas.DataFrame
            Training data containing features and target.

        Returns
        -------
        None

        Examples
        --------
        >>> model = model_trainer.build_model()
        >>> model_trainer.train(model, train_data)
        """
        X_train, y_train = train_data.drop(columns=["demand"]), train_data["demand"]
        model.fit(X_train, y_train)

    def evaluate(self, model, val_data):
        """Evaluate the model on validation data.

        Parameters
        ----------
        model : sklearn.ensemble.RandomForestRegressor
            The trained model to evaluate.
        val_data : pandas.DataFrame
            Validation data containing features and target.

        Returns
        -------
        None
            Logs evaluation metrics to the logger.

        Examples
        --------
        >>> model_trainer.evaluate(model, val_data)
        >>> # Logs: Out-of-bag RMSE: 0.85
        >>> # Logs: RMSE for validation data: 0.92
        """
        X_val, y_val = val_data.drop(columns=["demand"]), val_data["demand"]
        y_pred = model.predict(X_val)
        y_pred = [round(y) for y in y_pred]

        self.rmse = root_mean_squared_error(y_val, y_pred)
        self.oob_score = model.oob_score_
        logger.info(f"Out-of-bag RMSE: {self.oob_score}")
        logger.info(f"RMSE for validation data: {self.rmse}")

    def save(self, model):
        """Save the trained model to the model output directory.

        Parameters
        ----------
        model : sklearn.ensemble.RandomForestRegressor
            The trained model to save.

        Returns
        -------
        None

        Examples
        --------
        >>> model_trainer.save(model)
        >>> # Logs: Model successfully saved to {self.model_output_dir}
        """

        self.model_output_path = self.model_output_dir / "rf.joblib"
        joblib.dump(model, self.model_output_path, compress=("lzma", 3))
        logger.info(f"Model successfully saved to {self.model_output_dir}")

    def run(self):
        """Run the complete model training pipeline.

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
        >>> model_trainer = ModelTraining(config)
        >>> model_trainer.run()
        >>> # Logs: Model Training started.
        >>> # Logs: Out-of-bag RMSE: 0.85
        >>> # Logs: RMSE for validation data: 0.92
        >>> # Logs: Model Training completed successfully.
        """

        mlflow.set_experiment("tap30-challenge-mlops")
        with mlflow.start_run():
            logger.info("Model Training started.")
            logger.info("MLflow started.")
            mlflow.set_tag("model_type", "Random_forest")

            train_data, val_data = self.load_data()
            mlflow.log_artifact(self.train_path, "datasets")
            mlflow.log_artifact(self.val_path, "datasets")

            model = self.build_model()
            self.train(model, train_data)
            self.evaluate(model, val_data)
            mlflow.log_metric("rmse", self.rmse)
            mlflow.log_metric("oob_score", self.oob_score)

            self.save(model)
            mlflow.log_artifact(self.model_output_path, "models")

            params = model.get_params()
            mlflow.log_params(params)

            logger.info("MLflow completed successfully.")
            logger.info("Model Training completed successfully.")
