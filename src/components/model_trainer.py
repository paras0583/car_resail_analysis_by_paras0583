import os
import sys
import numpy as np

from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logging import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model trainer")

            # ===============================
            # DEV MODE: SAMPLE REDUCTION
            # ===============================
            MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", 3000))

            if train_array.shape[0] > MAX_SAMPLES:
                train_array = train_array[:MAX_SAMPLES]
                logging.info(f"Reduced training samples to {MAX_SAMPLES}")

            if test_array.shape[0] > MAX_SAMPLES // 4:
                test_array = test_array[: MAX_SAMPLES // 4]
                logging.info("Reduced test samples")

            # ===============================
            # SPLIT FEATURES & TARGET
            # ===============================
            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info(
                f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}"
            )

            # ===============================
            # MODELS
            # ===============================
            models = {
                # "Linear Regression": LinearRegression(),
                # "Decision Tree": DecisionTreeRegressor(),
                # "Random Forest": RandomForestRegressor(),
                # "Gradient Boosting": GradientBoostingRegressor(),
                # "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # ===============================
            # HYPERPARAMETERS
            # ===============================
            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ]
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "subsample": [0.7, 0.8, 0.9],
                    "n_estimators": [16, 32, 64],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [16, 32, 64],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8],
                    "learning_rate": [0.05, 0.1],
                    "iterations": [50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [16, 32, 64],
                },
            }

            # ===============================
            # MODEL EVALUATION
            # ===============================
            model_report = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info(
                f"Best model found: {best_model_name} with score {best_model_score}"
            )

            # ===============================
            # SAVE MODEL
            # ===============================
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # ===============================
            # FINAL METRIC
            # ===============================
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square, best_model

        except Exception as e:
            raise CustomException(e, sys)

