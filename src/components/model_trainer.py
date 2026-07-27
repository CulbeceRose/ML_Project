import os
import sys
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params = {

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "sqrt", "log2"]
                },

                "Random Forest Regressor": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200, 300],
                    "subsample": [0.8, 0.9, 1.0],
                    "max_depth": [3, 5],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "Linear Regression": {},

                "XGBRegressor": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 0.1, 0.3]
                },

                "CatBoostRegressor": {
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200, 300],
                    "l2_leaf_reg": [1, 3, 5, 7]
                },

                "AdaBoost Regressor": {
                    "learning_rate": [0.01, 0.05, 0.1, 0.5],
                    "n_estimators": [50, 100, 200],
                    "loss": ["linear", "square", "exponential"]
                },
                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10]
                },

                "Ridge": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10]
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                }

            }
            model_report: dict=evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_score_data = r2_score(y_test, predicted)
            return r2_score_data, 
        
        except Exception as e:
            raise CustomException(e, sys)