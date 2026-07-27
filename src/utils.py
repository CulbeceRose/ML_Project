import os 
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
        
    try:
        reports = {}
        best_models = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            #model.fit(X_train, y_train)

            gs = GridSearchCV(model, para, cv = 3, scoring = "r2", n_jobs = -1, verbose = 1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            reports[list(models.keys())[i]] = test_model_score
            best_models[list(models.keys())[i]] = best_model
        
        return reports, best_models
    except Exception as e:
        raise CustomException(e, sys)
