import os
import sys
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score           
from sklearn.model_selection import GridSearchCV  
import dill       # ✅ Used for advanced object serialization (optional alternative to pickle)
import pickle     # ✅ Standard Python module for saving/loading objects
from src.exception import CustomException

# ==========================================================
# ✅ SAVE OBJECT FUNCTION
# ==========================================================
def save_object(file_path, obj):
    """
    Saves ANY Python object (like a model, preprocessor, pipeline, etc.)
    to a file path using pickle.

    🔹 file_path → where to save the object (e.g., 'artifacts/preprocessor.pkl')
    🔹 obj       → the Python object you want to save
    """
    try:
        # 1️⃣ Get directory from the given path (e.g., 'artifacts/')
        dir_path = os.path.dirname(file_path)

        # 2️⃣ Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # 3️⃣ Open the file in write-binary mode and save object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)  # Save the object (model/preprocessor/etc.)

    except Exception as e:
        # Custom error handling to catch and trace any issues during saving
        raise CustomException(e, sys)

# ==========================================================
# ✅ EVALUATE MULTIPLE MODELS FUNCTION
# ==========================================================
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains & evaluates multiple models using GridSearchCV.

    🔹 X_train, y_train → training data
    🔹 X_test, y_test   → testing data
    🔹 models           → dictionary of model names & instances 
    🔹 param            → dictionary of hyperparameter grids

    🔁 For each model:
        - Perform hyperparameter tuning using GridSearchCV
        - Train best model
        - Predict on train/test
        - Calculate R2 score on test set
        - Store results in report dictionary

    🔁 Returns: dictionary of model_name → R2 test score
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # 🧪 GridSearchCV to find best hyperparameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # 🧠 Train model with best found parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # 🧮 Predictions & scores
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # 📊 Save test R2 score in report
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

# ==========================================================
# ✅ LOAD OBJECT FUNCTION
# ==========================================================
def load_object(file_path):
    """
    Loads a previously saved object using pickle.

    🔹 file_path → path to the file (e.g., 'artifacts/preprocessor.pkl')
    🔁 Returns: loaded Python object (model, preprocessor, etc.)
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)