import os
import sys
from dataclasses import dataclass

# ✅ ML Models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score      # ✅ To check how well the model predicts
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor          # ✅ Powerful gradient boosting library

# ✅ Custom project imports
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models   # ✅ Utility functions


# ======================================================
# 📌 CONFIG CLASS → Tells where to save trained model
# ======================================================
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  
    # ✅ All trained models will be saved here


# ======================================================
# 📌 MODEL TRAINER CLASS
# ======================================================
class ModelTrainer:
    def __init__(self):
        # ✅ Load config so we know where to save model
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        ✅ Main function:
        - Splits data into features (X) & labels (y)
        - Trains many models with hyperparameter tuning
        - Finds the BEST model
        - Saves the best model for future use
        - Returns R² score of the best model
        """
        try:
            logging.info("📊 Splitting training and test input data")

            # 1️⃣ Split arrays into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],   # All columns except last → features
                train_array[:, -1],    # Last column → target
                test_array[:, :-1],
                test_array[:, -1]
            )

            # 2️⃣ Define candidate models to test
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # 3️⃣ Define hyperparameters for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},  # ✅ No hyperparameters to tune
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # 4️⃣ Train & evaluate all models
            logging.info("🚀 Starting model evaluation with GridSearchCV...")
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # 5️⃣ Get best model from report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # 6️⃣ Check if best model is good enough
            if best_model_score < 0.6:
                raise CustomException("❌ No suitable model found (score < 0.6)")
            logging.info(f"🏆 Best model found: {best_model_name} with R² score: {best_model_score}")

            # 7️⃣ Save the best model as model.pkl
            logging.info("💾 Saving best model...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # 8️⃣ Make predictions on test set
            predicted = best_model.predict(X_test)

            # 9️⃣ Calculate final R² score
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            logging.error(f"🔥 Error during model training: {e}")
            raise CustomException(e, sys)


# ======================================================
# 📌 DEBUG-FRIENDLY MAIN BLOCK
# ======================================================
if __name__ == "__main__":
    try:
        print("🚀 Starting FULL ML pipeline (Ingestion → Transformation → Training)...")

        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation

        # 1️⃣ Run data ingestion
        print("📥 Running Data Ingestion...")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        print(f"✅ Data Ingestion Done: {train_path}, {test_path}")

        # 2️⃣ Run data transformation
        print("🔄 Running Data Transformation...")
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)
        print(f"✅ Data Transformation Done. Train shape: {train_arr.shape}, Test shape: {test_arr.shape}")

        # 3️⃣ Run model training
        print("🤖 Training models (this may take a while)...")
        trainer = ModelTrainer()
        r2 = trainer.initiate_model_trainer(train_arr, test_arr)

        # 4️⃣ Final output
        print("✅✅✅ Model Training Complete! ✅✅✅")
        print(f"🎯 Final R² Score: {r2}")
        print(f"💾 Model saved to: {trainer.model_trainer_config.trained_model_file_path}")

    except Exception as e:
        print("🔥 PIPELINE CRASHED!")
        print(f"DETAILS: {e}")