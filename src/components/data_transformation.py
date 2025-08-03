import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

# ==========================================================
# 📌 DATA TRANSFORMATION CONFIG CLASS
# ==========================================================
@dataclass
class DataTransformationConfig:
    # This will store the preprocessing pipeline object.
    # We'll save it in artifacts so we can load it later in training or inference
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


# ==========================================================
# 📌 MAIN DATA TRANSFORMATION CLASS
# ==========================================================
class DataTransformation:
    def __init__(self):
        # Load the config so we know where to save our preprocessor object
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define which columns are numeric & which are categorical
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # ================================
            # 🟢 NUMERICAL PIPELINE
            # ================================
            # 1️⃣ Fill missing numbers with the median
            # 2️⃣ Standardize (mean=0, std=1) so model trains better

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # ================================
            # 🔵 CATEGORICAL PIPELINE
            # ================================
            # 1️⃣ Fill missing categories with most_frequent (like "Male"/"Female")
            # 2️⃣ Convert text → numbers using OneHotEncoder
            # 3️⃣ Scale but don’t center (with_mean=False, since one-hot creates sparse data)

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        ✅ Reads train & test CSVs, applies preprocessing pipeline, 
        and saves the pipeline object for later use.
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("✅ Read train and test data completed")

            # ================================
            # 🔧 STEP 2: Get preprocessing object
            # ================================
            preprocessing_obj = self.get_data_transformer_object()

            # ================================
            # 🎯 STEP 3: Separate features & target
            # ================================
            target_column_name = "math_score"   # The column we want to predict
            numerical_columns = ["writing_score", "reading_score"]

            # Drop target column from input features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("🔄 Applying preprocessing object on train & test data.")

            # Fit on train (learn how to scale/encode)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # Only transform test (no fitting → prevents data leakage)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # ================================
            # 📝 STEP 5: Merge transformed features + target
            # ================================
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # ================================
            # 💾 STEP 6: Save preprocessor for future use
            # ================================
            logging.info(f"💾 Saving preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return transformed train, test, and file path for preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    # 1️⃣ First run data ingestion to get train/test files
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2️⃣ Then run data transformation
    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)

    print("✅ Data Transformation Complete!")
    print(f"🔹 Preprocessor saved at: {preprocessor_path}")
    print(f"🔹 Train Array Shape: {train_arr.shape}")
    print(f"🔹 Test Array Shape: {test_arr.shape}")
    print("\n🎉 Data Ingestion + Transformation Pipeline Completed Successfully!")

# ✅ In Jupyter Notebook:
#    - We usually experiment, test ideas, visualize, debug.
#    - Great for learning and exploration.
#
# ✅ In Python Scripts (like this one):
#    - Code is reusable (no need to re-run cells manually).
#    - Can run pipelines end-to-end (like a production system).
#    - Easier to integrate with CI/CD, deployment, or automation.
#    - Keeps code modular (data_ingestion.py, data_transformation.py, etc.).
#    - Any other team member can run the same script → same results.
#
# 👉 Notebook is for thinking. Scripts are for building.
#    This transformation pipeline will later be used by:
#       🔹 Model training
#       🔹 Model deployment
#       🔹 Real-world predictions