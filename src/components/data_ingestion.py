import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException     # Your custom exception class
from src.logger import logging                # Your logging setup

# ================================
# 1️⃣ CONFIG CLASS FOR FILE PATHS
# ================================
@dataclass
class DataIngestionConfig:
    """Holds file paths for train, test, and raw datasets."""
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


# ================================
# 2️⃣ MAIN DATA INGESTION CLASS
# ================================
class DataIngestion:
    """Handles dataset reading, saving, and splitting into train/test."""

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """Reads dataset, saves raw copy, splits into train/test, and saves them."""
        logging.info("Entered the data ingestion method.")
        print("🚀 Starting Data Ingestion...")

        try:
            # STEP 1: Read dataset from CSV
            data_path = 'notebook/data/stud.csv'   # ✅ Ensure this file exists
            logging.info(f"Reading dataset from: {data_path}")
            print(f"📄 Reading dataset from: {data_path}")

            df = pd.read_csv(data_path)
            logging.info(f"Dataset read successfully with shape {df.shape}")
            print(f"✅ Dataset loaded! Shape: {df.shape}")

            # STEP 2: Create artifacts folder (if not exists)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("Artifacts directory created.")

            # STEP 3: Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")
            print(f"💾 Raw data saved to: {self.ingestion_config.raw_data_path}")

            # STEP 4: Split into train/test
            logging.info("Performing train-test split...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            print(f"📊 Train set: {train_set.shape}, Test set: {test_set.shape}")

            # STEP 5: Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test datasets saved.")
            print(f"✅ Train data saved to: {self.ingestion_config.train_data_path}")
            print(f"✅ Test data saved to: {self.ingestion_config.test_data_path}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# ================================
# 3️⃣ EXECUTION BLOCK
# ================================
if __name__ == "__main__":
    """Run this file directly: python -m src.components.data_ingestion"""
    obj = DataIngestion()
    logging.info("Starting the data ingestion pipeline.")
    train_data, test_data = obj.initiate_data_ingestion()
    print("✅ Data ingestion completed successfully!")