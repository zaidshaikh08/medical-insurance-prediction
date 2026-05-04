import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw dataset, performs a train-test split, saves the files 
        to the artifacts directory, and returns the paths for the transformation layer.
        """
        logging.info("Entered the data ingestion component")
        try:
            # Note: Ensure 'medical_insurance.csv' is in the root directory or adjust path accordingly
            df = pd.read_csv('medical_insurance.csv')
            logging.info('Successfully read the raw dataset into a Pandas DataFrame')

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data for record keeping
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Initiating Train-Test Split")
            # 80/20 split with a fixed random state for reproducibility
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)

# Optional: Simple test block to verify ingestion works independently
if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        print(f"Ingestion successful.\nTrain path: {train_data}\nTest path: {test_data}")
    except Exception as e:
        print(f"Error during ingestion: {e}")