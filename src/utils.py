import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves a Python object (like a trained model or preprocessor) to a specified file path.
    Automatically creates the directory structure if it doesn't exist.
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        # Ensure the artifacts directory exists before saving
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Successfully saved object at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a pickled Python object from a specified file path.
    Used heavily by the predict_pipeline to retrieve models and preprocessors.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"Failed to load: The file path {file_path} does not exist.")
            
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            
        logging.info(f"Successfully loaded object from {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)