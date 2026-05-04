import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self, tier='C'):
        self.tier = tier

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", f"model_{self.tier}.pkl")
            preprocessor_path = os.path.join("artifacts", f"preprocessor_{self.tier}.pkl")

            logging.info(f"Loading Tier {self.tier} model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor_data = load_object(file_path=preprocessor_path)
            
            # Safe unpacking of preprocessor object
            if isinstance(preprocessor_data, dict):
                preprocessor = preprocessor_data["preprocessor"]
            else:
                preprocessor = preprocessor_data

            # Hardened validation: Ensure all expected columns are present
            expected_cols = preprocessor.feature_names_in_
            missing = set(expected_cols) - set(features.columns)
            if missing:
                raise Exception(f"Form validation failed. Missing required columns: {missing}")

            logging.info("Transforming incoming web data...")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Executing prediction...")
            # TransformedTargetRegressor automatically handles the expm1 conversion back to real dollars
            preds = model.predict(data_scaled)
            
            return preds
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, form_data: dict):
        self.form_data = form_data

    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame([self.form_data])
            
            # Numeric columns expected by Model C (Prior to MAP engineering)
            numeric_cols = [
                'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c', 
                'chronic_count', 'medication_count', 'income', 'household_size', 
                'dependents', 'visits_last_year', 'hospitalizations_last_3yrs', 
                'days_hospitalized_last_3yrs', 'deductible', 'copay', 
                'policy_term_years', 'policy_changes_last_2yrs', 'provider_quality'
            ]
            
            # Binary indicators (HTML forms might send strings like "1" or "0", or checkboxes might send "on")
            binary_cols = [
                'hypertension', 'diabetes', 'asthma', 'copd', 'cardiovascular_disease', 
                'cancer_history', 'kidney_disease', 'liver_disease', 'arthritis', 
                'mental_health', 'is_high_risk'
            ]
            
            # Safe Coercion for continuous numeric variables
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        raise Exception(f"Invalid non-numeric value provided for {col}")
                        
            # Safe Coercion for binary variables (handles string '0'/'1' or HTML checkboxes)
            for col in binary_cols:
                if col in df.columns:
                    val = df[col].iloc[0]
                    if val in ['1', 1, 'True', 'on', True]:
                        df[col] = 1
                    else:
                        df[col] = 0
                else:
                    # If a checkbox isn't checked, HTML doesn't send it. Default to 0.
                    df[col] = 0

            # Engineer MAP on the fly
            if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
                df['MAP'] = (2 * df['diastolic_bp'] + df['systolic_bp']) / 3
                df.drop(columns=['systolic_bp', 'diastolic_bp'], inplace=True)

            logging.info("Successfully validated and formatted custom user data.")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)