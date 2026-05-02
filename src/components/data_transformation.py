import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor_{tier}.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_feature_groups(self, model_tier):
        """
        Dynamically selects features based on the experimental tier (A, B, or C).
        """
        # Baseline Health & Socioeconomics (Model A)
        base_num = ['age', 'bmi', 'MAP', 'ldl', 'hba1c', 'chronic_count', 'medication_count', 'income', 'household_size', 'dependents']
        base_cat = ['sex', 'region', 'urban_rural', 'education', 'marital_status', 'employment_status', 'smoker', 'alcohol_freq']
        binary = ['hypertension', 'diabetes', 'asthma', 'copd', 'cardiovascular_disease', 'cancer_history', 'kidney_disease', 'liver_disease', 'arthritis', 'mental_health', 'is_high_risk']

        # Historical Utilization (Added in Model B)
        utilization = ['visits_last_year', 'hospitalizations_last_3yrs', 'days_hospitalized_last_3yrs']
        
        # Insurance Mechanics (Added in Model C)
        insurance_num = ['deductible', 'copay', 'policy_term_years', 'policy_changes_last_2yrs', 'provider_quality']
        insurance_cat = ['plan_type', 'network_tier']

        num_cols = base_num.copy()
        cat_cols = base_cat.copy()
        bin_cols = binary.copy()

        if model_tier in ['B', 'C']:
            num_cols.extend(utilization)
        if model_tier == 'C':
            num_cols.extend(insurance_num)
            cat_cols.extend(insurance_cat)

        return num_cols, cat_cols, bin_cols

    def get_data_transformer_object(self, model_tier):
        try:
            num_cols, cat_cols, bin_cols = self.get_feature_groups(model_tier)

            # Impute median and scale for numericals
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Impute unknown and encode for categoricals (sparse_output=False for dense array return)
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            logging.info(f"Tier {model_tier} initialized: {len(num_cols)} Num, {len(cat_cols)} Cat, {len(bin_cols)} Binary cols.")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipeline", cat_pipeline, cat_cols),
                ("binary_pipeline", 'passthrough', bin_cols) # Direct passthrough, no overengineering
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
            
    def engineer_features(self, df):
        try:
            # Safely engineer MAP if raw columns exist
            if all(col in df.columns for col in ['systolic_bp', 'diastolic_bp']):
                logging.info("Engineering MAP feature and dropping raw BP.")
                df['MAP'] = (2 * df['diastolic_bp'] + df['systolic_bp']) / 3
                df.drop(columns=['systolic_bp', 'diastolic_bp'], inplace=True)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, model_tier='C'):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            leakage_cols = [
                "person_id", "annual_premium", "monthly_premium", "total_claims_paid", 
                "avg_claim_amount", "claims_count", "risk_score", "proc_imaging_count", 
                "proc_surgery_count", "proc_physio_count", "proc_consult_count", 
                "proc_lab_count", "had_major_procedure"
            ]
            
            # Drop leakage columns silently ignoring if already dropped
            train_df = train_df.drop(columns=leakage_cols, errors='ignore')
            test_df = test_df.drop(columns=leakage_cols, errors='ignore')

            # Apply Feature Engineering
            train_df = self.engineer_features(train_df)
            test_df = self.engineer_features(test_df)

            target_column_name = "annual_medical_cost"
            
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            logging.info(f"Applying preprocessing object for Tier {model_tier}.")
            preprocessing_obj = self.get_data_transformer_object(model_tier)

            # Transform the data
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            # Robustly extract feature names post-transformation
            feature_names = preprocessing_obj.get_feature_names_out()

            # Save the preprocessor alongside its metadata for later interpretability
            save_path = self.data_transformation_config.preprocessor_obj_file_path.format(tier=model_tier)
            save_object(
                file_path=save_path, 
                obj={
                    "preprocessor": preprocessing_obj,
                    "features": list(feature_names),
                    "tier": model_tier
                }
            )
            logging.info(f"Saved preprocessing object and metadata to {save_path}.")

            # Return explicitly separated components, NOT a messy np.c_ array
            return X_train_transformed, X_test_transformed, y_train, y_test, feature_names
            
        except Exception as e:
            raise CustomException(e, sys)