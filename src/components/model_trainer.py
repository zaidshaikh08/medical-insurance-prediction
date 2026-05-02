import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import TransformedTargetRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Enforce global determinism
np.random.seed(42)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model_{tier}.pkl")
    evaluation_report_file_path = os.path.join("artifacts", "evaluation_report_{tier}.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        try:
            report = {}
            y_test_max = y_test.max()
            y_test_min = y_test.min()

            # Pre-calculate log targets for dual-space evaluation
            y_test_log = np.log1p(y_test)
            y_train_log = np.log1p(y_train)

            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]

                logging.info(f"Training {model_name}...")
                model.fit(X_train, y_train)

                # Predict (Outputs are automatically converted back to real dollars via expm1)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Real-World Space Metrics ($)
                train_r2 = r2_score(y_train, y_train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                
                test_r2 = r2_score(y_test, y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

                # Log Space Metrics (Model Optimization space - approximated via re-log)
                y_test_pred_log = np.log1p(y_test_pred)
                test_rmse_log = np.sqrt(mean_squared_error(y_test_log, y_test_pred_log))

                # Custom Selection Score: R2 penalized by Normalized RMSE
                rmse_norm = test_rmse / (y_test_max - y_test_min)
                selection_score = test_r2 - rmse_norm

                # Overfitting Guardrail
                r2_gap = train_r2 - test_r2
                if r2_gap > 0.20:
                    logging.warning(f"SEVERE Overfitting in {model_name}: Train R2={train_r2:.4f}, Test R2={test_r2:.4f} (Gap: {r2_gap:.4f})")
                elif r2_gap > 0.10:
                    logging.info(f"Mild Overfitting in {model_name}: Train R2={train_r2:.4f}, Test R2={test_r2:.4f}")

                report[model_name] = {
                    "Test_R2": test_r2,
                    "Train_R2": train_r2,
                    "Test_RMSE_Real": test_rmse,
                    "Train_RMSE_Real": train_rmse,
                    "Test_RMSE_Log": test_rmse_log,
                    "Selection_Score": selection_score,
                    "Model_Object": model
                }
                
                # Explicitly log the full comparison metrics for debugging and reporting
                logging.info(f"--- {model_name} Metrics ---")
                logging.info(f"Test R2: {test_r2:.4f} | Train R2: {train_r2:.4f}")
                logging.info(f"Test RMSE: ${test_rmse:.2f} | Train RMSE: ${train_rmse:.2f}")
                logging.info(f"Score: {selection_score:.4f}\n")

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, model_tier='C'):
        try:
            logging.info(f"Initializing Model Training for Tier {model_tier}")

            # Define base models with production-grade configurations
            base_models = {
                "Ridge Regression": Ridge(alpha=1.0, random_state=42),
                "Random Forest": RandomForestRegressor(
                    n_estimators=200, 
                    max_depth=None, # Relaxed depth
                    min_samples_split=5, # Controlled overfit
                    random_state=42, 
                    n_jobs=-1
                ),
                "XGBoost": XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            }

            models = {}
            for name, estimator in base_models.items():
                models[name] = TransformedTargetRegressor(
                    regressor=estimator,
                    func=np.log1p,
                    inverse_func=np.expm1
                )

            # Generate evaluation report
            model_report = self.evaluate_models(X_train, y_train, X_test, y_test, models)

            # Select the best model based on the Custom Selection Score
            best_model_name = max(model_report, key=lambda k: model_report[k]["Selection_Score"])
            best_model_data = model_report[best_model_name]

            if best_model_data["Test_R2"] < 0.4:
                raise CustomException(f"No acceptable model found for Tier {model_tier}. Best R2: {best_model_data['Test_R2']}", sys)

            logging.info(
                f"=== Tier {model_tier} Winner: {best_model_name} ==="
            )

            # Save the winning model
            save_path = self.model_trainer_config.trained_model_file_path.format(tier=model_tier)
            save_object(file_path=save_path, obj=best_model_data["Model_Object"])

            # Save the full evaluation report for this tier
            report_path = self.model_trainer_config.evaluation_report_file_path.format(tier=model_tier)
            
            # Clean out the model objects before saving the report dictionary to disk
            clean_report = {k: {key: val for key, val in v.items() if key != "Model_Object"} for k, v in model_report.items()}
            save_object(file_path=report_path, obj=clean_report)

            return {
                "Tier": model_tier,
                "Best Model": best_model_name,
                "Test R2": best_model_data["Test_R2"],
                "Test RMSE Real": best_model_data["Test_RMSE_Real"],
                "Test RMSE Log": best_model_data["Test_RMSE_Log"]
            }

        except Exception as e:
            raise CustomException(e, sys)