import sys
import time
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting Multi-Tier Experiment Pipeline")

            # 1. Data Ingestion
            logging.info("Initiating Data Ingestion Phase")
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            tiers = ['A', 'B', 'C']
            experiment_results = {}

            for tier in tiers:
                logging.info(f"========== Processing Tier {tier} ==========")
                start_time = time.time()
                
                # 2. Data Transformation
                transformer = DataTransformation()
                X_train, X_test, y_train, y_test, feature_names = transformer.initiate_data_transformation(
                    train_path, 
                    test_path, 
                    model_tier=tier
                )

                # 3. Model Training & Evaluation
                trainer = ModelTrainer()
                tier_result = trainer.initiate_model_trainer(
                    X_train, 
                    y_train, 
                    X_test, 
                    y_test, 
                    model_tier=tier
                )
                
                # Add metadata to the result dictionary
                tier_result['Feature Count'] = len(feature_names)
                
                end_time = time.time()
                tier_result['Execution Time (s)'] = round(end_time - start_time, 2)
                
                experiment_results[tier] = tier_result
                logging.info(f"Tier {tier} completed successfully in {tier_result['Execution Time (s)']} seconds.\n")

            # 4. Save Final Aggregated Report
            save_object("artifacts/final_experiment_results.pkl", experiment_results)
            logging.info("Final experiment results saved to artifacts/final_experiment_results.pkl")

            # 5. Output Interpreted Report to Console
            print("\n" + "="*50)
            print(" FINAL EXPERIMENT RESULTS ".center(50))
            print("="*50)
            
            # Explicit ordering ensures A -> B -> C printing
            for tier in ['A', 'B', 'C']:
                data = experiment_results[tier]
                print(f"Tier {tier} | Best Model: {data['Best Model']}")
                print(f"  -> Features Used: {data['Feature Count']}")
                print(f"  -> Test R2:       {data['Test R2']:.4f}")
                print(f"  -> Real RMSE:     ${data['Test RMSE Real']:,.2f}")
                print(f"  -> Log RMSE:      {data['Test RMSE Log']:.4f}")
                print("-" * 50)

            # 6. Automatic Insights Calculation
            print("\n[ EXPERIMENT INSIGHTS ]")
            best_tier = max(experiment_results, key=lambda t: experiment_results[t]['Test R2'])
            print(f">> Best Performing Architecture: Tier {best_tier} ({experiment_results[best_tier]['Best Model']})")

            print("\n>> Incremental Value Analysis (Information Gain):")
            gain_AB = experiment_results['B']['Test R2'] - experiment_results['A']['Test R2']
            gain_BC = experiment_results['C']['Test R2'] - experiment_results['B']['Test R2']
            total_gain = experiment_results['C']['Test R2'] - experiment_results['A']['Test R2']

            print(f"  A -> B (Utilization Value): +{gain_AB:.4f} R2")
            print(f"  B -> C (Insurance Value):   +{gain_BC:.4f} R2")
            print(f"  Total Information Gain:     +{total_gain:.4f} R2\n")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()