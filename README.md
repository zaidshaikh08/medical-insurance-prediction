Project Overview : This project focuses on the Regression-Based Estimation of Individual Medical Insurance Charges using health indicators. Unlike standard projects that use a single model, this repository implements a 3-Tier Experimental Architecture to decompose medical costs into biological, behavioral, and systemic components.The objective is to predict annual_medical_cost accurately while maintaining strict data integrity by eliminating "post-outcome" leakage variables.

Project Architecture : The project follows a modular, industry-standard pipeline structure as required:

medical_insurance_project/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading & splitting
│   │   ├── data_transformation.py # Tiered feature engineering & preprocessing
│   │   └── model_trainer.py       # Multi-model training & evaluation
│   ├── pipeline/
│   │   ├── training_pipeline.py   # Orchestrates the A/B/C experiment
│   │   └── predict_pipeline.py    # Handles web-based inference
│   ├── logger.py                  # Custom logging for all stages
│   ├── exception.py               # Hardened error handling
│   └── utils.py                   # Model/Preprocessor I/O
├── notebooks/
│   ├── 1_EDA.ipynb                # Statistical analysis & visualizations
│   └── 2_Model_Training.ipynb     # Initial model prototyping
├── templates/
│   └── index.html                 # Modern Flask-based UI
├── application.py                 # Flask entry point
├── requirements.txt               # Dependencies
└── setup.py                       # Package configuration



The 3-Tier Experiment
We analyzed the dataset through three distinct lenses to measure Information Gain:
Tier A (Baseline Health): Focuses solely on biometrics (BMI, Age, BP) and demographics.

Tier B (Utilization Proxy): Adds historical data (past hospitalizations, doctor visits) to measure behavioral risk.

Tier C (Insurance Mechanics): Adds policy-specific variables (deductibles, plan types) to measure systemic cost influence.



Key Features & Highlights
Data Leakage Prevention: Strictly removed columns like total_claims_paid and claims_count that are not available before a prediction is needed.

Target Transformation: Implemented log1p transformation to handle the severe right-skewness of medical costs found during EDA.

Feature Engineering: Engineered Mean Arterial Pressure (MAP) to resolve multicollinearity between Systolic and Diastolic blood pressure.

Hardened Deployment: The Flask UI includes a "Hidden Input" pattern for checkboxes to ensure consistent binary signal delivery to the model.



Installation & Usage
Clone the Repository:
git clone https://github.com/zaidshaikh08/medical-insurance-prediction.git
cd medical_insurance_project

Install Dependencies: 
pip install -r requirements.txt

Run Training Pipeline:
This will execute the A/B/C experiment and save the winning models in artifacts/.
python -m src.pipeline.training_pipeline

Launch Web App: 
    python app.py

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the cost predictor.

 
 
Results Summary
 Our experimental results indicated:
 A → B Gain: Adding utilization data significantly improves $R^2$, proving past behavior is a strong proxy for future risk.

 B → C Gain: Structural insurance mechanics provided negligible gain, suggesting billed costs are driven primarily by medical severity rather than policy design.



Contributors 
Mohammed Zaid Shaikh - Enrollment No: 220220202 (Contribution: 40)

Tahur Qureshi - Enrollment No. : 220220167 (Contribution: 20%)

Yash Devendra Mandal - Roll No: 220220192 (Contribution: 20%)

Pravin Mishra - Roll No: 220220 (Contribution: 10%)

