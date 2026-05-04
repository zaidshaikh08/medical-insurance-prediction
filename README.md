# 🧠 Medical Insurance Cost Prediction (3-Tier ML System)

## 📌 Overview
This project builds a **regression-based machine learning system** to estimate individual medical insurance costs using health, behavioral, and policy-related features.

Unlike traditional single-model approaches, this project introduces a **3-Tier Experimental Architecture** to systematically analyze what actually drives medical costs.

---

## 🎯 Problem Statement
Predicting healthcare costs is inherently difficult due to:
- High variability (accidents, sudden illness)
- Hidden systemic factors
- Data leakage risks

This project focuses on:
- Predicting `annual_medical_cost`
- Eliminating leakage variables
- Understanding **what truly drives cost**

---

## 🧱 Project Architecture
medical_insurance_project/
│
├── src/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ │
│ ├── pipeline/
│ │ ├── training_pipeline.py
│ │ └── predict_pipeline.py
│ │
│ ├── logger.py
│ ├── exception.py
│ └── utils.py
│
├── notebooks/
│ ├── 1_EDA.ipynb
│ └── 2_Model_Training.ipynb
│
├── templates/
│ └── index.html
│
├── app.py
├── requirements.txt
└── setup.py



---

## 🔬 3-Tier Experimental Design

| Tier | Description | Purpose |
|------|------------|--------|
| **A** | Baseline Health | Biometrics + demographics |
| **B** | Utilization | Adds hospital visits & history |
| **C** | Insurance | Adds policy & financial structure |

---

## 📊 Results

| Tier | Best Model | R² Score | RMSE |
|------|-----------|----------|------|
| A | Ridge Regression | 0.056 | ~$3047 |
| B | Ridge Regression | 0.113 | ~$2953 |
| C | Ridge Regression | 0.113 | ~$2953 |

---

## 📈 Key Insights

### 1. Healthcare is inherently unpredictable
- Model explains ~11% variance
- Remaining cost driven by stochastic events

---

### 2. Behavior > Biology
- Adding utilization **doubled model performance**
- Past hospital usage is strongest predictor

---

### 3. Insurance ≠ Cost driver
- Insurance features added **no real predictive value**
- They affect payment, not actual medical cost

---

### 4. Simpler models win
- Ridge Regression outperformed XGBoost & Random Forest
- Indicates low signal-to-noise ratio

---

## ⚙️ Key Features

- ✅ Data Leakage Removal  
- ✅ Log Transformation (`log1p`)  
- ✅ Feature Engineering (MAP)  
- ✅ Modular ML Pipeline  
- ✅ Flask Web Deployment  
- ✅ Robust Input Handling (checkbox fix)

---

## 🌐 Web Application

A Flask-based UI allows real-time prediction.

📸 **UI Preview:**


---

## 🚀 How to Run
Clone repository
git clone https://github.com/zaidshaikh08/medical-insurance-prediction.git
cd medical-insurance-prediction

Install dependencies
pip install -r requirements.txt

Train models
python -m src.pipeline.training_pipeline

Run app
python app.py

Open browser
http://127.0.0.1:5000/

---

Target Transformation: Implemented log1p transformation to handle the severe right-skewness of medical costs found during EDA.

Feature Engineering: Engineered Mean Arterial Pressure (MAP) to resolve multicollinearity between Systolic and Diastolic blood pressure.

Hardened Deployment: The Flask UI includes a "Hidden Input" pattern for checkboxes to ensure consistent binary signal delivery to the model.

 
Results Summary
 Our experimental results indicated:
 
 A → B Gain: Adding utilization data significantly improves $R^2$, proving past behavior is a strong proxy for future risk.
 
 B → C Gain: Structural insurance mechanics provided negligible gain, suggesting billed costs are driven primarily by medical severity rather than policy design.

---

Contributors 
Mohammed Zaid Shaikh - Enrollment No: 220220202 (Contribution: 40)

Tahur Qureshi - Enrollment No. : 220220167 (Contribution: 20%)

Yash Devendra Mandal - Roll No: 220220192 (Contribution: 20%)

Pravin Mishra - Roll No: 220220 (Contribution: 10%)
