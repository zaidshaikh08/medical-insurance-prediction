from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# 🔹 Minimal FULL Tier C input (all required features)
input_data = {
    "age": 45,
    "bmi": 28.5,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "ldl": 130,
    "hba1c": 5.5,
    "chronic_count": 1,
    "medication_count": 2,
    "income": 60000,
    "household_size": 3,
    "dependents": 1,

    "visits_last_year": 2,
    "hospitalizations_last_3yrs": 1,
    "days_hospitalized_last_3yrs": 3,

    "deductible": 500,
    "copay": 20,
    "policy_term_years": 2,
    "policy_changes_last_2yrs": 0,
    "provider_quality": 3,

    "sex": "Male",
    "region": "North",
    "urban_rural": "Urban",
    "education": "Graduate",
    "marital_status": "Single",
    "employment_status": "Employed",
    "smoker": "Never",
    "alcohol_freq": "Unknown",
    "plan_type": "Basic",
    "network_tier": "Tier1",

    "hypertension": 0,
    "diabetes": 0,
    "asthma": 0,
    "copd": 0,
    "cardiovascular_disease": 0,
    "cancer_history": 0,
    "kidney_disease": 0,
    "liver_disease": 0,
    "arthritis": 0,
    "mental_health": 0,
    "is_high_risk": 0
}

# 🔹 Convert to DataFrame
data = CustomData(input_data)
df = data.get_data_as_dataframe()

# 🔹 Load model and predict
pipeline = PredictPipeline(tier='C')
prediction = pipeline.predict(df)

print("\n✅ Prediction Successful")
print(f"Estimated Medical Cost: ${prediction[0]:,.2f}")