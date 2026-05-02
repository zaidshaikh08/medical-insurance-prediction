# Medical Insurance Cost Prediction

## Project Overview

This project predicts individual medical insurance costs using machine learning models based on health, demographic, and behavioral data.

## Objective

To analyze how different factors influence medical cost:

* Health indicators (BMI, BP, chronic conditions)
* Past medical utilization (visits, hospitalizations)
* Insurance policy structure

## Model Architecture

We designed a 3-tier experimental system:

* **Tier A:** Baseline Health Model
* **Tier B:** + Utilization Features
* **Tier C:** + Insurance Features

## Results Summary

| Tier | Best Model    | R² Score | RMSE |
| ---- | ------------- | -------- | ---- |
| A    | (your result) | ...      | ...  |
| B    | (your result) | ...      | ...  |
| C    | (your result) | ...      | ...  |

## Key Insights

* Utilization data improves prediction accuracy significantly.
* Insurance structure contributes to cost variation.
* Health indicators alone are not sufficient to fully explain cost.

## Tech Stack

* Python
* Scikit-learn
* XGBoost
* Flask (for deployment)

## How to Run

```bash
pip install -r requirements.txt
python src/pipeline/train_pipeline.py
```

## Author

* Shaikh Zaid
