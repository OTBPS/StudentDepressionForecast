# Student Depression Prediction Based on Machine Learning

A data-driven approach for early identification of student depression risk

## Author Information

Author: Peng Bo  
Role: Project Leader / Full-Stack ML Developer  
Affiliation: NUIST (Waterford Institute)  
Purpose: Research showcase, portfolio for graduate program applications  
Accepted by: AIMDL Conference, Code: AIDML-6291

## Project Overview

This project aims to develop an interpretable and generalizable machine learning prediction system to assess depression risk among students based on psychological and behavioral data. The model integrates traditional machine learning methods with interpretability tools to ensure both performance and transparency, supporting mental health screening efforts in educational institutions.

## My Core Contributions

\- Data preprocessing and cleaning (handling missing values, encoding, normalization)  
\- Design and tuning of three supervised learning models: Logistic Regression, Random Forest, XGBoost  
\- Model evaluation using metrics including Accuracy, Precision, Recall, F1-score, and AUC, with additional ROC curve, calibration curve, and robustness testing  
\- Model interpretability via SHAP value analysis  
\- Visualization and reporting using matplotlib and seaborn  
\- Full pipeline implementation using pandas, scikit-learn, xgboost, shap  
\- Research paper writing in IEEE style (abstract, model description, experimental analysis, references)  

## Dataset Information

Source: Kaggle Student Depression Dataset  
Samples: 27,901 Indian students  
Features: 17 depression-related variables (e.g., gender, financial stress, academic pressure, suicidal thoughts, sleep duration)

## Key Results

Model performance is summarized as follows:

| Model | AUC | F1-score | Remarks |
| --- | --- | --- | --- |
| Logistic Regression | 0.913 | 0.8627 | Best calibration and robustness |
| Random Forest | 0.906 | 0.8552 | \-  |
| XGBoost | 0.903 | 0.8521 | \-  |

Top 3 SHAP features: Suicidal thoughts, financial stress, academic pressure

## Project Structure

📦DepressionForecast  
┣ 📄 Forecast.py ← Main program  
┣ 📄 Cleaned_Depression_Dataset.csv  
┣ 📊 /figures ← SHAP, ROC, and Confusion Matrix images  
┗ 📄 Final_Model_Report.xlsx  

## Future Work

\- Introduce additional models (e.g., LightGBM, SVM) and explore multi-model fusion  
\- Expand dataset coverage to enhance cross-regional generalizability  
\- Deploy models into a Web or App platform for real-time prediction and recommendation  

## Contact

Email: <cnpengbo@outlook.com>  
Full PDF report available upon request