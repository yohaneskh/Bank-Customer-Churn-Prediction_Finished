# Bank-Customer-Churn-Prediction_Finished

This repository contains a complete notebook for predicting bank customer churn using a **Supervised Machine Learning** approach. The notebook walks through the process from data loading to final model evaluation, with a clear and structured pipeline including EDA, preprocessing, and modeling.

### A. **Objective**

The main goal of this project is to build and evaluate a classification model that can predict whether a customer is likely to churn based on their profile and banking behavior.

### B. **DATASET**

i. The dataset is sourced from Kaggle and contains 10,000 customer records from a European bank. 
The link to the original dataset on Kaggle is https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction.

ii. It includes features such as geography, gender, credit score, balance, tenure, estimated salary, etc.

iii. Target variable: Exited (1 = churned, 0 = retained)

iv. The raw CSV file has been stored in this repository and loaded via GitHub’s raw URL: **https://raw.githubusercontent.com/yohaneskh/Bank-Customer-Churn-Prediction_Finished/refs/heads/main/Churn_Modelling.csv**

### C. **Steps Performed in the Notebook**

1. Importing required libraries.

2. Version upgrade (optional), you may need to install or upgrade the following:
- !pip install -U scikit-learn
- !pip install -U imbalanced-learn
- !pip install -U xgboost

3. Loading the Dataset.

4. Exploratory Data Analysis (EDA) like:
- Class imbalance check.
- Correlation heatmap.
- Churn distribution.

5. Data preprocessing.
- One-hot encoding for Geography and Gender.
- Feature scaling with StandardScaler.
  
6. Train-Test split.
  
7. Model building & evaluation.
- Baseline vs. Tuned Model.
- Using SMOTE for imbalanced data handling.
- Classification report and confusion matrix.
  
8. Comparison with other models.
   
9. ROC AUC evaluation.

### D. **Machine Learning Techniques Used**

i. XGBoost Classifier (primary model).

ii. SMOTE (for oversampling minority class).

iii. GridSearchCV (hyperparameter tuning).

iv. OneHotEncoding and StandardScaler for data preprocessing.

v. Metrics: Recall, F1-score, Precision, ROC-AUC.

### E. **Evaluation Metrics Summary**

The models are evaluated using:

i. Confusion matrix.

ii. Classification report.

iii. ROC curve and AUC score.

### F. **Environment**

This notebook was developed and tested on **Kaggle Notebook**, for environments where dependencies are not pre-installed, use the optional !pip install commands at the top of the notebook.

## **AUTHOR**

**Yohanes Kurniawan Hertanto**

An aspiring Data Analyst with interest in Machine Learning

https://www.kaggle.com/yohaneskhyekaha
