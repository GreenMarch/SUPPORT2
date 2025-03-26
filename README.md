# SUPPORT2
Develop and validate a prognostic model that estimates survival over a 180-day period for seriously ill hospitalized adults (phase I of SUPPORT)

This Python code is part of a data science project that focuses on predicting medical charges (likely healthcare costs) using the SUPPORT2 dataset from the UCI Machine Learning Repository. Below is a detailed breakdown of what the regression code does:

1. Installation & Imports
The script starts by installing and importing necessary libraries:
Data Handling: pandas, numpy
Visualization: seaborn, matplotlib
Machine Learning: scikit-learn, xgboost, statsmodels
Data Fetching: ucimlrepo (to fetch the SUPPORT2 dataset)
Model Evaluation: Various metrics (accuracy_score, roc_auc_score, etc.)
Hyperparameter Tuning: RandomizedSearchCV

2. Data Loading & Preprocessing
Dataset: fetch_ucirepo(id=880) loads the SUPPORT2 dataset, which contains clinical data on seriously ill hospitalized patients.
Missing Values Handling:
Rows with missing values in the target variable (charges) are dropped.
The dataset is split into features (X) and target (y).
Feature Engineering:
Numerical and categorical features are separated (X_num_subset, X_cat_subset).
The target variable for regression (y_reg_charges) is extracted from X['charges'].

3. Model Training (XGBoost Regression)
Model: XGBRegressor (XGBoost for regression).
Hyperparameter Tuning:
RandomizedSearchCV is used to find the best hyperparameters (learning_rate, max_depth, n_estimators).
The search is optimized for minimizing Mean Squared Error (MSE).
Best Model:
The tuned model achieves:
Learning Rate: 0.0837
Max Depth: 5
Number of Estimators: 124
Training time: ~989 seconds.

4. Model Evaluation
Predictions:
The model predicts medical charges (y_pred).
Negative predictions are clipped to 0.
Predictions are rounded to integers.
Performance Metrics:
Correlation between predicted and true values: 0.855 (strong positive correlation).
Other regression metrics (like MAE, MSE, R²) are computed via regression_results() (though not fully shown in the output).
Feature Importance:
A horizontal bar plot shows which features most influence the predictions (e.g., age, num_co, slos, etc.).

5. Key Observations
Objective: Predict healthcare charges using patient clinical data.
Model Choice: XGBoost (a powerful ensemble method for regression).
Performance: The high correlation (0.855) suggests the model captures trends well, but further analysis (e.g., RMSE, residual plots) would be needed to assess accuracy fully.
Feature Importance: Helps identify which variables (e.g., length of stay, comorbidities) most impact cost predictions.

Potential Improvements
Handling Categorical Variables: The code separates them but doesn’t show encoding (e.g., one-hot encoding).
Cross-Validation: Only 3-fold CV is used; more folds could improve robustness.
Error Analysis: Residual plots or SHAP values could explain model behavior better.
Deployment: Saving the model (e.g., pickle) for future use.


This project is a regression task aimed at healthcare cost prediction, leveraging XGBoost for high accuracy and interpretability through feature importance.
