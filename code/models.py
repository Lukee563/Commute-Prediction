import pandas as pd
import numpy as np
from data_prep import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# --- 1. Load and Preprocess ---
# Assuming your preprocess_data function is defined above
df = preprocess_data('data/data.tsv')

# --- 2. Feature Engineering for Linear Models (Lasso/Ridge) ---
# Your paper mentions engineering interactions for county pairs
df_linear = df.copy()
df_linear['county_work_interaction'] = df_linear['county'].astype(str) + "_" + df_linear['workcnty'].astype(str)

# Convert categorical to Dummies
# Note: For Lasso/Ridge, we need one-hot encoding
df_linear = pd.get_dummies(df_linear, columns=[
    'mode', 'opurp', 'travday', 'county', 'workcnty', 
    'county_work_interaction', 'business', 'occuptn', 'vehicle'
], drop_first=True)

# Define X and y for Linear path
X_lin = df_linear.drop(columns=['target', 'otime'])
y_lin = df_linear['target']

# Feature Engineering for Non-Parametric (Random Forest)
df_rf = df.copy()
df_rf = pd.get_dummies(df_rf, columns=[
    'mode', 'opurp', 'travday', 'county', 'workcnty', 
    'business', 'occuptn', 'vehicle'
], drop_first=True)

X_rf = df_rf.drop(columns=['target', 'otime'])
y_rf = df_rf['target']

# Split and Scale
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_lin_scaled = scaler.fit_transform(X_train_lin)
X_test_lin_scaled = scaler.transform(X_test_lin)

#  Model Training & Evaluation 
# Lasso (L1) with Cross-Validation to find optimal Alpha
lasso = LassoCV(cv=20, random_state=42).fit(X_train_lin_scaled, y_train_lin)
y_pred_lasso = lasso.predict(X_test_lin_scaled)
lasso_r2 = r2_score(y_test_lin, y_pred_lasso)

# Ridge (L2) with Cross-Validation
ridge = RidgeCV(cv=20).fit(X_train_lin_scaled, y_train_lin)
y_pred_ridge = ridge.predict(X_test_lin_scaled)
ridge_r2 = r2_score(y_test_lin, y_pred_ridge)

# Random Forest (Non-Parametric)
rf = RandomForestRegressor(n_estimators=1000, min_samples_leaf=10, n_jobs=-1, random_state=42)
rf.fit(X_train_rf, y_train_rf)
y_pred_rf = rf.predict(X_test_rf)
rf_r2 = r2_score(y_test_rf, y_pred_rf)

# --- 6. Report Results ---
print(f"Lasso (L1) OOS R-squared: {lasso_r2:.4f}")
print(f"Ridge (L2) OOS R-squared: {ridge_r2:.4f}")
print(f"Random Forest OOS R-squared: {rf_r2:.4f}")

# Verification of non-linearity
improvement = (rf_r2 - lasso_r2) * 100
print(f"\nNon-parametric improvement over Linear: {improvement:.2f}%")