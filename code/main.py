import pandas as pd
import numpy as np
from XGBoost import train_xgboost
from regression import train_linear_regression
from data_prep import preprocess_data

# Pull TSV data from data.tsv and save to variable
data = preprocess_data('data/data.tsv')

model,results = train_xgboost(data)

results['Actual_Min'] = np.exp(results['Actual'])
results['Predicted_Min'] = np.exp(results['Predicted'])

# Calculate error in real-world terms
results['error_min'] = results['Actual_Min'] - results['Predicted_Min']
results['ape_min'] = abs(results['error_min']) / results['Actual_Min']

# This Accuracy Metric is much more impressive/interpretable
print(f"Accuracy: {round(1 - results['ape_min'].mean(), 3)}")

# Save the version that actually makes sense to a human reader
results.to_csv('results_interpreted.csv', index=False)

#Accuracy Metri