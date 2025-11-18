import pandas as pd
import numpy as np
from model import train_xgboost
from data_prep import preprocess_data

# Pull TSV data from data.tsv and save to variable
data = preprocess_data('data/data.tsv')

model,results = train_xgboost(data)

#Calculate error metrics
results['error'] = results['Actual'] - results['Predicted']
results = results[results.Actual != 0]
results['ape'] = abs(results['error']) / results['Actual']

#Accuracy Metric
print(1 - results['ape'].mean())

results.to_csv('results.csv', index=False)


