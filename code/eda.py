# Exploratory Data Analysis of the SF Commute Survey Data
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/data.tsv', sep='\t')
data.to_csv("Data.csv")

#Pre-Trimming
plt.scatter(data['otime'],data['dtime'])
plt.title(f"Original Survey Data, n = {data.shape[0]}")
plt.ylabel("Anonymized Destination Time")
plt.xlabel("Anonymized Departure Time")
plt.show()


#Post-Trimming
# Create outcome variable: time = dtime - otime 
data['time'] = data['dtime'] - data['otime']

# Remove outliers 
data = data[(data['time'] < 150) & (data['time'] > 40)]
data = data[data['age'] < 100]

plt.scatter(data['otime'],data['dtime'])
plt.title(f"Trips Between 1 Minutes and 250 Minutes, n = {data.shape[0]}")
plt.ylabel("Anonymized Destination Time")
plt.xlabel("Anonymized Departure Time")
plt.show()