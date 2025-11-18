import pandas as pd

def preprocess_data(file_path):

    # Load Data
    data = pd.read_csv(file_path, sep="\t")

    # Keep only work commutes (dpurp == 1) 
    data = data[data['dpurp'] == 1]

    # Create outcome variable: time = dtime - otime 
    data['time'] = data['dtime'] - data['otime']

    # Remove outliers 
    data = data[(data['time'] < 150) & (data['time'] > 30)]
    data = data[data['age'] < 100]

    # Define categorical variables
    categorical_cols = [
        'travday',     # travel day
        'opurp',       # origin purpose
        'mode',        # mode of transport
        'county',      # home county
        'workcnty',    # work county 
        'business',    # business type
        'occuptn',     # occupation code
        'vehicle'      # vehicle type
    ]

    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category')

    #  Choose model features 
    # 
    # time ~ mode + opurp + travday + tripno + vocc + bridge1 + bridge2 + county*workcnty + business + occuptn + vehicle
    feature_cols = [
        'mode', 'opurp', 'travday',
        'tripno', 'vocc',
        'bridge1', 'bridge2',
        'county', 'workcnty',    # interaction handled later via one-hot encoding
        'business', 'occuptn', 'vehicle'
    ]

    # Keep only rows that have all required columns
    data = data.dropna(subset=feature_cols + ['time'])

    # Prepare final dataset 
    X = data[feature_cols]
    y = data['time']

    # Add target column for your XGB function
    processed = X.copy()
    processed['target'] = y
    processed['dtime'] = data['dtime']   

    return processed