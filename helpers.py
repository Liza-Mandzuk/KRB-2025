import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def load_csv_data(filepath):
    return pd.read_csv(filepath)

def save_csv_data(data, filepath):
    data.to_csv(filepath, index=False)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def calculate_age_group(age):
    if age < 25:
        return 'young'
    elif age < 40:
        return 'middle'
    else:
        return 'senior'

def normalize_features(data, method='standard'):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    return scaler.fit_transform(data), scaler

def detect_outliers(data, method='iqr', factor=1.5):
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        return (data < lower) | (data > upper)
    return pd.Series([False] * len(data))

def get_risk_category(probability):
    for (lower, upper), category in [(0, 0.3, 'Low Risk'), (0.3, 0.6, 'Medium Risk'), (0.6, 1.0, 'High Risk')]:
        if lower <= probability < upper:
            return category
    return 'High Risk'