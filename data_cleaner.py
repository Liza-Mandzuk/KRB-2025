import pandas as pd
import numpy as np
from src.utils.helpers import detect_outliers
from src.utils.constants import FEATURE_MAPPINGS
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        
    def handle_missing_values(self, data, strategy='median'):
        data_clean = data.copy()
        
        for column in data_clean.columns:
            if data_clean[column].dtype in ['int64', 'float64']:
                if strategy == 'median':
                    fill_value = data_clean[column].median()
                else:
                    fill_value = data_clean[column].mean()
                data_clean[column] = data_clean[column].fillna(fill_value)
            else:
                data_clean[column] = data_clean[column].fillna(data_clean[column].mode()[0])
                
        return data_clean
    
    def remove_duplicates(self, data):
        return data.drop_duplicates()
    
    def handle_outliers(self, data, method='iqr', factor=1.5):
        data_clean = data.copy()
        
        for column in data_clean.select_dtypes(include=[np.number]).columns:
            outliers = detect_outliers(data_clean[column], method, factor)
            if method == 'remove':
                data_clean = data_clean[~outliers]
            elif method == 'cap':
                Q1 = data_clean[column].quantile(0.25)
                Q3 = data_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - factor * IQR
                upper = Q3 + factor * IQR
                data_clean[column] = data_clean[column].clip(lower, upper)
                
        return data_clean
    
    def encode_categorical(self, data):
        data_encoded = data.copy()
        
        for column, mapping in FEATURE_MAPPINGS.items():
            if column in data_encoded.columns:
                data_encoded[column] = data_encoded[column].map(mapping)
                
        for column in data_encoded.select_dtypes(include=['object']).columns:
            data_encoded = pd.get_dummies(data_encoded, columns=[column], prefix=column)
            
        return data_encoded
    
    def normalize_features(self, data, fit=True):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data_normalized = data.copy()
        
        if fit and not self.fitted:
            data_normalized[numeric_columns] = self.scaler.fit_transform(data[numeric_columns])
            self.fitted = True
        elif self.fitted:
            data_normalized[numeric_columns] = self.scaler.transform(data[numeric_columns])
        else:
            return data_normalized
            
        return data_normalized
    
    def clean_pipeline(self, data, remove_duplicates=True, handle_missing=True, 
                      handle_outliers=True, encode_categorical=True, normalize=True):
        cleaned_data = data.copy()
        
        if remove_duplicates:
            cleaned_data = self.remove_duplicates(cleaned_data)
        if handle_missing:
            cleaned_data = self.handle_missing_values(cleaned_data)
        if encode_categorical:
            cleaned_data = self.encode_categorical(cleaned_data)
        if handle_outliers:
            cleaned_data = self.handle_outliers(cleaned_data)
        if normalize:
            cleaned_data = self.normalize_features(cleaned_data)
            
        return cleaned_data