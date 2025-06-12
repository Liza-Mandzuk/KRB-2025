import pandas as pd
import numpy as np
from src.utils.helpers import load_csv_data
from config.settings import RAW_DATA_DIR, DEFAULT_FEATURES
import os

class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(RAW_DATA_DIR, 'dry_eye_dataset.csv')
        self.data = None
        self.features = DEFAULT_FEATURES
        
    def load_data(self):
        self.data = load_csv_data(self.data_path)
        return self.data
    
    def get_features_and_target(self, target_column='dry_eye_disease'):
        if self.data is None:
            self.load_data()
            
        available_features = [f for f in self.features if f in self.data.columns]
        X = self.data[available_features]
        y = self.data[target_column] if target_column in self.data.columns else None
        
        return X, y
    
    def get_patient_data(self, patient_data_dict):
        return pd.DataFrame([patient_data_dict])
    
    def validate_columns(self, required_columns):
        if self.data is None:
            return False
        return all(col in self.data.columns for col in required_columns)
    
    def get_data_info(self):
        if self.data is None:
            return None
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }