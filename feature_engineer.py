import pandas as pd
import numpy as np
from src.utils.helpers import calculate_age_group

class FeatureEngineer:
    def __init__(self):
        self.created_features = []
        
    def create_interaction_features(self, data):
        data_enhanced = data.copy()
        
        if 'screen_time' in data.columns and 'blink_frequency' in data.columns:
            data_enhanced['screen_blink_ratio'] = data['screen_time'] / (data['blink_frequency'] + 1)
            self.created_features.append('screen_blink_ratio')
            
        if 'age' in data.columns and 'stress_level' in data.columns:
            data_enhanced['age_stress_interaction'] = data['age'] * data['stress_level']
            self.created_features.append('age_stress_interaction')
            
        if 'sleep_quality' in data.columns and 'stress_level' in data.columns:
            data_enhanced['sleep_stress_score'] = data['sleep_quality'] / (data['stress_level'] + 1)
            self.created_features.append('sleep_stress_score')
            
        return data_enhanced
    
    def create_composite_features(self, data):
        data_enhanced = data.copy()
        
        lifestyle_columns = ['sleep_quality', 'physical_activity', 'stress_level']
        available_lifestyle = [col for col in lifestyle_columns if col in data.columns]
        
        if len(available_lifestyle) >= 2:
            data_enhanced['lifestyle_score'] = data[available_lifestyle].mean(axis=1)
            self.created_features.append('lifestyle_score')
            
        environment_columns = ['humidity', 'air_conditioner_use']
        available_environment = [col for col in environment_columns if col in data.columns]
        
        if len(available_environment) >= 1:
            data_enhanced['environment_risk'] = data[available_environment].mean(axis=1)
            self.created_features.append('environment_risk')
            
        return data_enhanced
    
    def create_categorical_features(self, data):
        data_enhanced = data.copy()
        
        if 'age' in data.columns:
            data_enhanced['age_group'] = data['age'].apply(calculate_age_group)
            data_enhanced = pd.get_dummies(data_enhanced, columns=['age_group'], prefix='age')
            
        if 'screen_time' in data.columns:
            data_enhanced['high_screen_time'] = (data['screen_time'] > 8).astype(int)
            self.created_features.append('high_screen_time')
            
        if 'blink_frequency' in data.columns:
            data_enhanced['low_blink_frequency'] = (data['blink_frequency'] < 15).astype(int)
            self.created_features.append('low_blink_frequency')
            
        return data_enhanced
    
    def create_polynomial_features(self, data, degree=2):
        data_enhanced = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns[:3]
        
        for col in numeric_columns:
            if col in data.columns:
                data_enhanced[f'{col}_squared'] = data[col] ** degree
                self.created_features.append(f'{col}_squared')
                
        return data_enhanced
    
    def engineer_features(self, data, create_interactions=True, create_composite=True, 
                         create_categorical=True, create_polynomial=False):
        engineered_data = data.copy()
        
        if create_interactions:
            engineered_data = self.create_interaction_features(engineered_data)
        if create_composite:
            engineered_data = self.create_composite_features(engineered_data)
        if create_categorical:
            engineered_data = self.create_categorical_features(engineered_data)
        if create_polynomial:
            engineered_data = self.create_polynomial_features(engineered_data)
            
        return engineered_data
    
    def get_created_features(self):
        return self.created_features