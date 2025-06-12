import pandas as pd
import numpy as np

class DataValidator:
    def __init__(self):
        self.validation_rules = {
            'age': {'min': 0, 'max': 120, 'type': 'numeric'},
            'screen_time': {'min': 0, 'max': 24, 'type': 'numeric'},
            'blink_frequency': {'min': 0, 'max': 50, 'type': 'numeric'},
            'sleep_quality': {'min': 1, 'max': 5, 'type': 'numeric'},
            'stress_level': {'min': 1, 'max': 5, 'type': 'numeric'},
            'physical_activity': {'min': 0, 'max': 300, 'type': 'numeric'},
            'humidity': {'min': 0, 'max': 100, 'type': 'numeric'},
            'gender': {'values': [0, 1, 'M', 'F'], 'type': 'categorical'}
        }
        
    def validate_data_types(self, data):
        errors = []
        
        for column in data.columns:
            if column in self.validation_rules:
                rule = self.validation_rules[column]
                if rule['type'] == 'numeric':
                    if not pd.api.types.is_numeric_dtype(data[column]):
                        errors.append(f"Column {column} should be numeric")
                        
        return errors
    
    def validate_ranges(self, data):
        errors = []
        
        for column in data.columns:
            if column in self.validation_rules:
                rule = self.validation_rules[column]
                if 'min' in rule and 'max' in rule:
                    invalid_values = data[
                        (data[column] < rule['min']) | (data[column] > rule['max'])
                    ]
                    if not invalid_values.empty:
                        errors.append(f"Column {column} has {len(invalid_values)} values outside range [{rule['min']}, {rule['max']}]")
                        
        return errors
    
    def validate_categorical_values(self, data):
        errors = []
        
        for column in data.columns:
            if column in self.validation_rules:
                rule = self.validation_rules[column]
                if 'values' in rule:
                    invalid_values = data[~data[column].isin(rule['values'])]
                    if not invalid_values.empty:
                        errors.append(f"Column {column} has invalid values: {invalid_values[column].unique()}")
                        
        return errors
    
    def check_missing_values(self, data, required_columns=None):
        errors = []
        
        if required_columns:
            for column in required_columns:
                if column not in data.columns:
                    errors.append(f"Required column {column} is missing")
                elif data[column].isnull().any():
                    missing_count = data[column].isnull().sum()
                    errors.append(f"Column {column} has {missing_count} missing values")
                    
        return errors
    
    def validate_data_quality(self, data):
        quality_issues = {}
        
        quality_issues['duplicates'] = data.duplicated().sum()
        quality_issues['missing_percentage'] = (data.isnull().sum() / len(data) * 100).to_dict()
        quality_issues['zero_variance'] = [col for col in data.select_dtypes(include=[np.number]).columns 
                                          if data[col].var() == 0]
        
        return quality_issues
    
    def validate_all(self, data, required_columns=None):
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_issues': {}
        }
        
        validation_results['errors'].extend(self.validate_data_types(data))
        validation_results['errors'].extend(self.validate_ranges(data))
        validation_results['errors'].extend(self.validate_categorical_values(data))
        validation_results['errors'].extend(self.check_missing_values(data, required_columns))
        validation_results['quality_issues'] = self.validate_data_quality(data)
        
        if validation_results['errors']:
            validation_results['is_valid'] = False
            
        return validation_results