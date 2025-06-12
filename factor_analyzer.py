import pandas as pd
import numpy as np
from src.utils.constants import FACTOR_WEIGHTS
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

class FactorAnalyzer:
    def __init__(self):
        self.factor_importance = {}
        self.factor_correlations = {}
        self.risk_factors = []
        
    def analyze_feature_importance(self, model, feature_names=None):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if feature_names:
                self.factor_importance = dict(zip(feature_names, importances))
            else:
                self.factor_importance = {f'feature_{i}': imp for i, imp in enumerate(importances)}
        return self.factor_importance
    
    def calculate_correlations(self, X, y):
        correlations = {}
        
        for column in X.columns if hasattr(X, 'columns') else range(X.shape[1]):
            if hasattr(X, 'columns'):
                feature_data = X[column]
                feature_name = column
            else:
                feature_data = X[:, column]
                feature_name = f'feature_{column}'
                
            try:
                corr_coef, p_value = pearsonr(feature_data, y)
                correlations[feature_name] = {
                    'correlation': corr_coef,
                    'p_value': p_value,
                    'strength': self._interpret_correlation(abs(corr_coef))
                }
            except:
                correlations[feature_name] = {
                    'correlation': 0,
                    'p_value': 1,
                    'strength': 'none'
                }
                
        self.factor_correlations = correlations
        return correlations
    
    def _interpret_correlation(self, corr_value):
        if corr_value >= 0.7:
            return 'strong'
        elif corr_value >= 0.5:
            return 'moderate'
        elif corr_value >= 0.3:
            return 'weak'
        else:
            return 'negligible'
    
    def identify_risk_factors(self, X, y, threshold=0.1):
        self.calculate_correlations(X, y)
        
        self.risk_factors = [
            factor for factor, data in self.factor_correlations.items()
            if abs(data['correlation']) >= threshold and data['p_value'] < 0.05
        ]
        
        return self.risk_factors
    
    def get_top_factors(self, n=10, by='importance'):
        if by == 'importance' and self.factor_importance:
            sorted_factors = sorted(self.factor_importance.items(), 
                                  key=lambda x: x[1], reverse=True)
        elif by == 'correlation' and self.factor_correlations:
            sorted_factors = sorted(self.factor_correlations.items(),
                                  key=lambda x: abs(x[1]['correlation']), reverse=True)
        else:
            return []
            
        return sorted_factors[:n]
    
    def analyze_factor_interactions(self, X, top_features=None):
        if top_features is None:
            top_features = [item[0] for item in self.get_top_factors(5)]
            
        interactions = {}
        
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                if hasattr(X, 'columns') and feat1 in X.columns and feat2 in X.columns:
                    try:
                        corr, p_val = pearsonr(X[feat1], X[feat2])
                        interactions[f'{feat1}_x_{feat2}'] = {
                            'correlation': corr,
                            'p_value': p_val,
                            'strength': self._interpret_correlation(abs(corr))
                        }
                    except:
                        continue
                        
        return interactions
    
    def get_factor_summary(self, patient_data, feature_names=None):
        summary = {}
        
        if feature_names is None:
            feature_names = list(self.factor_importance.keys()) if self.factor_importance else []
            
        for feature in feature_names:
            if feature in patient_data:
                value = patient_data[feature]
                importance = self.factor_importance.get(feature, 0)
                correlation_data = self.factor_correlations.get(feature, {})
                
                summary[feature] = {
                    'value': value,
                    'importance': importance,
                    'correlation': correlation_data.get('correlation', 0),
                    'risk_contribution': importance * abs(correlation_data.get('correlation', 0)),
                    'weight': FACTOR_WEIGHTS.get(feature, 0.1)
                }
                
        return summary
    
    def calculate_risk_score(self, patient_data, weights=None):
        if weights is None:
            weights = FACTOR_WEIGHTS
            
        risk_score = 0
        total_weight = 0
        
        for feature, weight in weights.items():
            if feature in patient_data:
                value = patient_data[feature]
                importance = self.factor_importance.get(feature, 0.1)
                
                normalized_value = min(max(value / 10, 0), 1)
                factor_risk = normalized_value * importance * weight
                
                risk_score += factor_risk
                total_weight += weight
                
        return risk_score / total_weight if total_weight > 0 else 0