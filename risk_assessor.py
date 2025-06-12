import numpy as np
import pandas as pd
from src.utils.helpers import get_risk_category
from src.utils.constants import RISK_CATEGORIES
from config.settings import RISK_THRESHOLD

class RiskAssessor:
    def __init__(self, predictor=None):
        self.predictor = predictor
        self.risk_threshold = RISK_THRESHOLD
        
    def assess_individual_risk(self, patient_data):
        if self.predictor is None:
            raise ValueError("Predictor model not initialized")
            
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
            
        risk_probability = self.predictor.predict_risk_probability(patient_df)[0]
        risk_binary = self.predictor.predict_risk(patient_df)[0]
        risk_category = get_risk_category(risk_probability)
        
        return {
            'risk_probability': risk_probability,
            'risk_binary': risk_binary,
            'risk_category': risk_category,
            'confidence': self._calculate_confidence(risk_probability)
        }
    
    def assess_population_risk(self, population_data):
        if self.predictor is None:
            raise ValueError("Predictor model not initialized")
            
        risk_probabilities = self.predictor.predict_risk_probability(population_data)
        risk_predictions = self.predictor.predict_risk(population_data)
        
        risk_distribution = {}
        for prob in risk_probabilities:
            category = get_risk_category(prob)
            risk_distribution[category] = risk_distribution.get(category, 0) + 1
            
        return {
            'mean_risk': np.mean(risk_probabilities),
            'risk_distribution': risk_distribution,
            'high_risk_count': np.sum(risk_predictions),
            'total_population': len(population_data),
            'high_risk_percentage': np.mean(risk_predictions) * 100
        }
    
    def _calculate_confidence(self, probability):
        distance_from_threshold = abs(probability - self.risk_threshold)
        confidence = min(distance_from_threshold * 2, 1.0)
        
        if confidence >= 0.8:
            return 'High'
        elif confidence >= 0.5:
            return 'Medium'
        else:
            return 'Low'
    
    def compare_risk_factors(self, patient_data, baseline_data=None):
        individual_risk = self.assess_individual_risk(patient_data)
        
        if baseline_data is not None:
            population_risk = self.assess_population_risk(baseline_data)
            relative_risk = individual_risk['risk_probability'] / population_risk['mean_risk']
        else:
            relative_risk = 1.0
            
        return {
            'individual_risk': individual_risk,
            'relative_risk': relative_risk,
            'risk_percentile': self._calculate_risk_percentile(
                individual_risk['risk_probability'], baseline_data
            ) if baseline_data is not None else None
        }
    
    def _calculate_risk_percentile(self, individual_risk, baseline_data):
        if baseline_data is None or self.predictor is None:
            return None
            
        baseline_risks = self.predictor.predict_risk_probability(baseline_data)
        percentile = (np.sum(baseline_risks <= individual_risk) / len(baseline_risks)) * 100
        return percentile
    
    def assess_temporal_risk(self, patient_history):
        if len(patient_history) < 2:
            return None
            
        risk_trend = []
        for data in patient_history:
            risk = self.assess_individual_risk(data)
            risk_trend.append(risk['risk_probability'])
            
        return {
            'risk_trend': risk_trend,
            'trend_direction': 'increasing' if risk_trend[-1] > risk_trend[0] else 'decreasing',
            'trend_magnitude': abs(risk_trend[-1] - risk_trend[0]),
            'volatility': np.std(risk_trend)
        }
    
    def generate_risk_alerts(self, patient_data, alert_thresholds=None):
        if alert_thresholds is None:
            alert_thresholds = {
                'high_risk': 0.7,
                'medium_risk': 0.4,
                'rapid_increase': 0.2
            }
            
        risk_assessment = self.assess_individual_risk(patient_data)
        alerts = []
        
        if risk_assessment['risk_probability'] >= alert_thresholds['high_risk']:
            alerts.append({
                'level': 'HIGH',
                'message': 'High risk of dry eye disease detected',
                'action': 'Immediate consultation recommended'
            })
        elif risk_assessment['risk_probability'] >= alert_thresholds['medium_risk']:
            alerts.append({
                'level': 'MEDIUM', 
                'message': 'Moderate risk of dry eye disease',
                'action': 'Preventive measures recommended'
            })
            
        return alerts
    
    def calculate_risk_factors_contribution(self, patient_data, feature_importance):
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
            
        contributions = {}
        total_risk = self.assess_individual_risk(patient_data)['risk_probability']
        
        for feature, importance in feature_importance.items():
            if feature in patient_df.columns:
                feature_value = patient_df[feature].iloc[0]
                normalized_value = self._normalize_feature_value(feature_value, feature)
                contribution = normalized_value * importance
                contributions[feature] = {
                    'value': feature_value,
                    'importance': importance,
                    'contribution': contribution,
                    'percentage': (contribution / total_risk * 100) if total_risk > 0 else 0
                }
                
        return contributions
    
    def _normalize_feature_value(self, value, feature):
        feature_ranges = {
            'age': (0, 100),
            'screen_time': (0, 16),
            'blink_frequency': (0, 30),
            'sleep_quality': (1, 5),
            'stress_level': (1, 5)
        }
        
        if feature in feature_ranges:
            min_val, max_val = feature_ranges[feature]
            return (value - min_val) / (max_val - min_val)
        
        return min(max(value / 10, 0), 1)