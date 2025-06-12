import numpy as np
import pandas as pd
from config.settings import SEVERITY_LEVELS

class SeverityClassifier:
    def __init__(self, predictor=None):
        self.predictor = predictor
        self.severity_levels = SEVERITY_LEVELS
        
    def classify_severity(self, patient_data):
        if self.predictor is None or self.predictor.severity_model is None:
            return self._rule_based_classification(patient_data)
            
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
            
        severity_prediction = self.predictor.predict_severity(patient_df)[0]
        severity_probabilities = self.predictor.predict_severity_probability(patient_df)[0]
        
        return {
            'severity_level': severity_prediction,
            'severity_name': self.severity_levels.get(severity_prediction, 'Unknown'),
            'probabilities': dict(enumerate(severity_probabilities)),
            'confidence': max(severity_probabilities)
        }
    
    def _rule_based_classification(self, patient_data):
        if isinstance(patient_data, dict):
            data = patient_data
        else:
            data = patient_data.iloc[0].to_dict() if hasattr(patient_data, 'iloc') else {}
            
        severity_score = 0
        
        severity_score += self._evaluate_screen_time(data.get('screen_time', 0))
        severity_score += self._evaluate_blink_frequency(data.get('blink_frequency', 20))
        severity_score += self._evaluate_age(data.get('age', 30))
        severity_score += self._evaluate_sleep_quality(data.get('sleep_quality', 3))
        severity_score += self._evaluate_stress_level(data.get('stress_level', 2))
        
        if severity_score <= 5:
            severity_level = 0
        elif severity_score <= 10:
            severity_level = 1
        elif severity_score <= 15:
            severity_level = 2
        else:
            severity_level = 3
            
        return {
            'severity_level': severity_level,
            'severity_name': self.severity_levels.get(severity_level, 'Unknown'),
            'severity_score': severity_score,
            'confidence': 0.7
        }
    
    def _evaluate_screen_time(self, screen_time):
        if screen_time >= 12:
            return 4
        elif screen_time >= 8:
            return 3
        elif screen_time >= 6:
            return 2
        elif screen_time >= 4:
            return 1
        return 0
    
    def _evaluate_blink_frequency(self, blink_frequency):
        if blink_frequency <= 10:
            return 4
        elif blink_frequency <= 12:
            return 3
        elif blink_frequency <= 15:
            return 2
        elif blink_frequency <= 18:
            return 1
        return 0
    
    def _evaluate_age(self, age):
        if age >= 50:
            return 3
        elif age >= 40:
            return 2
        elif age >= 30:
            return 1
        return 0
    
    def _evaluate_sleep_quality(self, sleep_quality):
        if sleep_quality <= 2:
            return 3
        elif sleep_quality <= 3:
            return 2
        elif sleep_quality <= 4:
            return 1
        return 0
    
    def _evaluate_stress_level(self, stress_level):
        if stress_level >= 4:
            return 3
        elif stress_level >= 3:
            return 2
        elif stress_level >= 2:
            return 1
        return 0
    
    def get_severity_description(self, severity_level):
        descriptions = {
            0: "No significant dry eye symptoms. Maintain good eye hygiene.",
            1: "Mild dry eye symptoms. Monitor and implement basic preventive measures.",
            2: "Moderate dry eye symptoms. Consider lifestyle modifications and regular monitoring.",
            3: "Severe dry eye symptoms. Professional consultation and treatment recommended."
        }
        return descriptions.get(severity_level, "Unknown severity level")
    
    def assess_progression_risk(self, current_severity, risk_factors):
        progression_score = current_severity * 2
        
        high_risk_factors = ['high_screen_time', 'low_blink_frequency', 'poor_sleep_quality', 'high_stress']
        for factor in high_risk_factors:
            if risk_factors.get(factor, False):
                progression_score += 1
                
        if progression_score <= 3:
            return 'Low'
        elif progression_score <= 6:
            return 'Medium'
        else:
            return 'High'
    
    def recommend_monitoring_frequency(self, severity_level, progression_risk):
        if severity_level >= 3 or progression_risk == 'High':
            return 'Weekly monitoring recommended'
        elif severity_level >= 2 or progression_risk == 'Medium':
            return 'Bi-weekly monitoring recommended'
        elif severity_level >= 1:
            return 'Monthly monitoring recommended'
        else:
            return 'Quarterly monitoring sufficient'
    
    def compare_severity_over_time(self, severity_history):
        if len(severity_history) < 2:
            return None
            
        trend_direction = 'stable'
        if severity_history[-1] > severity_history[0]:
            trend_direction = 'worsening'
        elif severity_history[-1] < severity_history[0]:
            trend_direction = 'improving'
            
        volatility = np.std(severity_history) if len(severity_history) > 2 else 0
        
        return {
            'trend_direction': trend_direction,
            'severity_change': severity_history[-1] - severity_history[0],
            'volatility': volatility,
            'consistency': 'stable' if volatility <= 0.5 else 'variable'
        }