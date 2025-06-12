from src.utils.constants import RECOMMENDATION_RULES
import pandas as pd

class RecommendationGenerator:
    def __init__(self):
        self.rules = RECOMMENDATION_RULES
        self.recommendations = []
        
    def generate_lifestyle_recommendations(self, patient_data, risk_factors):
        recommendations = []
        
        for factor, rule in self.rules.items():
            if factor in patient_data:
                value = patient_data[factor]
                if rule['condition'](value):
                    recommendations.append({
                        'category': 'Lifestyle',
                        'factor': factor,
                        'current_value': value,
                        'recommendation': rule['message'],
                        'priority': self._get_priority(factor, risk_factors)
                    })
                    
        return recommendations
    
    def generate_environmental_recommendations(self, patient_data):
        recommendations = []
        
        humidity = patient_data.get('humidity', 50)
        if humidity < 40:
            recommendations.append({
                'category': 'Environment',
                'factor': 'humidity',
                'current_value': humidity,
                'recommendation': 'Use a humidifier to maintain humidity between 40-60%',
                'priority': 'High'
            })
            
        air_conditioner = patient_data.get('air_conditioner_use', 0)
        if air_conditioner:
            recommendations.append({
                'category': 'Environment',
                'factor': 'air_conditioning',
                'current_value': air_conditioner,
                'recommendation': 'Reduce direct exposure to air conditioning, use eye drops',
                'priority': 'Medium'
            })
            
        return recommendations
    
    def generate_medical_recommendations(self, severity_level, risk_probability):
        recommendations = []
        
        if risk_probability >= 0.7:
            recommendations.append({
                'category': 'Medical',
                'factor': 'high_risk',
                'recommendation': 'Schedule immediate ophthalmological consultation',
                'priority': 'Critical'
            })
        elif risk_probability >= 0.4:
            recommendations.append({
                'category': 'Medical',
                'factor': 'medium_risk',
                'recommendation': 'Consider routine eye examination within 3 months',
                'priority': 'High'
            })
            
        if severity_level >= 2:
            recommendations.append({
                'category': 'Medical',
                'factor': 'severity',
                'recommendation': 'Consider artificial tears or prescription eye drops',
                'priority': 'High'
            })
            
        return recommendations
    
    def generate_behavioral_recommendations(self, patient_data):
        recommendations = []
        
        screen_time = patient_data.get('screen_time', 0)
        if screen_time > 8:
            recommendations.append({
                'category': 'Behavioral',
                'factor': 'screen_time',
                'current_value': screen_time,
                'recommendation': 'Follow 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds',
                'priority': 'High'
            })
            
        blink_frequency = patient_data.get('blink_frequency', 20)
        if blink_frequency < 15:
            recommendations.append({
                'category': 'Behavioral',
                'factor': 'blinking',
                'current_value': blink_frequency,
                'recommendation': 'Practice conscious blinking exercises, blink fully and frequently',
                'priority': 'High'
            })
            
        return recommendations
    
    def _get_priority(self, factor, risk_factors):
        high_priority_factors = ['screen_time', 'blink_frequency', 'sleep_quality']
        
        if factor in high_priority_factors:
            importance = risk_factors.get(factor, {}).get('importance', 0)
            if importance > 0.7:
                return 'Critical'
            elif importance > 0.4:
                return 'High'
            else:
                return 'Medium'
        return 'Medium'
    
    def generate_comprehensive_recommendations(self, patient_data, risk_assessment, severity_assessment, risk_factors):
        all_recommendations = []
        
        all_recommendations.extend(self.generate_lifestyle_recommendations(patient_data, risk_factors))
        all_recommendations.extend(self.generate_environmental_recommendations(patient_data))
        all_recommendations.extend(self.generate_medical_recommendations(
            severity_assessment.get('severity_level', 0),
            risk_assessment.get('risk_probability', 0)
        ))
        all_recommendations.extend(self.generate_behavioral_recommendations(patient_data))
        
        sorted_recommendations = sorted(all_recommendations, 
                                      key=lambda x: self._priority_score(x['priority']), 
                                      reverse=True)
        
        return sorted_recommendations
    
    def _priority_score(self, priority):
        priority_scores = {
            'Critical': 4,
            'High': 3,
            'Medium': 2,
            'Low': 1
        }
        return priority_scores.get(priority, 1)
    
    def format_recommendations_for_patient(self, recommendations):
        formatted = {
            'Critical': [],
            'High': [],
            'Medium': [],
            'Low': []
        }
        
        for rec in recommendations:
            priority = rec.get('priority', 'Medium')
            formatted[priority].append({
                'message': rec['recommendation'],
                'category': rec['category'],
                'factor': rec.get('factor', 'general')
            })
            
        return formatted
    
    def generate_follow_up_schedule(self, risk_level, severity_level):
        if risk_level == 'High Risk' or severity_level >= 3:
            return {
                'next_assessment': '2 weeks',
                'follow_up_frequency': 'Bi-weekly',
                'monitoring_duration': '3 months'
            }
        elif risk_level == 'Medium Risk' or severity_level >= 2:
            return {
                'next_assessment': '1 month',
                'follow_up_frequency': 'Monthly',
                'monitoring_duration': '6 months'
            }
        else:
            return {
                'next_assessment': '3 months',
                'follow_up_frequency': 'Quarterly',
                'monitoring_duration': '1 year'
            }