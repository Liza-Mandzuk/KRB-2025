import pandas as pd
import json
from datetime import datetime
from src.utils.helpers import get_timestamp
import os

class ReportGenerator:
    def __init__(self):
        self.report_template = {
            'patient_info': {},
            'assessment_date': '',
            'risk_assessment': {},
            'severity_assessment': {},
            'key_factors': {},
            'recommendations': [],
            'follow_up': {}
        }
        
    def generate_patient_report(self, patient_data, risk_assessment, severity_assessment, 
                              recommendations, factor_analysis):
        report = self.report_template.copy()
        
        report['patient_info'] = self._extract_patient_info(patient_data)
        report['assessment_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report['risk_assessment'] = risk_assessment
        report['severity_assessment'] = severity_assessment
        report['key_factors'] = factor_analysis
        report['recommendations'] = recommendations
        report['follow_up'] = self._generate_follow_up_plan(risk_assessment, severity_assessment)
        
        return report
    
    def _extract_patient_info(self, patient_data):
        if isinstance(patient_data, dict):
            data = patient_data
        else:
            data = patient_data.iloc[0].to_dict() if hasattr(patient_data, 'iloc') else {}
            
        return {
            'age': data.get('age', 'N/A'),
            'gender': 'Female' if data.get('gender', 1) == 0 else 'Male',
            'screen_time': f"{data.get('screen_time', 'N/A')} hours/day",
            'sleep_quality': f"{data.get('sleep_quality', 'N/A')}/5",
            'stress_level': f"{data.get('stress_level', 'N/A')}/5"
        }
    
    def _generate_follow_up_plan(self, risk_assessment, severity_assessment):
        risk_level = risk_assessment.get('risk_category', 'Low Risk')
        severity_level = severity_assessment.get('severity_level', 0)
        
        if risk_level == 'High Risk' or severity_level >= 3:
            interval = '2 weeks'
            duration = '3 months'
        elif risk_level == 'Medium Risk' or severity_level >= 2:
            interval = '1 month'
            duration = '6 months'
        else:
            interval = '3 months'
            duration = '1 year'
            
        return {
            'next_assessment': interval,
            'monitoring_duration': duration,
            'specialist_referral': risk_level == 'High Risk' or severity_level >= 3
        }
    
    def format_text_report(self, report_data):
        text_report = []
        text_report.append("=" * 60)
        text_report.append("DRY EYE DISEASE RISK ASSESSMENT REPORT")
        text_report.append("=" * 60)
        text_report.append(f"Assessment Date: {report_data['assessment_date']}")
        text_report.append("")
        
        # Patient Information
        text_report.append("PATIENT INFORMATION")
        text_report.append("-" * 20)
        patient_info = report_data['patient_info']
        for key, value in patient_info.items():
            text_report.append(f"{key.replace('_', ' ').title()}: {value}")
        text_report.append("")
        
        # Risk Assessment
        text_report.append("RISK ASSESSMENT")
        text_report.append("-" * 15)
        risk_data = report_data['risk_assessment']
        text_report.append(f"Risk Probability: {risk_data.get('risk_probability', 0):.3f}")
        text_report.append(f"Risk Category: {risk_data.get('risk_category', 'Unknown')}")
        text_report.append(f"Confidence Level: {risk_data.get('confidence', 'Unknown')}")
        text_report.append("")
        
        # Severity Assessment
        text_report.append("SEVERITY ASSESSMENT")
        text_report.append("-" * 18)
        severity_data = report_data['severity_assessment']
        text_report.append(f"Severity Level: {severity_data.get('severity_level', 0)}")
        text_report.append(f"Severity Description: {severity_data.get('severity_name', 'Unknown')}")
        text_report.append("")
        
        # Key Risk Factors
        text_report.append("KEY RISK FACTORS")
        text_report.append("-" * 16)
        key_factors = report_data['key_factors']
        for factor, data in key_factors.items():
            if isinstance(data, dict):
                importance = data.get('importance', 0)
                value = data.get('value', 'N/A')
                text_report.append(f"{factor.replace('_', ' ').title()}: {value} (Importance: {importance:.3f})")
        text_report.append("")
        
        # Recommendations
        text_report.append("RECOMMENDATIONS")
        text_report.append("-" * 13)
        recommendations = report_data['recommendations']
        for i, rec in enumerate(recommendations, 1):
            priority = rec.get('priority', 'Medium')
            category = rec.get('category', 'General')
            message = rec.get('recommendation', '')
            text_report.append(f"{i}. [{priority}] {category}: {message}")
        text_report.append("")
        
        # Follow-up Plan
        text_report.append("FOLLOW-UP PLAN")
        text_report.append("-" * 13)
        follow_up = report_data['follow_up']
        text_report.append(f"Next Assessment: {follow_up.get('next_assessment', 'TBD')}")
        text_report.append(f"Monitoring Duration: {follow_up.get('monitoring_duration', 'TBD')}")
        text_report.append(f"Specialist Referral: {'Yes' if follow_up.get('specialist_referral', False) else 'No'}")
        
        return "\n".join(text_report)
    
    def save_report(self, report_data, format='json', save_path=None):
        if save_path is None:
            timestamp = get_timestamp()
            filename = f"dry_eye_report_{timestamp}"
            save_path = os.path.join('results', 'reports', f"{filename}.{format}")
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if format == 'json':
            with open(save_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        elif format == 'txt':
            text_report = self.format_text_report(report_data)
            with open(save_path, 'w') as f:
                f.write(text_report)
        elif format == 'csv':
            df = self._convert_to_dataframe(report_data)
            df.to_csv(save_path, index=False)
            
        return save_path
    
    def _convert_to_dataframe(self, report_data):
        flat_data = {}
        
        # Flatten patient info
        patient_info = report_data.get('patient_info', {})
        for key, value in patient_info.items():
            flat_data[f'patient_{key}'] = value
            
        # Flatten risk assessment
        risk_data = report_data.get('risk_assessment', {})
        for key, value in risk_data.items():
            flat_data[f'risk_{key}'] = value
            
        # Flatten severity assessment
        severity_data = report_data.get('severity_assessment', {})
        for key, value in severity_data.items():
            flat_data[f'severity_{key}'] = value
            
        flat_data['assessment_date'] = report_data.get('assessment_date', '')
        flat_data['recommendations_count'] = len(report_data.get('recommendations', []))
        
        return pd.DataFrame([flat_data])
    
    def generate_summary_statistics(self, multiple_reports):
        if not multiple_reports:
            return {}
            
        risk_probabilities = [r.get('risk_assessment', {}).get('risk_probability', 0) 
                            for r in multiple_reports]
        severity_levels = [r.get('severity_assessment', {}).get('severity_level', 0) 
                         for r in multiple_reports]
        
        return {
            'total_assessments': len(multiple_reports),
            'average_risk': sum(risk_probabilities) / len(risk_probabilities),
            'high_risk_count': sum(1 for p in risk_probabilities if p >= 0.6),
            'average_severity': sum(severity_levels) / len(severity_levels),
            'severe_cases': sum(1 for s in severity_levels if s >= 3),
            'risk_distribution': {
                'low': sum(1 for p in risk_probabilities if p < 0.3),
                'medium': sum(1 for p in risk_probabilities if 0.3 <= p < 0.6),
                'high': sum(1 for p in risk_probabilities if p >= 0.6)
            }
        }
    
    def create_batch_report(self, multiple_reports, save_path=None):
        summary_stats = self.generate_summary_statistics(multiple_reports)
        
        batch_report = {
            'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary_statistics': summary_stats,
            'individual_reports': multiple_reports
        }
        
        if save_path:
            self.save_report(batch_report, 'json', save_path)
            
        return batch_report