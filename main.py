import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.feature_engineer import FeatureEngineer
from src.modeling.xgboost_predictor import XGBoostPredictor
from src.analysis.risk_assessor import RiskAssessor
from src.analysis.severity_classifier import SeverityClassifier
from src.analysis.factor_analyzer import FactorAnalyzer
from src.recommendations.recommendation_generator import RecommendationGenerator
from src.recommendations.personalization import Personalization
from src.visualization.report_generator import ReportGenerator
from config.settings import MODELS_DIR
import pandas as pd

class DryEyePredictionSystem:
    def __init__(self):
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.predictor = XGBoostPredictor()
        self.risk_assessor = RiskAssessor(self.predictor)
        self.severity_classifier = SeverityClassifier(self.predictor)
        self.factor_analyzer = FactorAnalyzer()
        self.recommendation_generator = RecommendationGenerator()
        self.personalization = Personalization()
        self.report_generator = ReportGenerator()
        
    def load_models(self):
        self.predictor.load_models()
        
    def predict_for_patient(self, patient_data):
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
            
        cleaned_data = self.data_cleaner.clean_pipeline(patient_df)
        engineered_data = self.feature_engineer.engineer_features(cleaned_data)
        
        risk_assessment = self.risk_assessor.assess_individual_risk(engineered_data)
        severity_assessment = self.severity_classifier.classify_severity(engineered_data)
        
        feature_importance = self.predictor.get_feature_importance('risk')
        if feature_importance:
            self.factor_analyzer.analyze_feature_importance(self.predictor.risk_model, 
                                                          list(feature_importance.keys()))
        
        factor_summary = self.factor_analyzer.get_factor_summary(
            patient_data, list(feature_importance.keys()) if feature_importance else None
        )
        
        recommendations = self.recommendation_generator.generate_comprehensive_recommendations(
            patient_data, risk_assessment, severity_assessment, factor_summary
        )
        
        personalized_recommendations = self.personalization.create_personalized_action_plan(
            recommendations, patient_data
        )
        
        report = self.report_generator.generate_patient_report(
            patient_data, risk_assessment, severity_assessment, 
            recommendations, factor_summary
        )
        
        return {
            'risk_assessment': risk_assessment,
            'severity_assessment': severity_assessment,
            'recommendations': personalized_recommendations,
            'factor_analysis': factor_summary,
            'report': report
        }
    
    def batch_predict(self, patients_data):
        results = []
        for patient_data in patients_data:
            result = self.predict_for_patient(patient_data)
            results.append(result)
        return results

def main():
    system = DryEyePredictionSystem()
    
    try:
        system.load_models()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please train models first using scripts/train_model.py")
        return
    
    sample_patient = {
        'age': 35,
        'gender': 1,
        'screen_time': 10,
        'blink_frequency': 12,
        'sleep_quality': 3,
        'stress_level': 4,
        'physical_activity': 30,
        'humidity': 35,
        'air_conditioner_use': 1
    }
    
    print("\nAnalyzing sample patient...")
    result = system.predict_for_patient(sample_patient)
    
    print(f"\nRisk Assessment:")
    print(f"  Probability: {result['risk_assessment']['risk_probability']:.3f}")
    print(f"  Category: {result['risk_assessment']['risk_category']}")
    print(f"  Confidence: {result['risk_assessment']['confidence']}")
    
    print(f"\nSeverity Assessment:")
    print(f"  Level: {result['severity_assessment']['severity_level']}")
    print(f"  Description: {result['severity_assessment']['severity_name']}")
    
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(result['recommendations']['immediate_actions'][:3], 1):
        print(f"  {i}. {rec['recommendation']}")
    
    report_path = system.report_generator.save_report(result['report'], 'txt')
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()