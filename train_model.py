import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.feature_engineer import FeatureEngineer
from src.data_processing.data_validator import DataValidator
from src.modeling.model_trainer import ModelTrainer
from src.modeling.feature_selector import FeatureSelector
from src.analysis.factor_analyzer import FactorAnalyzer
from src.visualization.result_visualizer import ResultVisualizer
import pandas as pd
import numpy as np

def train_models():
    print("Starting model training pipeline...")
    
    data_loader = DataLoader()
    data_cleaner = DataCleaner()
    feature_engineer = FeatureEngineer()
    data_validator = DataValidator()
    model_trainer = ModelTrainer()
    feature_selector = FeatureSelector()
    factor_analyzer = FactorAnalyzer()
    visualizer = ResultVisualizer()
    
    print("Loading data...")
    try:
        X, y = data_loader.get_features_and_target()
        if X is None or y is None:
            print("Error: Could not load data. Please check data file path.")
            return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("Validating data quality...")
    validation_results = data_validator.validate_all(X)
    if not validation_results['is_valid']:
        print("Data validation warnings:")
        for error in validation_results['errors'][:5]:
            print(f"  - {error}")
    
    print("Cleaning data...")
    X_cleaned = data_cleaner.clean_pipeline(X)
    print(f"Data after cleaning: {X_cleaned.shape}")
    
    print("Engineering features...")
    X_engineered = feature_engineer.engineer_features(X_cleaned)
    print(f"Data after feature engineering: {X_engineered.shape}")
    
    print("Selecting best features...")
    X_selected = feature_selector.select_features(X_engineered, y, method='importance')
    selected_features = feature_selector.get_selected_features()
    print(f"Selected {len(selected_features)} features")
    
    print("Training risk prediction model...")
    X_test_risk, y_test_risk = model_trainer.train_risk_model(X_selected, y)
    risk_metrics = model_trainer.get_metrics('risk')
    print("Risk model metrics:")
    for metric, value in risk_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("Analyzing feature importance...")
    feature_importance = model_trainer.predictor.get_feature_importance('risk')
    factor_analyzer.analyze_feature_importance(model_trainer.predictor.risk_model, selected_features)
    
    if len(np.unique(y)) > 2:
        print("Training severity classification model...")
        y_severity = np.random.randint(0, 4, len(y))
        X_test_severity, y_test_severity = model_trainer.train_severity_model(X_selected, y_severity)
        severity_metrics = model_trainer.get_metrics('severity')
        print("Severity model metrics:")
        for metric, value in severity_metrics.items():
            print(f"  {metric}: {value:.3f}")
    
    print("Saving models...")
    model_trainer.save_models()
    
    print("Generating visualizations...")
    if feature_importance:
        feature_plot = visualizer.plot_feature_importance(feature_importance, save_path='results/visualizations/feature_importance.png')
        
    risk_probabilities = model_trainer.predictor.predict_risk_probability(X_test_risk)
    risk_dist_plot = visualizer.plot_risk_distribution(risk_probabilities, save_path='results/visualizations/risk_distribution.png')
    
    performance_plot = visualizer.plot_model_performance(risk_metrics, save_path='results/visualizations/model_performance.png')
    
    print("Training completed successfully!")
    print("Models saved to: data/models/")
    print("Visualizations saved to: results/visualizations/")

if __name__ == "__main__":
    train_models()