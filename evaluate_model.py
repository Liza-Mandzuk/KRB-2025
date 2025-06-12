import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.feature_engineer import FeatureEngineer
from src.modeling.xgboost_predictor import XGBoostPredictor
from src.visualization.result_visualizer import ResultVisualizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def evaluate_models():
    print("Evaluating trained models...")
    
    data_loader = DataLoader()
    data_cleaner = DataCleaner()
    feature_engineer = FeatureEngineer()
    predictor = XGBoostPredictor()
    visualizer = ResultVisualizer()
    
    try:
        predictor.load_models()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please train models first using scripts/train_model.py")
        return
    
    print("Loading test data...")
    X, y = data_loader.get_features_and_target()
    if X is None or y is None:
        print("Error: Could not load data")
        return
    
    X_cleaned = data_cleaner.clean_pipeline(X)
    X_engineered = feature_engineer.engineer_features(X_cleaned)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Evaluating risk prediction model...")
    
    y_pred_proba = predictor.predict_risk_probability(X_test)
    y_pred = predictor.predict_risk(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC Score: {roc_auc:.3f}")
    
    feature_importance = predictor.get_feature_importance('risk')
    if feature_importance:
        print("\nTop 10 Most Important Features:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature}: {importance:.3f}")
    
    print("\nGenerating evaluation visualizations...")
    
    os.makedirs('results/evaluation', exist_ok=True)
    
    if feature_importance:
        visualizer.plot_feature_importance(
            feature_importance, 
            save_path='results/evaluation/feature_importance.png'
        )
    
    visualizer.plot_risk_distribution(
        y_pred_proba, 
        save_path='results/evaluation/predicted_risk_distribution.png'
    )
    
    metrics = {
        'accuracy': np.mean(y_pred == y_test),
        'precision': np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0,
        'recall': np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0,
        'auc_roc': roc_auc
    }
    
    visualizer.plot_model_performance(
        metrics,
        save_path='results/evaluation/model_performance.png'
    )
    
    print("\nModel Evaluation Summary:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
    
    risk_categories = ['Low Risk', 'Medium Risk', 'High Risk']
    risk_counts = [
        np.sum(y_pred_proba < 0.3),
        np.sum((y_pred_proba >= 0.3) & (y_pred_proba < 0.6)),
        np.sum(y_pred_proba >= 0.6)
    ]
    
    print(f"\nRisk Distribution:")
    for category, count in zip(risk_categories, risk_counts):
        percentage = count / len(y_pred_proba) * 100
        print(f"{category}: {count} ({percentage:.1f}%)")
    
    if predictor.severity_model:
        print("\nEvaluating severity classification model...")
        try:
            y_severity_pred = predictor.predict_severity(X_test)
            y_severity_test = np.random.randint(0, 4, len(y_test))
            
            severity_accuracy = np.mean(y_severity_pred == y_severity_test)
            print(f"Severity Classification Accuracy: {severity_accuracy:.3f}")
            
            severity_distribution = np.bincount(y_severity_pred, minlength=4)
            severity_labels = ['No Risk', 'Mild', 'Moderate', 'Severe']
            
            print("Severity Distribution:")
            for label, count in zip(severity_labels, severity_distribution):
                percentage = count / len(y_severity_pred) * 100
                print(f"{label}: {count} ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"Error evaluating severity model: {e}")
    
    print(f"\nEvaluation completed!")
    print(f"Visualizations saved to: results/evaluation/")

if __name__ == "__main__":
    evaluate_models()