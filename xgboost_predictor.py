import xgboost as xgb
import numpy as np
import joblib
from config.model_config import XGBOOST_PARAMS, XGBOOST_SEVERITY_PARAMS
from config.settings import RISK_THRESHOLD, MODELS_DIR
import os

class XGBoostPredictor:
    def __init__(self):
        self.risk_model = None
        self.severity_model = None
        self.feature_names = None
        self.is_trained = False
        
    def train_risk_model(self, X_train, y_train, X_val=None, y_val=None):
        self.risk_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.risk_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.risk_model.fit(X_train, y_train)
            
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.is_trained = True
        
    def train_severity_model(self, X_train, y_train, X_val=None, y_val=None):
        self.severity_model = xgb.XGBClassifier(**XGBOOST_SEVERITY_PARAMS)
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.severity_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.severity_model.fit(X_train, y_train)
            
    def predict_risk_probability(self, X):
        if self.risk_model is None:
            raise ValueError("Risk model not trained")
        return self.risk_model.predict_proba(X)[:, 1]
    
    def predict_risk(self, X, threshold=RISK_THRESHOLD):
        probabilities = self.predict_risk_probability(X)
        return (probabilities >= threshold).astype(int)
    
    def predict_severity(self, X):
        if self.severity_model is None:
            raise ValueError("Severity model not trained")
        return self.severity_model.predict(X)
    
    def predict_severity_probability(self, X):
        if self.severity_model is None:
            raise ValueError("Severity model not trained")
        return self.severity_model.predict_proba(X)
    
    def get_feature_importance(self, model_type='risk'):
        model = self.risk_model if model_type == 'risk' else self.severity_model
        if model is None:
            return None
            
        importance = model.feature_importances_
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return importance
    
    def save_models(self, risk_path=None, severity_path=None):
        if risk_path is None:
            risk_path = os.path.join(MODELS_DIR, 'xgboost_risk_model.pkl')
        if severity_path is None:
            severity_path = os.path.join(MODELS_DIR, 'xgboost_severity_model.pkl')
            
        if self.risk_model:
            joblib.dump(self.risk_model, risk_path)
        if self.severity_model:
            joblib.dump(self.severity_model, severity_path)
            
    def load_models(self, risk_path=None, severity_path=None):
        if risk_path is None:
            risk_path = os.path.join(MODELS_DIR, 'xgboost_risk_model.pkl')
        if severity_path is None:
            severity_path = os.path.join(MODELS_DIR, 'xgboost_severity_model.pkl')
            
        if os.path.exists(risk_path):
            self.risk_model = joblib.load(risk_path)
            self.is_trained = True
        if os.path.exists(severity_path):
            self.severity_model = joblib.load(severity_path)