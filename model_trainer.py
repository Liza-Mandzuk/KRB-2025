from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from src.modeling.xgboost_predictor import XGBoostPredictor
from config.settings import TEST_SIZE, RANDOM_STATE, CV_FOLDS

class ModelTrainer:
    def __init__(self):
        self.predictor = XGBoostPredictor()
        self.metrics = {}
        
    def split_data(self, X, y, test_size=TEST_SIZE, stratify=True):
        stratify_param = y if stratify else None
        return train_test_split(X, y, test_size=test_size, 
                              random_state=RANDOM_STATE, stratify=stratify_param)
    
    def train_risk_model(self, X, y, validation_split=0.2):
        X_train_full, X_test, y_train_full, y_test = self.split_data(X, y)
        
        if validation_split > 0:
            X_train, X_val, y_train, y_val = self.split_data(
                X_train_full, y_train_full, test_size=validation_split
            )
            self.predictor.train_risk_model(X_train, y_train, X_val, y_val)
        else:
            self.predictor.train_risk_model(X_train_full, y_train_full)
            
        self._evaluate_model(X_test, y_test, 'risk')
        return X_test, y_test
    
    def train_severity_model(self, X, y, validation_split=0.2):
        X_train_full, X_test, y_train_full, y_test = self.split_data(X, y)
        
        if validation_split > 0:
            X_train, X_val, y_train, y_val = self.split_data(
                X_train_full, y_train_full, test_size=validation_split
            )
            self.predictor.train_severity_model(X_train, y_train, X_val, y_val)
        else:
            self.predictor.train_severity_model(X_train_full, y_train_full)
            
        self._evaluate_model(X_test, y_test, 'severity')
        return X_test, y_test
    
    def _evaluate_model(self, X_test, y_test, model_type='risk'):
        if model_type == 'risk':
            y_pred = self.predictor.predict_risk(X_test)
            y_pred_proba = self.predictor.predict_risk_probability(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba)
            }
        else:
            y_pred = self.predictor.predict_severity(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
        self.metrics[model_type] = metrics
        
    def cross_validate_model(self, X, y, model_type='risk', cv=CV_FOLDS):
        if model_type == 'risk':
            model = self.predictor.risk_model
        else:
            model = self.predictor.severity_model
            
        if model is None:
            return None
            
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores
        }
    
    def hyperparameter_tuning(self, X, y, param_grid, model_type='risk'):
        base_model = self.predictor.risk_model if model_type == 'risk' else self.predictor.severity_model
        
        grid_search = GridSearchCV(base_model, param_grid, cv=CV_FOLDS, 
                                 scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def get_metrics(self, model_type=None):
        if model_type:
            return self.metrics.get(model_type, {})
        return self.metrics
    
    def save_models(self):
        self.predictor.save_models()
        
    def load_models(self):
        self.predictor.load_models()