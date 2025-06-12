from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from config.model_config import FEATURE_SELECTION_PARAMS
from config.settings import FEATURE_IMPORTANCE_THRESHOLD

class FeatureSelector:
    def __init__(self):
        self.selected_features = []
        self.feature_importance = {}
        self.selector = None
        
    def select_by_importance(self, X, y, model, threshold=FEATURE_IMPORTANCE_THRESHOLD):
        model.fit(X, y)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = abs(model.coef_[0])
        else:
            return X
            
        feature_names = X.columns if hasattr(X, 'columns') else range(len(importances))
        self.feature_importance = dict(zip(feature_names, importances))
        
        selected_mask = importances >= threshold
        self.selected_features = [feat for feat, mask in zip(feature_names, selected_mask) if mask]
        
        if hasattr(X, 'columns'):
            return X[self.selected_features]
        return X[:, selected_mask]
    
    def select_k_best(self, X, y, k=15, score_func=f_classif):
        self.selector = SelectKBest(score_func=score_func, k=k)
        X_selected = self.selector.fit_transform(X, y)
        
        if hasattr(X, 'columns'):
            selected_indices = self.selector.get_support(indices=True)
            self.selected_features = X.columns[selected_indices].tolist()
            return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        
        return X_selected
    
    def recursive_feature_elimination(self, X, y, estimator=None, n_features=10):
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            
        self.selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = self.selector.fit_transform(X, y)
        
        if hasattr(X, 'columns'):
            selected_mask = self.selector.support_
            self.selected_features = X.columns[selected_mask].tolist()
            return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
            
        return X_selected
    
    def correlation_filter(self, X, threshold=0.95):
        if not hasattr(X, 'corr'):
            return X
            
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                            if any(upper_triangle[column] > threshold)]
        
        self.selected_features = [col for col in X.columns if col not in high_corr_features]
        return X[self.selected_features]
    
    def variance_filter(self, X, threshold=0.01):
        if hasattr(X, 'var'):
            variances = X.var()
            low_variance_features = variances[variances <= threshold].index.tolist()
            self.selected_features = [col for col in X.columns if col not in low_variance_features]
            return X[self.selected_features]
        return X
    
    def select_features(self, X, y, method='importance', **kwargs):
        if method == 'importance':
            from sklearn.ensemble import RandomForestClassifier
            model = kwargs.get('model', RandomForestClassifier(n_estimators=50, random_state=42))
            threshold = kwargs.get('threshold', FEATURE_IMPORTANCE_THRESHOLD)
            return self.select_by_importance(X, y, model, threshold)
        elif method == 'k_best':
            k = kwargs.get('k', FEATURE_SELECTION_PARAMS['k_best'])
            return self.select_k_best(X, y, k)
        elif method == 'rfe':
            n_features = kwargs.get('n_features', 10)
            estimator = kwargs.get('estimator', None)
            return self.recursive_feature_elimination(X, y, estimator, n_features)
        elif method == 'correlation':
            threshold = kwargs.get('threshold', 0.95)
            return self.correlation_filter(X, threshold)
        elif method == 'variance':
            threshold = kwargs.get('threshold', 0.01)
            return self.variance_filter(X, threshold)
        else:
            return X
    
    def get_selected_features(self):
        return self.selected_features
    
    def get_feature_importance(self):
        return self.feature_importance
    
    def transform(self, X):
        if self.selected_features and hasattr(X, 'columns'):
            return X[self.selected_features]
        elif self.selector:
            return self.selector.transform(X)
        return X