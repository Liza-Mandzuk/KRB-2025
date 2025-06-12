XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss'
}

XGBOOST_SEVERITY_PARAMS = {
    'objective': 'multi:softprob',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 150,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'num_class': 4
}

FEATURE_SELECTION_PARAMS = {
    'method': 'importance',
    'threshold': 0.01,
    'k_best': 15
}

PREPROCESSING_PARAMS = {
    'fill_strategy': 'median',
    'scale_method': 'standard',
    'outlier_method': 'iqr',
    'outlier_factor': 1.5
}