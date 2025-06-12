import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

FEATURE_IMPORTANCE_THRESHOLD = 0.01
RISK_THRESHOLD = 0.5

DEFAULT_FEATURES = [
    'age', 'gender', 'screen_time', 'blink_frequency', 'sleep_quality',
    'stress_level', 'physical_activity', 'humidity', 'air_conditioner_use'
]

SEVERITY_LEVELS = {
    0: 'No Risk',
    1: 'Mild',
    2: 'Moderate', 
    3: 'Severe'
}