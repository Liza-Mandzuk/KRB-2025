RISK_CATEGORIES = {
    (0, 0.3): 'Low Risk',
    (0.3, 0.6): 'Medium Risk', 
    (0.6, 1.0): 'High Risk'
}

FEATURE_MAPPINGS = {
    'gender': {'M': 1, 'F': 0},
    'smoking': {'Yes': 1, 'No': 0},
    'air_conditioner_use': {'Yes': 1, 'No': 0},
    'contact_lenses': {'Yes': 1, 'No': 0}
}

RECOMMENDATION_RULES = {
    'screen_time': {
        'condition': lambda x: x > 8,
        'message': 'Reduce screen time, take regular breaks every 20 minutes'
    },
    'blink_frequency': {
        'condition': lambda x: x < 15,
        'message': 'Practice conscious blinking exercises'
    },
    'sleep_quality': {
        'condition': lambda x: x < 3,
        'message': 'Improve sleep hygiene and aim for 7-8 hours of quality sleep'
    },
    'stress_level': {
        'condition': lambda x: x > 3,
        'message': 'Consider stress management techniques like meditation'
    },
    'humidity': {
        'condition': lambda x: x < 40,
        'message': 'Use a humidifier to maintain optimal humidity levels'
    }
}

FACTOR_WEIGHTS = {
    'screen_time': 1.0,
    'blink_frequency': 0.85,
    'age': 0.71,
    'sleep_quality': 0.63,
    'stress_level': 0.58,
    'humidity': 0.47,
    'air_conditioner_use': 0.42
}