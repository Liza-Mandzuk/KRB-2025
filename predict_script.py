import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from main import DryEyePredictionSystem

def predict_from_file(file_path):
    system = DryEyePredictionSystem()
    
    try:
        system.load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    if file_path.endswith('.csv'):
        patients_data = pd.read_csv(file_path)
        patients_list = patients_data.to_dict('records')
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            patients_list = json.load(f)
    else:
        print("Unsupported file format. Use CSV or JSON.")
        return
    
    print(f"Processing {len(patients_list)} patients...")
    
    results = []
    for i, patient_data in enumerate(patients_list):
        print(f"Processing patient {i+1}/{len(patients_list)}")
        try:
            result = system.predict_for_patient(patient_data)
            results.append(result)
        except Exception as e:
            print(f"Error processing patient {i+1}: {e}")
            continue
    
    batch_report = system.report_generator.create_batch_report(
        [r['report'] for r in results], 
        'results/reports/batch_predictions.json'
    )
    
    print(f"\nBatch processing completed!")
    print(f"Processed {len(results)} patients successfully")
    print(f"Average risk: {batch_report['summary_statistics']['average_risk']:.3f}")
    print(f"High risk patients: {batch_report['summary_statistics']['high_risk_count']}")
    print(f"Reports saved to: results/reports/")

def predict_interactive():
    system = DryEyePredictionSystem()
    
    try:
        system.load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    print("Interactive Dry Eye Risk Assessment")
    print("=" * 40)
    
    patient_data = {}
    
    try:
        patient_data['age'] = int(input("Age: "))
        patient_data['gender'] = 1 if input("Gender (M/F): ").upper() == 'M' else 0
        patient_data['screen_time'] = float(input("Daily screen time (hours): "))
        patient_data['blink_frequency'] = float(input("Blink frequency (per minute): "))
        patient_data['sleep_quality'] = int(input("Sleep quality (1-5): "))
        patient_data['stress_level'] = int(input("Stress level (1-5): "))
        patient_data['physical_activity'] = float(input("Physical activity (minutes/day): "))
        patient_data['humidity'] = float(input("Home humidity (%): "))
        patient_data['air_conditioner_use'] = 1 if input("Use air conditioner (Y/N): ").upper() == 'Y' else 0
    except ValueError:
        print("Invalid input. Please enter numeric values where required.")
        return
    
    print("\nAnalyzing...")
    result = system.predict_for_patient(patient_data)
    
    print("\n" + "=" * 40)
    print("ASSESSMENT RESULTS")
    print("=" * 40)
    
    risk = result['risk_assessment']
    print(f"Risk Probability: {risk['risk_probability']:.3f}")
    print(f"Risk Category: {risk['risk_category']}")
    print(f"Confidence: {risk['confidence']}")
    
    severity = result['severity_assessment']
    print(f"\nSeverity Level: {severity['severity_level']}")
    print(f"Description: {severity['severity_name']}")
    
    print("\nTop Recommendations:")
    immediate_actions = result['recommendations']['immediate_actions']
    for i, rec in enumerate(immediate_actions[:5], 1):
        print(f"{i}. {rec['recommendation']}")
    
    save_report = input("\nSave detailed report? (Y/N): ").upper() == 'Y'
    if save_report:
        report_path = system.report_generator.save_report(result['report'], 'txt')
        print(f"Report saved to: {report_path}")

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            predict_from_file(file_path)
        else:
            print(f"File not found: {file_path}")
    else:
        predict_interactive()

if __name__ == "__main__":
    main()