import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from azure_ml.readmission_risk_model import ReadmissionRiskModel
import json
import matplotlib.pyplot as plt

# Create output directory
import os
os.makedirs('output', exist_ok=True)

# Generate test data
np.random.seed(42)
n_samples = 1000

print("🚀 Starting Clinical Operations Model Training...")
print("=" * 50)

demo_data = pd.DataFrame({
    'patient_id': range(n_samples),
    'admission_date': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
    'discharge_date': pd.date_range('2024-01-02', periods=n_samples, freq='H'),
    'date_of_birth': pd.date_range('1950-01-01', periods=n_samples, freq='D'),
    'has_diabetes': np.random.binomial(1, 0.3, n_samples),
    'has_hypertension': np.random.binomial(1, 0.4, n_samples),
    'has_chf': np.random.binomial(1, 0.1, n_samples),
    'has_copd': np.random.binomial(1, 0.15, n_samples),
    'has_cad': np.random.binomial(1, 0.2, n_samples),
    'previous_admissions_12m': np.random.poisson(1, n_samples),
    'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], n_samples),
    'discharge_disposition': np.random.choice(['Home', 'Home Health', 'SNF', 'Rehab'], n_samples),
    'ed_visits_6m': np.random.poisson(2, n_samples),
    'medication_count': np.random.poisson(8, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'primary_diagnosis_category': np.random.choice(['Cardiac', 'Pulmonary', 'Diabetes', 'Infection'], n_samples),
    'insurance_type': np.random.choice(['Medicare', 'Medicaid', 'Commercial'], n_samples),
    'is_readmitted_30day': np.random.binomial(1, 0.15, n_samples)
})

# Train model
model = ReadmissionRiskModel()
model.train(demo_data)

# Test predictions
test_patients = [
    {
        'admission_date': datetime.now() - timedelta(days=5),
        'discharge_date': datetime.now(),
        'date_of_birth': datetime.now() - timedelta(days=75*365),
        'has_diabetes': 1,
        'has_hypertension': 1,
        'has_chf': 1,
        'has_copd': 0,
        'has_cad': 1,
        'previous_admissions_12m': 3,
        'admission_type': 'Emergency',
        'discharge_disposition': 'Home Health',
        'ed_visits_6m': 4,
        'medication_count': 12,
        'gender': 'M',
        'primary_diagnosis_category': 'Cardiac',
        'insurance_type': 'Medicare'
    },
    {
        'admission_date': datetime.now() - timedelta(days=2),
        'discharge_date': datetime.now(),
        'date_of_birth': datetime.now() - timedelta(days=45*365),
        'has_diabetes': 0,
        'has_hypertension': 0,
        'has_chf': 0,
        'has_copd': 0,
        'has_cad': 0,
        'previous_admissions_12m': 0,
        'admission_type': 'Elective',
        'discharge_disposition': 'Home',
        'ed_visits_6m': 0,
        'medication_count': 2,
        'gender': 'F',
        'primary_diagnosis_category': 'Pulmonary',
        'insurance_type': 'Commercial'
    }
]

print("\n📊 Test Predictions:")
print("-" * 50)

results = []
for i, patient in enumerate(test_patients):
    prediction = model.predict_risk(patient)
    results.append(prediction)
    print(f"\nPatient {i+1}:")
    print(f"  Risk Probability: {prediction['readmission_probability']*100:.1f}%")
    print(f"  Category: {prediction['risk_category']}")
    print(f"  Action: {prediction['recommended_action']}")
    if prediction['top_risk_factors']:
        print(f"  Risk Factors: {', '.join(prediction['top_risk_factors'])}")

# Save results
with open('output/predictions.json', 'w') as f:
    json.dump(results, f, indent=2)

# Feature importance plot
plt.figure(figsize=(10, 6))
feature_imp = model.feature_importance.head(10)
plt.barh(feature_imp['feature'], feature_imp['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Readmission Risk Factors')
plt.tight_layout()
plt.savefig('output/feature_importance.png')
print("\n✅ Feature importance chart saved to output/feature_importance.png")

# Risk distribution
plt.figure(figsize=(8, 5))
risks = [r['readmission_probability'] for r in 
         [model.predict_risk({**test_patients[0], 'has_diabetes': np.random.choice([0,1]), 
          'previous_admissions_12m': np.random.randint(0,5)}) for _ in range(100)]]
plt.hist(risks, bins=20, edgecolor='black')
plt.xlabel('Readmission Probability')
plt.ylabel('Count')
plt.title('Distribution of Risk Scores (Sample)')
plt.savefig('output/risk_distribution.png')
print("✅ Risk distribution chart saved to output/risk_distribution.png")

print("\n🎉 Model execution complete! Check the 'output' folder.")