"""
Hospital Readmission Risk Prediction Model
Azure Machine Learning implementation for 30-day readmission prediction
Integrated with Power BI for real-time risk scoring
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import json
from datetime import datetime, timedelta

class ReadmissionRiskModel:
    """
    Predicts 30-day hospital readmission risk using clinical and demographic data.
    Deployed as Azure ML web service and consumed by Power BI via REST API.
    """
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.label_encoders = {}
        
    def feature_engineering(self, df):
        """
        Create features from raw FHIR/EHR data
        """
        df = df.copy()
        
        # Convert dates
        df['admission_date'] = pd.to_datetime(df['admission_date'])
        df['discharge_date'] = pd.to_datetime(df['discharge_date'])
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
        
        # Calculate Length of Stay (LOS)
        df['los_days'] = (df['discharge_date'] - df['admission_date']).dt.days
        
        # Age at admission
        df['age'] = (df['admission_date'] - df['date_of_birth']).dt.days / 365.25
        
        # Comorbidity count (simplified)
        comorbidity_cols = ['has_diabetes', 'has_hypertension', 'has_chf', 
                           'has_copd', 'has_cad']
        df['comorbidity_count'] = df[comorbidity_cols].sum(axis=1)
        
        # Previous admissions in last 12 months
        df['previous_admissions_12m'] = df['previous_admissions_12m'].fillna(0)
        
        # Emergency vs Elective
        df['is_emergency'] = (df['admission_type'] == 'Emergency').astype(int)
        
        # Discharge disposition risk
        high_risk_dispositions = ['Home Health', 'SNF', 'Left Against Medical Advice']
        df['high_risk_discharge'] = df['discharge_disposition'].isin(high_risk_dispositions).astype(int)
        
        # LACE Index components (simplified)
        # L = Length of stay, A = Acuity, C = Comorbidity, E = ED visits
        df['lace_score'] = (
            np.where(df['los_days'] < 1, 0,
            np.where(df['los_days'] <= 4, 1,
            np.where(df['los_days'] <= 6, 2,
            np.where(df['los_days'] <= 13, 3, 4)))) +
            df['is_emergency'] * 2 +  # Acuity
            np.where(df['comorbidity_count'] == 0, 0,
            np.where(df['comorbidity_count'] <= 2, 1, 2)) +  # Comorbidity
            np.where(df['ed_visits_6m'] == 0, 0,
            np.where(df['ed_visits_6m'] <= 2, 1, 
            np.where(df['ed_visits_6m'] <= 4, 2, 3))))  # ED visits
        )
        
        # Medication count (polypharmacy indicator)
        df['medication_count'] = df['medication_count'].fillna(0)
        df['polypharmacy'] = (df['medication_count'] > 5).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """
        Select and encode features for model training
        """
        feature_cols = [
            'age', 'los_days', 'comorbidity_count', 'previous_admissions_12m',
            'is_emergency', 'high_risk_discharge', 'lace_score', 'polypharmacy',
            'medication_count', 'ed_visits_6m'
        ]
        
        # Encode categorical
        categorical_cols = ['gender', 'primary_diagnosis_category', 'insurance_type']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col + '_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                feature_cols.append(col + '_encoded')
        
        X = df[feature_cols]
        y = df['is_readmitted_30day'] if 'is_readmitted_30day' in df.columns else None
        
        return X, y, feature_cols
    
    def train(self, df):
        """
        Train the Random Forest model
        """
        print("Starting model training...")
        
        # Feature engineering
        df = self.feature_engineering(df)
        X, y, feature_names = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Model AUC Score: {auc_score:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, self.model.predict(X_test)))
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(self.feature_importance.head(10))
        
        return self
    
    def predict_risk(self, patient_data):
        """
        Predict readmission risk for single patient or batch
        Returns probability 0.0 to 1.0
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert to DataFrame if dict
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        
        # Feature engineering
        patient_data = self.feature_engineering(patient_data)
        X, _, _ = self.prepare_features(patient_data)
        
        # Predict probability
        risk_probability = self.model.predict_proba(X)[:, 1]
        
        # Return with risk category
        results = []
        for prob in risk_probability:
            if prob >= 0.7:
                category = "High Risk"
                action = "Immediate case management intervention"
            elif prob >= 0.4:
                category = "Medium Risk"
                action = "Standard discharge planning protocol"
            else:
                category = "Low Risk"
                action = "Standard discharge process"
            
            results.append({
                'readmission_probability': round(float(prob), 3),
                'risk_category': category,
                'recommended_action': action,
                'top_risk_factors': self._get_top_factors(patient_data.iloc[0]) if len(patient_data) == 1 else None
            })
        
        return results[0] if len(results) == 1 else results
    
    def _get_top_factors(self, patient_row):
        """Identify top risk factors for individual patient"""
        factors = []
        
        if patient_row['age'] > 75:
            factors.append("Advanced age (>75)")
        if patient_row['los_days'] > 5:
            factors.append("Extended length of stay")
        if patient_row['comorbidity_count'] >= 2:
            factors.append("Multiple chronic conditions")
        if patient_row['previous_admissions_12m'] >= 2:
            factors.append("Frequent admissions")
        if patient_row['lace_score'] >= 7:
            factors.append("High LACE score")
            
        return factors
    
    def save_model(self, filepath='readmission_model.pkl'):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='readmission_model.pkl'):
        """Load pre-trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.label_encoders = data['label_encoders']
        self.feature_importance = data['feature_importance']
        print("Model loaded successfully")

# Azure ML Deployment wrapper
def init():
    """
    Initialization function for Azure ML deployment
    Loads the pre-trained model
    """
    global model
    model = ReadmissionRiskModel()
    model.load_model('readmission_model.pkl')

def run(raw_data):
    """
    Azure ML scoring function
    Input: JSON with patient features
    Output: JSON with risk prediction
    """
    try:
        data = json.loads(raw_data)
        result = model.predict_risk(data)
        return result
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Example training workflow
    print("Training Readmission Risk Prediction Model...")
    
    # In real scenario, load from database/FHIR
    # For demo, creating synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    demo_data = pd.DataFrame({
        'patient_id': range(n_samples),
        'admission_date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'discharge_date': pd.date_range('2023-01-02', periods=n_samples, freq='D'),
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
        'is_readmitted_30day': np.random.binomial(1, 0.15, n_samples)  # 15% readmission rate
    })
    
    # Train model
    model = ReadmissionRiskModel()
    model.train(demo_data)
    
    # Example prediction
    sample_patient = {
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
    }
    
    prediction = model.predict_risk(sample_patient)
    print("\nSample Prediction:")
    print(json.dumps(prediction, indent=2))
