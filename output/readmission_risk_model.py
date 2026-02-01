"""
Hospital Readmission Risk Prediction Model
Azure Machine Learning implementation for 30-day readmission prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import json
from datetime import datetime, timedelta

class ReadmissionRiskModel:
    """
    Predicts 30-day hospital readmission risk using clinical and demographic data.
    """
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.label_encoders = {}
        
    def feature_engineering(self, df):
        """Create features from raw FHIR/EHR data"""
        df = df.copy()
        
        # Convert dates
        df['admission_date'] = pd.to_datetime(df['admission_date'])
        df['discharge_date'] = pd.to_datetime(df['discharge_date'])
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
        
        # Calculate Length of Stay (LOS)
        df['los_days'] = (df['discharge_date'] - df['admission_date']).dt.days
        
        # Age at admission
        df['age'] = (df['admission_date'] - df['date_of_birth']).dt.days / 365.25
        
        # Comorbidity count
        comorbidity_cols = ['has_diabetes', 'has_hypertension', 'has_chf', 
                           'has_copd', 'has_cad']
        df['comorbidity_count'] = df[comorbidity_cols].sum(axis=1)
        
        # Previous admissions
        df['previous_admissions_12m'] = df['previous_admissions_12m'].fillna(0)
        
        # Emergency vs Elective
        df['is_emergency'] = (df['admission_type'] == 'Emergency').astype(int)
        
        # High risk discharge
        high_risk_dispositions = ['Home Health', 'SNF', 'Left Against Medical Advice']
        df['high_risk_discharge'] = df['discharge_disposition'].isin(high_risk_dispositions).astype(int)
        
        # ED visits
        df['ed_visits_6m'] = df['ed_visits_6m'].fillna(0)
        
        # Polypharmacy
        df['medication_count'] = df['medication_count'].fillna(0)
        df['polypharmacy'] = (df['medication_count'] > 5).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """Select and encode features for model training"""
        feature_cols = [
            'age', 'los_days', 'comorbidity_count', 'previous_admissions_12m',
            'is_emergency', 'high_risk_discharge', 'polypharmacy',
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
        """Train the Random Forest model"""
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
            class_weight='balanced'
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
        """Predict readmission risk for single patient"""
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
        """Identify top risk factors"""
        factors = []
        
        if patient_row['age'] > 75:
            factors.append("Advanced age (>75)")
        if patient_row['los_days'] > 5:
            factors.append("Extended length of stay")
        if patient_row['comorbidity_count'] >= 2:
            factors.append("Multiple chronic conditions")
        if patient_row['previous_admissions_12m'] >= 2:
            factors.append("Frequent admissions")
            
        return factors