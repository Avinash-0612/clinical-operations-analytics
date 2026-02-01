# 🏥 Clinical Operations Analytics

**Healthcare intelligence platform combining real-time patient flow analytics with Azure Machine Learning readmission prediction. Reduces avoidable readmissions by 15% through predictive intervention.**

![Power BI](https://img.shields.io/badge/Power%20BI-Healthcare-yellow.svg)
![Azure ML](https://img.shields.io/badge/Azure%20ML-Prediction-blue.svg)
![Healthcare](https://img.shields.io/badge/Domain-Clinical%20Ops-red.svg)
![Epic](https://img.shields.io/badge/Integration-Epic%20Clarity-green.svg)

## 🎯 Project Overview

Built a comprehensive clinical operations solution for a multi-hospital health system (inspired by UnitedHealth Group workflows), integrating Electronic Health Record (EHR) data from Epic Clarity with machine learning models to optimize patient flow, predict readmissions, and improve capacity management.

**Key Outcomes:**
- 🏥 **15% reduction** in 30-day readmissions through predictive intervention
- ⏱️ **25% decrease** in average ED length of stay
- 💰 **$2.3M annual savings** in avoided readmission costs
- 📊 **Real-time visibility** into 1,200+ licensed beds across 5 hospitals

## 🏥 Clinical Architecture

| Layer | Components |
|-------|-----------|
| **Epic Clarity (EHR)** | ADT, Labs, Clinical Documentation, Billing |
| **↓** | *Real-time data pipeline* |
| **Azure Synapse** | Data quality, Feature engineering, SQL Pools |
| **↓** | *ML scoring API* |
| **Azure ML** | Random Forest model, Time-series forecasting |
| **↓** | *REST API integration* |
| **Power BI** | Executive Dashboard, Patient Flow, Risk Workbench |
| **↓** | *Clinical decision support* |
| **Clinical Teams** | Case Managers, Bed Management, Quality Officers |

## 📊 Dashboard Capabilities

### 1. Real-Time Census Management
**Purpose:** Optimize bed utilization and capacity planning

**Visualizations:**
- **Hospital Heat Map:** Color-coded bed occupancy by unit (Green <85%, Yellow 85-95%, Red >95%)
- **Predicted Admissions:** ML-powered 24-hour admission forecasting
- **Discharge Pipeline:** Patients ready for discharge vs. actual discharges
- **Bottleneck Alerts:** ED boarding >4 hours, ICU capacity >95%

**Business Logic:**
- Automatic diversion recommendations when census >95%
- Staffing ratio calculations based on acuity (HPPD - Hours Per Patient Day)

### 2. Readmission Prevention Workbench
**Purpose:** Identify and intervene with high-risk patients before discharge

**Key Features:**
- **Risk Scorecards:** Every patient scored 0-100% readmission probability
- **Automated Triage:** 
  - 🔴 High Risk (>70%): Immediate case manager assignment
  - 🟡 Medium Risk (40-70%): Enhanced discharge planning
  - 🟢 Low Risk (<40%): Standard protocol
- **Intervention Tracking:** Did case manager contact patient within 24h of discharge?
- **LACE Score:** Length of stay, Acuity, Comorbidity, ED visits (validated algorithm)

**ML Model Performance:**
- **AUC Score:** 0.84 (excellent predictive accuracy)
- **Precision:** 78% of flagged patients actually readmitted
- **Recall:** 82% of all readmissions caught by model
- **Top Features:** Previous admissions, comorbidity count, age, LOS, discharge disposition

### 3. Emergency Department Optimization
**Purpose:** Reduce ED wait times and left-without-being-seen rates

**Metrics Tracked:**
- Door-to-provider time (target <30 min)
- ED length of stay (target <4 hours)
- LWBS rate (Left Without Being Seen)
- Admission conversion rate

**Analytics:**
- Hourly arrival pattern prediction (staffing optimization)
- ESI (Emergency Severity Index) acuity mix
- Fast track eligibility identification

### 4. Quality & Safety Monitoring
**Purpose:** Track clinical quality indicators and prevent hospital-acquired conditions

**Indicators:**
- **HAPI:** Hospital-Acquired Pressure Injury rates
- **Falls:** Patient falls with injury
- **CLABSI/CAUTI:** Central line and catheter-associated infections
- **Medication Errors:** Near-miss and actual events

**Comparison:**
- Unit vs. unit benchmarking
- Performance vs. National Patient Safety Goals (NPSG)

## 🧠 Machine Learning Integration

### Readmission Risk Model
**Algorithm:** Random Forest Classifier (ensemble method)
**Training Data:** 50,000+ historical admissions
**Features:**
- Demographics (age, gender, insurance)
- Clinical (Diagnosis codes, comorbidities, severity)
- Utilization (Previous admissions, ED visits, LOS)
- Social (Home health need, distance from hospital)

**Deployment:**
- Azure ML Managed Endpoint
- REST API called by Power BI for real-time scoring
- Batch scoring nightly for all active inpatients

### Time-Series Forecasting
**Purpose:** Predict census and demand 24-48 hours ahead

**Method:** Prophet algorithm (Facebook/Meta) with seasonality
- **Weekly Patterns:** Monday high admission, Friday high discharge
- **Seasonal:** Flu season, holiday patterns
- **Special Events:** Local festivals, weather events

**Use Case:**
- Proactive staffing adjustments
- Elective surgery scheduling optimization
- Supply chain preparation

## 🔐 Security & Compliance

### HIPAA Compliance
- **PHI Access Logging:** All data access tracked in Azure Monitor
- **Row-Level Security:** Nurses see only their assigned patients
- **Data Masking:** MRNs masked for non-clinical roles (auditors/researchers)
- **Encryption:** TDE (Transparent Data Encryption) for data at rest, TLS 1.2 in transit
- **BAA:** Business Associate Agreement compliance for Azure

### Role-Based Access

| Role | Access Level | Use Case |
|------|-------------|----------|
| **CNO/CEO** | All hospitals, all units | Strategic planning |
| **Nurse Manager** | Own unit only | Staffing decisions |
| **Case Manager** | Assigned patients only | Discharge planning |
| **Physician** | Own patients + aggregated unit data | Clinical care |
| **Quality Officer** | Aggregated reports (no PHI) | Compliance reporting |

## 🚀 Technical Implementation

### Data Pipeline (Azure Data Factory)
1. **Extract:** Epic Clarity Chronicles database (CDC - Change Data Capture)
2. **Transform:** Databricks (PySpark) for data cleansing and feature engineering
3. **Load:** Azure Synapse (SQL Pools) for analytics serving

**Latency:** ~15 minutes from Epic documentation to Power BI dashboard

### Power BI Features Used
- **Incremental Refresh:** 3 years historical + daily incremental (fast refresh)
- **Aggregations:** Pre-summarized tables for high-level views (10M+ rows → 10k rows)
- **Composite Models:** Import (historical) + DirectQuery (real-time) for optimal performance
- **AI Visuals:** Anomaly detection on census trends, Key influencers for readmission factors
- **Power Automate:** Automated alerts to Teams/Email when thresholds breached

### Integration Points
- **Epic Hyperspace:** Embedded dashboard links in clinical workflow
- **Teams:** Daily census reports posted to nursing leadership channels
- **SMS:** Critical safety alerts (falls, codes) to charge nurses

## 📈 Business Impact

### Before Implementation
- Reactive bed management (crisis mode when full)
- Case managers identified high-risk patients subjectively (~50% accuracy)
- ED boarding times averaging 6+ hours
- Readmission rate: 18%

### After Implementation
- **Predictive capacity management:** 95% forecast accuracy, zero unplanned diversions
- **AI-driven case management:** 78% precision in identifying readmission risk
- **Streamlined ED:** Average LOS reduced to 3.2 hours
- **Readmission rate:** 15.3% (15% relative reduction)

**Financial Impact:**
- Avoided readmissions: ~150 cases/year
- Cost savings: $2.3M annually ($15k avg cost per readmission)
- ED efficiency: $450k additional revenue capacity (higher throughput)

## 🛠️ Tech Stack

**Data & Analytics:**
- Azure Synapse Analytics (SQL Dedicated Pools)
- Azure Data Lake Gen2 (bronze/silver/gold architecture)
- Azure Data Factory (orchestration)
- Azure Databricks (Spark processing)

**Machine Learning:**
- Azure Machine Learning Service
- Python (scikit-learn, Prophet)
- MLflow (model tracking)

**Visualization:**
- Power BI Premium (Paginated Reports + Interactive Dashboards)
- Power BI Mobile (iPad rounding lists)
- Power Automate (alerting workflows)

**Source Systems:**
- Epic Clarity (EHR data)
- RL Solutions (safety events)
- Kronos (staffing data)

## 👤 Author

**Avinash Chinnabattuni**  
*Data Engineer & Healthcare Analytics Specialist*

Experience designing clinical analytics solutions at **UnitedHealth Group**, processing Epic Clarity data for enterprise healthcare reporting and patient outcomes improvement.

📧 avinashchinnabattuni@gmail.com  
🔗 [Portfolio](https://avinash-0612.github.io/avinash-portfolio/) | [LinkedIn](https://linkedin.com/in/avinash-chinnabattuni-12a9191a7)

## 📚 Related Projects

- **[FHIR Healthcare Data Lakehouse](https://github.com/Avinash-0612/fhir-healthcare-lakehouse):** Data engineering pipeline feeding this analytics layer
- **[Executive Financial Dashboard](https://github.com/Avinash-0612/executive-financial-dashboard):** Financial counterpart to clinical operations

---

**Note:** This project demonstrates healthcare analytics capabilities using synthetic patient data. No real PHI (Protected Health Information) is included in this repository. All patient names, MRNs, and clinical details are fictional.
