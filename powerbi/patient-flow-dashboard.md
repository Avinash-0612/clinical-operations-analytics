# Clinical Operations Dashboard Design

## Overview
Real-time patient flow analytics for hospital operations, integrating Epic Clarity data with predictive models for capacity planning and readmission prevention.

## Dashboard Pages

### 1. Executive Command Center
**Audience:** CNO (Chief Nursing Officer), Hospital CEO

**Key Visuals:**
- **Bed Occupancy Gauge:** Current census vs licensed beds (color-coded: Green &lt;85%, Yellow 85-95%, Red &gt;95%)
- **Length of Stay (LOS) Trend:** Average LOS by unit, with target lines
- **Readmission Risk Alerts:** High-risk patients flagged by ML model (&gt;0.7 probability)
- **Financial Impact:** Cost of avoidable readmissions, revenue at risk

**Data Refresh:** Real-time (DirectQuery to Azure Synapse)

### 2. Patient Flow Tracker
**Audience:** Bed Management, Nursing Supervisors

**Key Features:**
- **Sankey Diagram:** Patient journey (ED → Admission → Unit → Discharge)
- **Bottleneck Identification:** ED wait times, bed turnover rates
- **Discharge Planning:** Predicted discharge dates vs actual
- **Transportation Status:** Pending discharges awaiting transport

**Interactivity:**
- Drill-through from hospital level → Unit level → Individual patient
- Filter by unit, attending physician, admission type

### 3. Capacity Planning
**Audience:** Capacity Management, Nursing Leadership

**Predictive Analytics:**
- **24-Hour Census Forecast:** Time-series forecasting (Azure ML)
- **Admission Prediction:** Expected admissions by hour (based on historical patterns + seasonality)
- **Staffing Recommendations:** Optimal nurse-to-patient ratios based on acuity

**Metrics:**
- Current Occupancy: 87%
- Predicted Tonight: 94% (High - consider diversion)
- Discharge Potential: 12 patients (could free up beds)

### 4. Readmission Risk Stratification
**Audience:** Case Management, Care Coordination

**ML Integration:**
- **Risk Scorecards:** Each inpatient ranked 0-100% readmission risk
- **Risk Factors:** Top contributing factors (age, comorbidities, previous admissions)
- **Intervention Tracking:** Did case manager contact high-risk patient? (documentation required)

**Color Coding:**
- 🔴 High Risk (&gt;70%): Immediate intervention required
- 🟡 Medium Risk (40-70%): Discharge planning protocol
- 🟢 Low Risk (&lt;40%): Standard discharge process

### 5. Quality & Safety Metrics
**Audience:** Quality Officers, Physicians

**Visuals:**
- **Fall Risk Assessment:** Compliance with hourly rounding
- **Medication Administration:** On-time med pass percentage
- **Infection Control:** CLABSI/CAUTI rates by unit
- **Pressure Injuries:** Hospital-acquired pressure injury (HAPI) rates

## Row-Level Security (RLS)

### Roles

**1. Chief Nursing Officer (CNO)**
- Access: All hospitals, all units
- Filter: `TRUE()`

**2. Nurse Manager**
- Access: Own unit only
- Filter: `[UnitID] = LOOKUPVALUE(User[UnitID], User[Email], USERNAME())`

**3. Physician**
- Access: Own patients only
- Filter: `[AttendingPhysicianID] = USERNAME()`

**4. Case Manager**
- Access: Assigned patients only
- Filter: `[CaseManagerID] = USERNAME()`

**5. Quality Auditor**
- Access: Aggregated data only (no patient names)
- Filter: `[DataLevel] = "Aggregated"`
- OLS: Cannot see PatientName, MRN, DOB columns

## Alert Configuration

### Automated Alerts (Power Automate Integration)

**High Priority Alerts:**
- ED wait time &gt;4 hours → Alert to ED Director + CNO
- Unit occupancy &gt;95% → Alert to Bed Management
- Readmission risk score &gt;0.8 → Alert to Case Manager
- Patient fall → Immediate alert to charge nurse + safety officer

**Delivery Methods:**
- Power BI dashboard notifications
- Teams messages
- Email alerts
- SMS for critical safety events

## Mobile Layout

**iPad Optimized Views:**
- ED snapshot (current wait times, bed availability)
- Unit rounding lists (patient location, LOS, discharge status)
- Executive summary (KPIs only)

**Voice Q&A Enabled:**
- "Show me ICU occupancy"
- "Which patients are high readmission risk"
- "Compare LOS this month vs last month"

## Data Sources Integration

| System | Data Type | Refresh |
|--------|-----------|---------|
| Epic Clarity | ADT, Labs, Meds | Hourly |
| Azure ML API | Risk Scores | Real-time |
| Bed Management System | Bed status | Real-time |
| Nurse Call System | Response times | 15 minutes |
| Case Management | Documentation status | Hourly |

## Screenshots Reference

*Note: In actual portfolio presentation, include 3-4 high-quality screenshots showing:*
- Bed occupancy heat map
- Patient flow Sankey diagram
- Readmission risk scorecards
- Mobile view on tablet
