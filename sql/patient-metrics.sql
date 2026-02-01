-- ============================================================================
-- CLINICAL OPERATIONS METRICS
-- SQL Views for Power BI Dashboard
-- Source: Epic Clarity / Azure Synapse
-- Author: Avinash Chinnabattuni
-- ============================================================================

-- 1. Current Census View (Real-time bed occupancy)
CREATE OR ALTER VIEW vw_CurrentCensus AS
SELECT 
    u.UnitName,
    u.UnitType,
    COUNT(*) AS OccupiedBeds,
    u.LicensedBeds,
    ROUND(COUNT(*) * 100.0 / u.LicensedBeds, 1) AS OccupancyPercentage,
    CASE 
        WHEN COUNT(*) * 100.0 / u.LicensedBeds > 95 THEN 'Critical'
        WHEN COUNT(*) * 100.0 / u.LicensedBeds > 85 THEN 'High'
        ELSE 'Normal'
    END AS CapacityStatus
FROM 
    ADT_CurrentCensus c
    INNER JOIN DimUnit u ON c.UnitID = u.UnitID
WHERE 
    c.AdmissionStatus = 'Admitted'
GROUP BY 
    u.UnitName, u.UnitType, u.LicensedBeds;

-- 2. Length of Stay Analysis
CREATE OR ALTER VIEW vw_LengthOfStay AS
SELECT 
    p.PatientID,
    p.MRN,
    p.Name,
    a.AdmissionDate,
    a.DischargeDate,
    DATEDIFF(day, a.AdmissionDate, COALESCE(a.DischargeDate, GETDATE())) AS LOS_Days,
    a.UnitName,
    a.AttendingPhysician,
    drg.DRGCode,
    drg.DRGDescription,
    CASE 
        WHEN DATEDIFF(day, a.AdmissionDate, COALESCE(a.DischargeDate, GETDATE())) > drg.AvgLOSTarget * 1.5 THEN 'Extended'
        WHEN DATEDIFF(day, a.AdmissionDate, COALESCE(a.DischargeDate, GETDATE())) > drg.AvgLOSTarget THEN 'Above Target'
        ELSE 'On Target'
    END AS LOS_Status
FROM 
    FactAdmissions a
    INNER JOIN DimPatient p ON a.PatientID = p.PatientID
    LEFT JOIN DimDRG drg ON a.DRGID = drg.DRGID
WHERE 
    a.AdmissionDate >= DATEADD(year, -1, GETDATE());

-- 3. Readmission Risk Score Integration (ML Model Results)
CREATE OR ALTER VIEW vw_ReadmissionRisk AS
SELECT 
    p.PatientID,
    p.MRN,
    p.Name,
    p.Age,
    a.AdmissionDate,
    a.DischargeDate,
    ml.ReadmissionProbability,
    ml.RiskCategory,
    ml.RecommendedAction,
    CASE 
        WHEN ml.ReadmissionProbability >= 0.7 THEN '🔴 High'
        WHEN ml.ReadmissionProbability >= 0.4 THEN '🟡 Medium'
        ELSE '🟢 Low'
    END AS RiskIndicator,
    cm.CaseManagerName,
    cm.InterventionStatus,
    cm.ContactDate
FROM 
    FactAdmissions a
    INNER JOIN DimPatient p ON a.PatientID = p.PatientID
    INNER JOIN ML_ReadmissionScores ml ON a.AdmissionID = ml.AdmissionID
    LEFT JOIN CaseManagementLog cm ON a.AdmissionID = cm.AdmissionID
WHERE 
    a.DischargeDate IS NULL OR a.DischargeDate >= DATEADD(day, -30, GETDATE());

-- 4. ED Throughput Metrics
CREATE OR ALTER VIEW vw_EDMetrics AS
SELECT 
    CAST(e.ArrivalTime AS DATE) AS Date,
    DATEPART(hour, e.ArrivalTime) AS HourOfDay,
    COUNT(*) AS TotalArrivals,
    AVG(DATEDIFF(minute, e.ArrivalTime, e.RoomAssignmentTime)) AS AvgTimeToBed_Minutes,
    AVG(DATEDIFF(minute, e.ArrivalTime, e.ProviderAssignmentTime)) AS AvgTimeToProvider_Minutes,
    AVG(DATEDIFF(minute, e.ArrivalTime, e.DepartureTime)) AS AvgLOS_Minutes,
    SUM(CASE WHEN e.AdmissionDisposition = 'Admitted' THEN 1 ELSE 0 END) AS AdmittedCount,
    SUM(CASE WHEN e.AdmissionDisposition = 'Discharged' THEN 1 ELSE 0 END) AS DischargedCount,
    AVG(CASE WHEN e.ELevel = 'ESI 1' THEN 1 ELSE 0 END) AS CriticalPatientPct
FROM 
    FactEDVisits e
WHERE 
    e.ArrivalTime >= DATEADD(day, -30, GETDATE())
GROUP BY 
    CAST(e.ArrivalTime AS DATE),
    DATEPART(hour, e.ArrivalTime);

-- 5. Discharge Planning Efficiency
CREATE OR ALTER VIEW vw_DischargeEfficiency AS
SELECT 
    a.UnitName,
    COUNT(*) AS TotalDischarges,
    AVG(DATEDIFF(hour, a.DischargeOrderTime, a.ActualDischargeTime)) AS AvgDischargeLag_Hours,
    SUM(CASE WHEN a.DischargeDelayReason IS NOT NULL THEN 1 ELSE 0 END) AS DelayedDischarges,
    ROUND(
        SUM(CASE WHEN a.DischargeDelayReason IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 
        1
    ) AS DelayPercentage,
    a.DischargeDelayReason,
    COUNT(*) AS DelayCount
FROM 
    FactAdmissions a
WHERE 
    a.DischargeDate >= DATEADD(month, -1, GETDATE())
GROUP BY 
    a.UnitName, a.DischargeDelayReason;

-- 6. Quality Indicators (HAPI, Falls, CLABSI)
CREATE OR ALTER VIEW vw_QualityIndicators AS
SELECT 
    qi.EventDate,
    qi.UnitName,
    qi.EventType,
    COUNT(*) AS EventCount,
    u.LicensedBeds,
    ROUND(COUNT(*) * 1000.0 / u.LicensedBeds, 2) AS RatePer1000BedDays,
    CASE qi.EventType
        WHEN 'Fall' THEN 'Safety'
        WHEN 'HAPI' THEN 'Skin Integrity'
        WHEN 'CLABSI' THEN 'Infection Control'
        WHEN 'CAUTI' THEN 'Infection Control'
        WHEN 'MedError' THEN 'Medication Safety'
    END AS Category
FROM 
    FactQualityEvents qi
    INNER JOIN DimUnit u ON qi.UnitID = u.UnitID
WHERE 
    qi.EventDate >= DATEADD(quarter, -1, GETDATE())
GROUP BY 
    qi.EventDate, qi.UnitName, qi.EventType, u.LicensedBeds;

-- ============================================================================
-- Power BI Incremental Refresh Support
-- ============================================================================

-- Index for faster filtering
CREATE INDEX IX_FactAdmissions_AdmissionDate ON FactAdmissions(AdmissionDate);
CREATE INDEX IX_FactEDVisits_ArrivalTime ON FactEDVisits(ArrivalTime);
