from pyhealth.data import Patient, Visit
from datetime import datetime

# ------------------------------------------------------------
def drug_recommendation_mimic4_fn(patient: Patient):
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs      = [d[:4] for d in visit.get_code_list(table="prescriptions")]

        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        visit_date = visit.encounter_time.date().isoformat()     # ★ 加入

        samples.append({
            "visit_id"   : visit.visit_id,
            "patient_id" : patient.patient_id,
            "visit_date" : visit_date,                           # ★
            "conditions" : conditions,
            "procedures" : procedures,
            "drugs"      : drugs,
            "drugs_all"  : drugs,
        })

    if len(samples) < 2:
        return []

    # accumulate history …
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"]  = [samples[0]["drugs_all"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i-1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i-1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs_all"]  = samples[i-1]["drugs_all"]  + [samples[i]["drugs_all"]]
    return samples

# ------------------------------------------------------------
def mortality_prediction_mimic4_fn(patient: Patient):
    samples = []
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit   : Visit = patient[i + 1]

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs      = visit.get_code_list(table="prescriptions")
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        visit_date = visit.encounter_time.date().isoformat()     # ★

        label = int(next_visit.discharge_status) if next_visit.discharge_status in (0, 1) else 0
        samples.append({
            "visit_id"   : visit.visit_id,
            "patient_id" : patient.patient_id,
            "visit_date" : visit_date,                           # ★
            "conditions" : conditions,
            "procedures" : procedures,
            "drugs"      : drugs,
            "label"      : label,
        })

    if not samples:
        return []

    # accumulate history …
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"]      = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i-1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i-1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"]      = samples[i-1]["drugs"]      + [samples[i]["drugs"]]
    return samples

# ------------------------------------------------------------
def readmission_prediction_mimic4_fn(patient: Patient, time_window: int = 15):
    samples = []
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit  : Visit = patient[i + 1]

        td   = (next_visit.encounter_time - visit.encounter_time).days
        label = 1 if td < time_window else 0

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs      = visit.get_code_list(table="prescriptions")
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        visit_date = visit.encounter_time.date().isoformat()     # ★

        samples.append({
            "visit_id"   : visit.visit_id,
            "patient_id" : patient.patient_id,
            "visit_date" : visit_date,                           # ★
            "conditions" : conditions,
            "procedures" : procedures,
            "drugs"      : drugs,
            "label"      : label,
        })

    if not samples:
        return []

    # accumulate history …
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"]      = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i-1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i-1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"]      = samples[i-1]["drugs"]      + [samples[i]["drugs"]]
    return samples

# ------------------------------------------------------------
def length_of_stay_prediction_mimic4_fn(patient: Patient):
    samples = []
    for visit in patient:
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs      = visit.get_code_list(table="prescriptions")
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        days       = (visit.discharge_time - visit.encounter_time).days
        label      = categorize_los(days)
        visit_date = visit.encounter_time.date().isoformat()     # ★

        samples.append({
            "visit_id"   : visit.visit_id,
            "patient_id" : patient.patient_id,
            "visit_date" : visit_date,                           # ★
            "conditions" : conditions,
            "procedures" : procedures,
            "drugs"      : drugs,
            "label"      : label,
        })

    if not samples:
        return []

    # accumulate history …
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"]      = [samples[0]["drugs"]]
    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i-1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i-1]["procedures"] + [samples[i]["procedures"]]
        samples[i]["drugs"]      = samples[i-1]["drugs"]      + [samples[i]["drugs"]]
    return samples

